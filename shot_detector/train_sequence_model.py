"""
Train sequence models (TCN / Transformer) on raw 30x41 pose+ball sequences.

Unlike train_shot_model.py there is NO temporal aggregation and NO top-K
feature selection: models consume the (SEQUENCE_LENGTH, N_FEATURES_RAW)
window directly. NaNs (missing ball / missing frames) are zero-filled; the
ball_visible column already encodes ball presence.

Same dataset loader, label mapping, stratified 5-fold protocol and metrics
as the XGB path, so results are directly comparable.

Usage:
    uv run python shot_detector/train_sequence_model.py \
        --data-dir shot_detector/data/extract_intermediate_poses \
        --output-dir shot_detector/runs/seq_tcn --arch tcn --cv-folds 5
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from collections import Counter
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from shot_detector.train_shot_model import (
    compute_sample_weights,
    fold_metrics_dict,
    load_dataset_wall_flat,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TCN(nn.Module):
    """Dilated temporal conv stack -> global average pool -> linear head."""

    def __init__(self, n_features: int, n_classes: int, channels: int = 96,
                 levels: int = 4, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        c_in = n_features
        for i in range(levels):
            dilation = 2 ** i
            pad = (kernel_size - 1) * dilation // 2
            layers += [
                nn.Conv1d(c_in, channels, kernel_size, padding=pad, dilation=dilation),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            c_in = channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels, n_classes)

    def forward(self, x):  # x: (B, T, F)
        h = self.tcn(x.transpose(1, 2))  # (B, C, T)
        return self.head(h.mean(dim=2))


class TransformerClassifier(nn.Module):
    """Linear proj -> learned positional emb -> TransformerEncoder -> mean pool."""

    def __init__(self, n_features: int, n_classes: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 3, dropout: float = 0.2,
                 seq_len: int = 30):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):  # (B, T, F)
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


def make_model(arch: str, n_features: int, n_classes: int) -> nn.Module:
    if arch == "tcn":
        return TCN(n_features, n_classes)
    if arch == "transformer":
        return TransformerClassifier(n_features, n_classes)
    raise ValueError(arch)


def train_one_fold(
    X_tr, y_tr, w_tr, X_va, y_va, arch: str, n_classes: int,
    epochs: int, batch_size: int, lr: float, patience: int, seed: int,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(arch, X_tr.shape[2], n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(reduction="none")

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    wt = torch.tensor(w_tr, dtype=torch.float32)
    Xv = torch.tensor(X_va, dtype=torch.float32).to(DEVICE)

    ds = torch.utils.data.TensorDataset(Xt, yt, wt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_f1, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb, wb in dl:
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            opt.zero_grad()
            loss = (crit(model(xb), yb) * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xv).argmax(dim=1).cpu().numpy()
        f1 = f1_score(y_va, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(Xv), dim=1).cpu().numpy()
    return model, proba, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--arch", choices=["tcn", "transformer"], required=True)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--weight-scheme", type=str, default="max_inverse",
                    choices=["max_inverse", "sklearn_balanced"])
    ap.add_argument("--non-idle-weight-mult", type=float, default=1.0)
    ap.add_argument("--serve-weight-mult", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs, lbls = load_dataset_wall_flat(args.data_dir, args.min_samples)
    X = np.nan_to_num(seqs.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    le = LabelEncoder()
    y = le.fit_transform(lbls)
    class_names = le.classes_.tolist()
    n_classes = len(class_names)
    print(f"X={X.shape} classes={class_names}")

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    fold_rows: List[Dict] = []
    proba_oof = np.zeros((len(y), n_classes))
    for fi, (tr, va) in enumerate(cv.split(X, y)):
        # per-fold z-norm fitted on train
        mu = X[tr].reshape(-1, X.shape[2]).mean(axis=0)
        sd = X[tr].reshape(-1, X.shape[2]).std(axis=0) + 1e-6
        Xn_tr = (X[tr] - mu) / sd
        Xn_va = (X[va] - mu) / sd
        sw = compute_sample_weights(
            y[tr], le.classes_, args.serve_weight_mult, args.non_idle_weight_mult,
            weight_scheme=args.weight_scheme,
        )
        _, proba_va, best_f1 = train_one_fold(
            Xn_tr, y[tr], sw, Xn_va, y[va], args.arch, n_classes,
            args.epochs, args.batch_size, args.lr, args.patience, seed=42 + fi,
        )
        proba_oof[va] = proba_va
        pred_va = proba_va.argmax(axis=1)
        m = fold_metrics_dict(y[va], pred_va, class_names)
        fold_rows.append(m)
        print(f"fold {fi}: f1_macro={m['f1_macro']:.4f} acc={m['accuracy']:.4f} (best epoch f1={best_f1:.4f})")

    y_oof = proba_oof.argmax(axis=1)
    oof = fold_metrics_dict(y, y_oof, class_names)
    oof["confusion_matrix"] = confusion_matrix(y, y_oof, labels=range(n_classes)).tolist()
    oof["classification_report"] = classification_report(
        y, y_oof, labels=range(n_classes), target_names=class_names,
        zero_division=0, output_dict=True,
    )

    f1s = [r["f1_macro"] for r in fold_rows]
    accs = [r["accuracy"] for r in fold_rows]
    report = {
        "arch": args.arch,
        "cv": {
            "n_folds": args.cv_folds,
            "per_fold_global": fold_rows,
            "f1_macro_mean": float(np.mean(f1s)),
            "f1_macro_std": float(np.std(f1s)),
            "accuracy_mean": float(np.mean(accs)),
            "oof_aggregate": oof,
        },
        "config": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "patience": args.patience, "weight_scheme": args.weight_scheme,
            "non_idle_weight_mult": args.non_idle_weight_mult,
            "serve_weight_mult": args.serve_weight_mult,
            "classes": class_names, "class_counts": dict(Counter(lbls)),
            "input_shape": list(X.shape[1:]),
        },
    }
    path = out_dir / f"seq_{args.arch}_cv_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n=== {args.arch} {args.cv_folds}-fold CV ===")
    print(f"  f1_macro: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}")
    print(f"  accuracy: {np.mean(accs):.4f}")
    print(f"  OOF f1_macro: {oof['f1_macro']:.4f}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
