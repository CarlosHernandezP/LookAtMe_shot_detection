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

COURT_COLS_ALL = ["court_y", "court_x", "dist_to_near_baseline"]


def load_dataset_with_court(data_dir, min_samples, court_cols, exclude_token=None, only_token=None):
    """
    Like load_dataset_wall_flat but appends selected court columns per frame to the
    base 41 features. court_cols is an ordered subset of COURT_COLS_ALL. Court values
    are z-normless here (the trainer z-norms everything). Missing court -> 0.
    exclude_token / only_token filter files by substring (for leave-one-match-out).
    Returns (sequences (N,30,41+k), labels).
    """
    import glob
    import os
    from collections import Counter

    import pandas as pd
    from shot_detector.ball_features import get_ball_feature_names
    from shot_detector.pose_io import N_FEATURES_RAW, load_pose_csv, pad_or_trim_sequence
    from shot_detector.shot_mapper import DEFAULT_SHOT_MAPPING, extract_shot_type_from_filename, map_shot_to_class

    seq_len = 30
    n_ball = len(get_ball_feature_names())
    files = sorted(glob.glob(os.path.join(data_dir, "*_pose.csv")))
    sequences, labels = [], []
    for path in files:
        bn = os.path.basename(path)
        if exclude_token and exclude_token in bn:
            continue
        if only_token and only_token not in bn:
            continue
        raw = extract_shot_type_from_filename(path)
        if not raw:
            continue
        mapped = map_shot_to_class(raw, DEFAULT_SHOT_MAPPING)
        if mapped is None:
            continue
        feats = load_pose_csv(path)
        if feats is None or feats.shape[1] < N_FEATURES_RAW:
            continue
        feats = pad_or_trim_sequence(feats[:, :N_FEATURES_RAW])
        if mapped != "idle" and np.all(np.isnan(feats[:, -n_ball:])):
            continue
        if court_cols:
            df = pd.read_csv(path, usecols=lambda c: c in court_cols)
            court = df[court_cols].to_numpy(dtype=np.float32)
            court = pad_or_trim_sequence(court)
            court = np.nan_to_num(court, nan=0.0)
            feats = np.concatenate([feats, court], axis=1)
        sequences.append(feats)
        labels.append(mapped)
    if not sequences:
        raise ValueError(f"No usable pose CSVs in {data_dir}")
    sequences = np.array(sequences)
    labels = np.array(labels)
    counts = Counter(labels)
    keep = {c for c, n in counts.items() if n >= min_samples}
    mask = np.array([l in keep for l in labels])
    sequences, labels = sequences[mask], labels[mask]
    print(f"Loaded {len(sequences)} seqs, court_cols={court_cols} -> feat dim {sequences.shape[2]}")
    for cls, cnt in sorted(Counter(labels).items(), key=lambda x: -x[1]):
        print(f"  {cls:<12} {cnt:>5}")
    return sequences, labels


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


def augment_batch(xb: torch.Tensor, noise_std: float = 0.02,
                  resample_range: float = 0.1, frame_dropout: float = 0.05) -> torch.Tensor:
    """
    On-the-fly augmentation on z-normed sequences (B, T, F):
    - random temporal resampling (speed 1±resample_range, linear interp back to T)
    - gaussian keypoint noise
    - random frame dropout (frame copied from previous frame)
    """
    B, T, F = xb.shape
    dev = xb.device
    # speed perturbation per sample
    rates = 1.0 + (torch.rand(B, device=dev) * 2 - 1) * resample_range
    base = torch.arange(T, device=dev, dtype=torch.float32)
    out = torch.empty_like(xb)
    for b in range(B):
        src = torch.clamp(base * rates[b], 0, T - 1)
        lo = src.floor().long()
        hi = torch.clamp(lo + 1, max=T - 1)
        frac = (src - lo.float()).unsqueeze(1)
        out[b] = xb[b, lo] * (1 - frac) + xb[b, hi] * frac
    # frame dropout: replace frame t with t-1
    drop = torch.rand(B, T, device=dev) < frame_dropout
    drop[:, 0] = False
    idx = torch.arange(T, device=dev).expand(B, T).clone()
    idx[drop] = idx[drop] - 1
    out = out.gather(1, idx.unsqueeze(2).expand(B, T, F))
    # additive noise
    out = out + torch.randn_like(out) * noise_std
    return out


def train_one_fold(
    X_tr, y_tr, w_tr, X_va, y_va, arch: str, n_classes: int,
    epochs: int, batch_size: int, lr: float, patience: int, seed: int,
    augment: bool = False, label_smoothing: float = 0.0,
    channels: int = 96, kernel_size: int = 3, dropout: float = 0.2,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if arch == "tcn":
        model = TCN(X_tr.shape[2], n_classes, channels=channels,
                    kernel_size=kernel_size, dropout=dropout).to(DEVICE)
    else:
        model = make_model(arch, X_tr.shape[2], n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)

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
            if augment:
                xb = augment_batch(xb)
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


def run_heldout_ablation(args, court_cols):
    """
    Train on all matches except --heldout-token, then test on the held-out
    match's annotated shot windows. Reports per-class recall (esp. serve).
    This isolates whether court features improve shot discrimination on an
    unseen court, sidestepping the FP/precision question (annotations are
    incomplete; recall is the trustworthy axis).
    """
    tok = args.heldout_token
    Xtr_s, ytr = load_dataset_with_court(args.data_dir, args.min_samples, court_cols, exclude_token=tok)
    Xte_s, yte = load_dataset_with_court(args.data_dir, 1, court_cols, only_token=tok)
    Xtr = np.nan_to_num(Xtr_s.astype(np.float32), nan=0.0)
    Xte = np.nan_to_num(Xte_s.astype(np.float32), nan=0.0)

    le = LabelEncoder()
    ytr_e = le.fit_transform(ytr)
    classes = le.classes_.tolist()
    # held-out labels may include classes; map, drop any unseen
    keep = np.array([l in classes for l in yte])
    Xte, yte = Xte[keep], yte[keep]
    yte_e = np.array([classes.index(l) for l in yte])

    mu = Xtr.reshape(-1, Xtr.shape[2]).mean(axis=0)
    sd = Xtr.reshape(-1, Xtr.shape[2]).std(axis=0) + 1e-6
    Xtr_n, Xte_n = (Xtr - mu) / sd, (Xte - mu) / sd
    sw = compute_sample_weights(ytr_e, le.classes_, args.serve_weight_mult,
                                args.non_idle_weight_mult, weight_scheme=args.weight_scheme)
    model, _, _ = train_one_fold(
        Xtr_n, ytr_e, sw, Xte_n, yte_e, args.arch, len(classes),
        args.epochs, args.batch_size, args.lr, args.patience, seed=4242,
        augment=args.augment, label_smoothing=args.label_smoothing,
        channels=args.channels, kernel_size=args.kernel_size, dropout=args.dropout,
    )
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte_n, device=DEVICE)).argmax(1).cpu().numpy()

    rep = classification_report(yte_e, pred, labels=range(len(classes)),
                                target_names=classes, zero_division=0, output_dict=True)
    out = {"heldout": tok, "court_cols": court_cols, "feat_dim": int(Xtr.shape[2]),
           "n_test": int(len(yte_e)), "per_class": {}}
    print(f"\n=== HELD-OUT {tok}  court={court_cols or 'none'}  featdim={Xtr.shape[2]}  n_test={len(yte_e)} ===")
    print(f"{'class':<12} {'recall':>7} {'prec':>7} {'n':>5}")
    for c in classes:
        r = rep[c]
        out["per_class"][c] = {"recall": r["recall"], "precision": r["precision"], "n": int(r["support"])}
        mark = "  <== SERVE" if c == "serve" else ""
        print(f"{c:<12} {r['recall']:>7.3f} {r['precision']:>7.3f} {int(r['support']):>5}{mark}")
    out["accuracy"] = float(rep["accuracy"])
    out["serve_recall"] = rep["serve"]["recall"] if "serve" in rep else None
    print(f"overall accuracy {rep['accuracy']:.3f}  |  serve recall {out['serve_recall']}")
    Path(args.output_dir, f"ablation_{tok}_{'-'.join(court_cols) or 'base'}.json").write_text(json.dumps(out, indent=2))


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
    ap.add_argument("--augment", action="store_true",
                    help="On-the-fly augmentation: speed perturbation, frame dropout, gaussian noise")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--channels", type=int, default=96, help="TCN channels")
    ap.add_argument("--kernel-size", type=int, default=3, help="TCN kernel size")
    ap.add_argument("--dropout", type=float, default=0.2, help="TCN dropout")
    ap.add_argument("--save-model", action="store_true",
                    help="After CV: refit on ALL data (with an internal early-stop split) and export weights + z-norm stats + meta")
    ap.add_argument("--court-features", type=str, default="",
                    help="Comma list of court columns to append: court_y, court_x, dist_to_near_baseline")
    ap.add_argument("--heldout-token", type=str, default=None,
                    help="If set: train on data excluding this match token, then report per-class recall on that match's annotated shot windows (ablation mode)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    court_cols = [c.strip() for c in args.court_features.split(",") if c.strip()]

    if args.heldout_token:
        run_heldout_ablation(args, court_cols)
        return

    if court_cols:
        seqs, lbls = load_dataset_with_court(args.data_dir, args.min_samples, court_cols)
    else:
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
            augment=args.augment, label_smoothing=args.label_smoothing,
            channels=args.channels, kernel_size=args.kernel_size, dropout=args.dropout,
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
            "augment": args.augment, "label_smoothing": args.label_smoothing,
            "channels": args.channels, "kernel_size": args.kernel_size,
            "dropout": args.dropout,
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

    if args.save_model:
        from sklearn.model_selection import train_test_split

        # z-norm on ALL data for the exported model
        mu = X.reshape(-1, X.shape[2]).mean(axis=0)
        sd = X.reshape(-1, X.shape[2]).std(axis=0) + 1e-6
        Xn = (X - mu) / sd
        # internal stratified split only for early stopping
        tr, va = train_test_split(np.arange(len(y)), test_size=0.1, stratify=y, random_state=42)
        sw = compute_sample_weights(
            y[tr], le.classes_, args.serve_weight_mult, args.non_idle_weight_mult,
            weight_scheme=args.weight_scheme,
        )
        model, _, best_f1 = train_one_fold(
            Xn[tr], y[tr], sw, Xn[va], y[va], args.arch, n_classes,
            args.epochs, args.batch_size, args.lr, args.patience, seed=4242,
            augment=args.augment, label_smoothing=args.label_smoothing,
            channels=args.channels, kernel_size=args.kernel_size, dropout=args.dropout,
        )
        bundle = {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "arch": args.arch,
            "arch_kwargs": {"channels": args.channels, "kernel_size": args.kernel_size,
                            "dropout": args.dropout} if args.arch == "tcn" else {},
            "n_features": int(X.shape[2]),
            "seq_len": int(X.shape[1]),
            "classes": class_names,
            "norm_mean": mu.tolist(),
            "norm_std": sd.tolist(),
            "early_stop_val_f1": float(best_f1),
            "cv_report": str(path.name),
        }
        mpath = out_dir / f"seq_{args.arch}_model.pt"
        torch.save(bundle, mpath)
        print(f"Saved model bundle: {mpath} (internal-split val f1={best_f1:.4f})")


if __name__ == "__main__":
    main()
