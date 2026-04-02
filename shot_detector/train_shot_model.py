"""
Train flat XGBoost on pose CSVs (``DEFAULT_SHOT_MAPPING``).

- ``--cv-folds N``: stratified K-fold CV, metrics JSON, plots (mean±std).
- Default without CV: single train/val split.
- By default: trains on full data and exports ``flat_wall_shot_xgb.joblib`` + label encoder.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import csv
import json
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from shot_detector.pose_io import N_FEATURES_RAW, load_pose_csv, pad_or_trim_sequence as _pad_or_trim
from shot_detector.shot_mapper import (
    DEFAULT_SHOT_MAPPING,
    extract_shot_type_from_filename,
    map_shot_to_class,
)
from shot_detector.temporal_features import extract_temporal_features
import glob
import os


def make_xgb(n_classes: int, random_state: int = 42) -> xgb.XGBClassifier:
    params = dict(
        n_estimators=220,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
        verbosity=0,
    )
    if n_classes == 2:
        params["objective"] = "binary:logistic"
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes
    return xgb.XGBClassifier(**params)


def compute_sample_weights(
    y: np.ndarray,
    class_names: np.ndarray,
    serve_mult: float = 1.0,
    non_idle_mult: float = 1.0,
    weight_scheme: str = "max_inverse",
) -> np.ndarray:
    classes_present = np.unique(y)
    if weight_scheme == "sklearn_balanced":
        cw = compute_class_weight("balanced", classes=classes_present, y=y)
        idx_map = {c: i for i, c in enumerate(classes_present)}
        sw = np.array([cw[idx_map[v]] for v in y], dtype=np.float64)
    else:
        counts = np.bincount(y)
        max_count = float(counts.max())
        class_weights = max_count / np.maximum(counts, 1).astype(float)
        sw = class_weights[y].astype(np.float64)

    class_names = np.asarray(class_names)
    if serve_mult != 1.0 and "serve" in class_names:
        serve_idx = int(np.where(class_names == "serve")[0][0])
        sw[y == serve_idx] *= serve_mult

    if non_idle_mult != 1.0 and "idle" in class_names:
        idle_idx = int(np.where(class_names == "idle")[0][0])
        sw[y != idle_idx] *= non_idle_mult

    return sw


def load_dataset_wall_flat(data_dir: str, min_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    files = sorted(glob.glob(os.path.join(data_dir, "*_pose.csv")))
    sequences, labels = [], []

    for path in files:
        raw = extract_shot_type_from_filename(path)
        if not raw:
            continue
        mapped = map_shot_to_class(raw, DEFAULT_SHOT_MAPPING)
        if mapped is None:
            continue
        features = load_pose_csv(path)
        if features is None or features.shape[1] < N_FEATURES_RAW:
            continue
        features = _pad_or_trim(features[:, :N_FEATURES_RAW])
        if mapped != "idle" and np.all(np.isnan(features[:, 27:37])):
            continue
        sequences.append(features)
        labels.append(mapped)

    if not sequences:
        raise ValueError(f"No usable pose CSVs in {data_dir}")

    sequences = np.array(sequences)
    labels = np.array(labels)
    counts = Counter(labels)
    keep = {c for c, n in counts.items() if n >= min_samples}
    dropped = {c: n for c, n in counts.items() if n < min_samples}
    if dropped:
        print(f"Dropping classes with < {min_samples} samples: {dropped}")
    mask = np.array([l in keep for l in labels])
    sequences = sequences[mask]
    labels = labels[mask]

    print(f"Loaded {len(sequences)} sequences (DEFAULT_SHOT_MAPPING)")
    for cls, cnt in sorted(Counter(labels).items(), key=lambda x: -x[1]):
        print(f"  {cls:<12} {cnt:>5}")
    return sequences, labels


def apply_idle_min_prob(
    proba: np.ndarray,
    le: LabelEncoder,
    idle_min_prob: float,
) -> np.ndarray:
    idle_idx = int(np.where(le.classes_ == "idle")[0][0])
    pred = np.argmax(proba, axis=1)
    p_idle = proba[:, idle_idx]
    for i in range(len(pred)):
        if pred[i] == idle_idx and p_idle[i] < idle_min_prob:
            mask = np.ones(le.classes_.shape[0], dtype=bool)
            mask[idle_idx] = False
            sub = np.argmax(proba[i, mask])
            non_idle_indices = np.where(mask)[0]
            pred[i] = non_idle_indices[sub]
    return pred


def fold_metrics_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict:
    rep = classification_report(
        y_true, y_pred, labels=range(len(class_names)),
        target_names=class_names, zero_division=0, output_dict=True
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if "idle" in class_names:
        idle = rep["idle"]
        out["idle_precision"] = float(idle["precision"])
        out["idle_recall"] = float(idle["recall"])
        out["idle_f1"] = float(idle["f1-score"])
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        idle_i = class_names.index("idle")
        out["shot_as_idle_fp"] = int(cm[:, idle_i].sum() - cm[idle_i, idle_i])
        out["idle_as_shot_fn"] = int(cm[idle_i, :].sum() - cm[idle_i, idle_i])
    if "serve" in class_names:
        s = rep["serve"]
        out["serve_precision"] = float(s["precision"])
        out["serve_recall"] = float(s["recall"])
        out["serve_f1"] = float(s["f1-score"])
    if "wall_shot" in class_names:
        w = rep["wall_shot"]
        out["wall_shot_precision"] = float(w["precision"])
        out["wall_shot_recall"] = float(w["recall"])
        out["wall_shot_f1"] = float(w["f1-score"])
    return out


def build_cv(n_folds: int, y: np.ndarray) -> StratifiedKFold:
    min_class = int(np.min(np.bincount(y)))
    folds = max(2, min(n_folds, min_class))
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)


def _aggregate_global_mean_std(fold_rows: List[Dict], keys: List[str]) -> Dict:
    out = {}
    for k in keys:
        vals = [r[k] for r in fold_rows if k in r]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals))
    return out


def _aggregate_per_class_mean_std(
    fold_crs: List[Dict], class_names: List[str]
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for c in class_names:
        precs, recs, f1s = [], [], []
        for cr in fold_crs:
            if c not in cr or not isinstance(cr[c], dict):
                continue
            row = cr[c]
            if "precision" not in row:
                continue
            precs.append(float(row["precision"]))
            recs.append(float(row["recall"]))
            f1s.append(float(row["f1-score"]))
        if not precs:
            continue
        out[c] = {
            "precision_mean": float(np.mean(precs)),
            "precision_std": float(np.std(precs)),
            "recall_mean": float(np.mean(recs)),
            "recall_std": float(np.std(recs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
        }
    return out


def run_cv(
    X: np.ndarray,
    labels: np.ndarray,
    n_folds: int,
    serve_mult: float,
    non_idle_mult: float,
    idle_min_prob: Optional[float],
    weight_scheme: str = "max_inverse",
    random_state: int = 42,
) -> Tuple[Dict, LabelEncoder]:
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = le.classes_.tolist()
    cv = build_cv(n_folds, y)
    n_splits = cv.get_n_splits()

    fold_global: List[Dict] = []
    fold_crs: List[Dict] = []
    y_oof = np.full_like(y, -1)
    proba_oof = np.zeros((len(y), len(class_names)))

    for fi, (tr, va) in enumerate(cv.split(X, y)):
        model = make_xgb(len(class_names), random_state=random_state + fi)
        sw = compute_sample_weights(
            y[tr], le.classes_, serve_mult, non_idle_mult, weight_scheme=weight_scheme
        )
        model.fit(X[tr], y[tr], sample_weight=sw)
        proba_va = model.predict_proba(X[va])
        proba_oof[va] = proba_va

        if idle_min_prob is not None and "idle" in class_names:
            pred_va = apply_idle_min_prob(proba_va, le, idle_min_prob)
        else:
            pred_va = np.argmax(proba_va, axis=1)

        y_oof[va] = pred_va
        fold_global.append(fold_metrics_dict(y[va], pred_va, class_names))
        fold_crs.append(
            classification_report(
                y[va],
                pred_va,
                labels=range(len(class_names)),
                target_names=class_names,
                zero_division=0,
                output_dict=True,
            )
        )

    if idle_min_prob is not None and "idle" in class_names:
        y_oof_final = apply_idle_min_prob(proba_oof, le, idle_min_prob)
    else:
        y_oof_final = np.argmax(proba_oof, axis=1)

    oof_summary = fold_metrics_dict(y, y_oof_final, class_names)
    oof_summary["n_folds"] = n_splits
    oof_summary["confusion_matrix"] = confusion_matrix(
        y, y_oof_final, labels=range(len(class_names))
    ).tolist()
    oof_summary["classification_report"] = classification_report(
        y,
        y_oof_final,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    global_keys = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "idle_precision",
        "idle_recall",
        "idle_f1",
        "shot_as_idle_fp",
        "idle_as_shot_fn",
        "serve_precision",
        "serve_recall",
        "serve_f1",
        "wall_shot_precision",
        "wall_shot_recall",
        "wall_shot_f1",
    ]
    per_fold_mean_std = _aggregate_global_mean_std(
        fold_global, [k for k in global_keys if any(k in r for r in fold_global)]
    )
    per_class_mean_std = _aggregate_per_class_mean_std(fold_crs, class_names)

    report = {
        "cv": {
            "n_folds": n_splits,
            "per_fold_global": fold_global,
            "per_fold_mean_std": per_fold_mean_std,
            "per_class_mean_std": per_class_mean_std,
            "oof_aggregate": oof_summary,
        },
        "config": {
            "serve_weight_mult": serve_mult,
            "non_idle_weight_mult": non_idle_mult,
            "idle_min_prob": idle_min_prob,
            "weight_scheme": weight_scheme,
            "classes": class_names,
            "class_counts": dict(Counter(labels)),
        },
    }
    return report, le


def run_single_split(
    X: np.ndarray,
    labels: np.ndarray,
    val_fraction: float,
    serve_mult: float,
    non_idle_mult: float,
    idle_min_prob: Optional[float],
    weight_scheme: str = "max_inverse",
    random_state: int = 42,
) -> Tuple[Dict, LabelEncoder]:
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names = le.classes_.tolist()

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=random_state
    )

    model = make_xgb(len(le.classes_), random_state=random_state)
    sw = compute_sample_weights(
        y_tr, le.classes_, serve_mult, non_idle_mult, weight_scheme=weight_scheme
    )
    model.fit(X_tr, y_tr, sample_weight=sw)
    proba_va = model.predict_proba(X_va)

    if idle_min_prob is not None and "idle" in class_names:
        pred_va = apply_idle_min_prob(proba_va, le, idle_min_prob)
    else:
        pred_va = np.argmax(proba_va, axis=1)

    val_metrics = fold_metrics_dict(y_va, pred_va, class_names)
    val_metrics["n_train"] = int(len(y_tr))
    val_metrics["n_val"] = int(len(y_va))
    val_metrics["confusion_matrix"] = confusion_matrix(
        y_va, pred_va, labels=range(len(class_names))
    ).tolist()
    val_metrics["classification_report"] = classification_report(
        y_va, pred_va, labels=range(len(class_names)),
        target_names=class_names, zero_division=0, output_dict=True
    )

    report = {
        "validation": val_metrics,
        "split": {
            "val_fraction": val_fraction,
            "random_state": random_state,
        },
        "config": {
            "serve_weight_mult": serve_mult,
            "non_idle_weight_mult": non_idle_mult,
            "idle_min_prob": idle_min_prob,
            "weight_scheme": weight_scheme,
            "classes": class_names,
        },
    }
    return report, le


def main():
    parser = argparse.ArgumentParser(description="Flat multiclass: train/val or CV + full-data export")
    parser.add_argument("--data-dir", type=str, default="shot_detector/data_csv_only")
    parser.add_argument("--output-dir", type=str, default="shot_detector/retrain_results_wall_flat")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="If >=2, run stratified K-fold CV (mean±std per class); 0 = single train/val split",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Used when --cv-folds is 0: fraction for validation")
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--serve-weight-mult", type=float, default=1.0)
    parser.add_argument("--non-idle-weight-mult", type=float, default=1.0)
    parser.add_argument("--idle-min-prob", type=float, default=None)
    parser.add_argument(
        "--weight-scheme",
        type=str,
        choices=["max_inverse", "sklearn_balanced"],
        default="max_inverse",
    )
    parser.add_argument("--export-counts", action="store_true",
                        help="Write aggregated_class_counts.csv")
    parser.add_argument(
        "--no-save-model",
        action="store_false",
        dest="save_model",
        help="Skip final fit on all data and skip exporting joblib artifacts (default: export)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs, lbls = load_dataset_wall_flat(args.data_dir, args.min_samples)
    X = extract_temporal_features(seqs[:, :, :N_FEATURES_RAW])
    print(f"Temporal features: {X.shape}")

    if args.export_counts:
        counts_path = out_dir / "aggregated_class_counts.csv"
        with open(counts_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["aggregated_class", "count"])
            for cls, cnt in sorted(Counter(lbls).items(), key=lambda x: -x[1]):
                w.writerow([cls, cnt])
        print(f"Wrote {counts_path}")

    json_path: Path
    if args.cv_folds >= 2:
        report, le = run_cv(
            X,
            lbls,
            args.cv_folds,
            serve_mult=args.serve_weight_mult,
            non_idle_mult=args.non_idle_weight_mult,
            idle_min_prob=args.idle_min_prob,
            weight_scheme=args.weight_scheme,
        )
        json_path = out_dir / "flat_wall_shot_cv_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        pfs = report["cv"]["per_fold_mean_std"]
        print(f"\n=== {report['cv']['n_folds']}-fold CV: global mean ± std ===")
        for k in sorted(pfs.keys()):
            if k.endswith("_mean"):
                base = k[:-5]
                print(f"  {base}: {pfs[k]:.4f} ± {pfs.get(base + '_std', 0.0):.4f}")
        oof = report["cv"]["oof_aggregate"]
        print(f"\n=== OOF aggregate (stacked val predictions) ===")
        print(f"  accuracy: {oof['accuracy']:.4f}")
        print(f"  f1_macro: {oof['f1_macro']:.4f}")

        try:
            from shot_detector.plot_shot_metrics import (
                plot_cv_all_classes_prf_mean_std,
                plot_cv_global_metrics_mean_std,
            )

            plot_cv_all_classes_prf_mean_std(
                report, out_dir / "flat_cv_all_classes_prf_mean_std.png"
            )
            plot_cv_global_metrics_mean_std(
                report, out_dir / "flat_cv_global_metrics_mean_std.png"
            )
            print(f"Wrote {out_dir / 'flat_cv_all_classes_prf_mean_std.png'}")
            print(f"Wrote {out_dir / 'flat_cv_global_metrics_mean_std.png'}")
        except Exception as ex:
            print(f"(Plot skipped: {ex})")

        print(f"\nSaved report: {json_path}")
    else:
        report, le = run_single_split(
            X,
            lbls,
            args.val_fraction,
            serve_mult=args.serve_weight_mult,
            non_idle_mult=args.non_idle_weight_mult,
            idle_min_prob=args.idle_min_prob,
            weight_scheme=args.weight_scheme,
        )
        json_path = out_dir / "flat_wall_shot_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        vm = report["validation"]
        sp = report["split"]
        print(
            f"\n=== Validation ({vm['n_train']} train / {vm['n_val']} val, "
            f"val_fraction={sp['val_fraction']}) ==="
        )
        print(f"  accuracy: {vm['accuracy']:.4f}")
        print(f"  balanced_accuracy: {vm['balanced_accuracy']:.4f}")
        print(f"  f1_macro: {vm['f1_macro']:.4f}")
        if "idle" in le.classes_:
            print(f"  idle precision: {vm['classification_report']['idle']['precision']:.4f}")
            print(f"  idle recall: {vm['classification_report']['idle']['recall']:.4f}")
        if "serve" in le.classes_:
            print(f"  serve recall: {vm['classification_report']['serve']['recall']:.4f}")
        if "wall_shot" in le.classes_:
            print(f"  wall_shot recall: {vm['classification_report']['wall_shot']['recall']:.4f}")

        print(f"\nSaved report: {json_path}")

    if args.save_model:
        import joblib

        y = le.transform(lbls)
        sw = compute_sample_weights(
            y,
            le.classes_,
            args.serve_weight_mult,
            args.non_idle_weight_mult,
            weight_scheme=args.weight_scheme,
        )
        model = make_xgb(len(le.classes_))
        model.fit(X, y, sample_weight=sw)
        joblib.dump(model, out_dir / "flat_wall_shot_xgb.joblib")
        joblib.dump(le, out_dir / "flat_wall_shot_label_encoder.joblib")
        meta = {
            "mapping": "DEFAULT_SHOT_MAPPING",
            "classes": le.classes_.tolist(),
            "weight_scheme": args.weight_scheme,
            "n_samples": int(len(lbls)),
            "trained_on": "full_dataset",
            "metrics_report": json_path.name,
        }
        with open(out_dir / "flat_wall_shot_model_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n=== Full-data retrain (export) ===")
        print(f"Saved model: {out_dir / 'flat_wall_shot_xgb.joblib'}")
        print(f"Saved encoder: {out_dir / 'flat_wall_shot_label_encoder.joblib'}")
        print(f"Saved meta:    {out_dir / 'flat_wall_shot_model_meta.json'}")


if __name__ == "__main__":
    main()
