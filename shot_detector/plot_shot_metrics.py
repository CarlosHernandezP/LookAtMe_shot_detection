"""
Plot confusion matrices and per-class metrics from ``train_shot_model.py`` JSON reports.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _class_names_from_cr(cr: Dict) -> List[str]:
    skip = {"accuracy", "macro avg", "weighted avg"}
    return [k for k in cr if k not in skip]


def plot_confusion_matrix(
    cm: List[List[float]],
    labels: List[str],
    title: str,
    out_path: Path,
):
    import seaborn as sns

    cm = np.array(cm, dtype=float)
    row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm / row_sum

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(7, n * 0.85), max(6, n * 0.75)))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        square=True,
        linewidths=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Row-normalized"},
    )
    ax.set_title(title)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _class_support_for_ticks(report: dict, class_names: List[str]) -> Dict[str, int]:
    c = (report.get("config") or {}).get("class_counts")
    if isinstance(c, dict) and c:
        return {str(k): int(v) for k, v in c.items()}
    cr = (report.get("cv") or {}).get("oof_aggregate", {}).get("classification_report", {})
    out: Dict[str, int] = {}
    for n in class_names:
        if n in cr and isinstance(cr[n], dict):
            out[n] = int(cr[n].get("support", 0))
    return out


def plot_cv_all_classes_prf_mean_std(report: dict, out_path: Path):
    cv = report.get("cv")
    if not cv or "per_class_mean_std" not in cv:
        raise ValueError("Report must contain cv.per_class_mean_std (from --cv-folds training)")
    pcm = cv["per_class_mean_std"]
    names = sorted(pcm.keys())
    counts = _class_support_for_ticks(report, names)
    w = 0.25
    x = np.arange(len(names))
    pm = [pcm[c]["precision_mean"] for c in names]
    ps = [pcm[c]["precision_std"] for c in names]
    rm = [pcm[c]["recall_mean"] for c in names]
    rs = [pcm[c]["recall_std"] for c in names]
    fm = [pcm[c]["f1_mean"] for c in names]
    fs = [pcm[c]["f1_std"] for c in names]

    fig, ax = plt.subplots(figsize=(max(13, len(names) * 1.1), 6.8))
    ax.bar(x - w, pm, w, yerr=ps, capsize=2.5, label="Precision", color="#4C72B0", ecolor="#333333", alpha=0.92)
    ax.bar(x, rm, w, yerr=rs, capsize=2.5, label="Recall", color="#DD8452", ecolor="#333333", alpha=0.92)
    ax.bar(x + w, fm, w, yerr=fs, capsize=2.5, label="F1", color="#55A868", ecolor="#333333", alpha=0.92)

    tick_labels = [f"{n}\n(n={counts[n]})" if n in counts else n for n in names]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (mean ± std over folds)")
    ax.set_title("Per-class precision, recall, F1 — K-fold CV")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for i, c in enumerate(names):
        if max(pm[i], rm[i], fm[i]) < 1e-6:
            ax.text(
                i,
                0.04,
                "no correct\npreds in val",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#555555",
                linespacing=0.9,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cv_global_metrics_mean_std(report: dict, out_path: Path):
    cv = report.get("cv")
    if not cv or "per_fold_mean_std" not in cv:
        raise ValueError("Report must contain cv.per_fold_mean_std")
    pfs = cv["per_fold_mean_std"]
    keys = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced acc"),
        ("f1_macro", "F1 macro"),
        ("f1_weighted", "F1 weighted"),
    ]
    labels = []
    means = []
    stds = []
    for key, lab in keys:
        km = f"{key}_mean"
        ks = f"{key}_std"
        if km in pfs:
            labels.append(lab)
            means.append(pfs[km])
            stds.append(pfs.get(ks, 0.0))

    if not labels:
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, means, yerr=stds, capsize=4, color="#4C72B0", ecolor="#333333", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (mean ± std)")
    ax.set_title("Global metrics — K-fold CV")
    ax.grid(axis="y", alpha=0.3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, min(m + s + 0.03, 1.0), f"{m:.3f}±{s:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_validation_all_classes_prf(cr: Dict, title: str, out_path: Path):
    names = sorted(_class_names_from_cr(cr))
    prec = [float(cr[c]["precision"]) for c in names]
    rec = [float(cr[c]["recall"]) for c in names]
    f1 = [float(cr[c]["f1-score"]) for c in names]
    sup = [int(cr[c]["support"]) for c in names]

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(12, len(names) * 1.05), 5.8))
    ax.bar(x - w, prec, w, label="Precision", color="#4C72B0")
    ax.bar(x, rec, w, label="Recall", color="#DD8452")
    ax.bar(x + w, f1, w, label="F1", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(n={s})" for n, s in zip(names, sup)], rotation=28, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_flat_report(report_path: Path, out_dir: Path):
    data = load_json(report_path)
    block = data.get("validation") or data.get("oof_aggregate")
    if not block:
        raise KeyError(f"{report_path} has no 'validation' or 'oof_aggregate'")
    classes = data.get("config", {}).get("classes")
    cr = block["classification_report"]
    if not classes:
        classes = _class_names_from_cr(cr)

    plot_confusion_matrix(
        block["confusion_matrix"],
        classes,
        "DEFAULT_SHOT_MAPPING — validation",
        out_dir / "confusion_matrix_validation.png",
    )
    plot_validation_all_classes_prf(
        cr,
        "All classes: precision, recall, F1 (validation)",
        out_dir / "all_classes_precision_recall_f1.png",
    )
    print(f"Wrote {out_dir / 'confusion_matrix_validation.png'}")
    print(f"Wrote {out_dir / 'all_classes_precision_recall_f1.png'}")


def plot_flat_cv_report(report_path: Path, out_dir: Path):
    data = load_json(report_path)
    plot_cv_all_classes_prf_mean_std(data, out_dir / "flat_cv_all_classes_prf_mean_std.png")
    plot_cv_global_metrics_mean_std(data, out_dir / "flat_cv_global_metrics_mean_std.png")
    print(f"Wrote {out_dir / 'flat_cv_all_classes_prf_mean_std.png'}")
    print(f"Wrote {out_dir / 'flat_cv_global_metrics_mean_std.png'}")


def main():
    p = argparse.ArgumentParser(description="Plot metrics from train_shot_model JSON reports")
    p.add_argument(
        "--single-report",
        type=str,
        default="shot_detector/retrain_results_wall_flat/flat_wall_shot_report.json",
        help="Train/val report (when not using CV)",
    )
    p.add_argument(
        "--cv-report",
        type=str,
        default="shot_detector/retrain_results_wall_flat/flat_wall_shot_cv_report.json",
        help="K-fold CV report",
    )
    p.add_argument("--output-dir", type=str, default="shot_detector/retrain_results_wall_flat")
    p.add_argument("--skip-single", action="store_true")
    p.add_argument("--skip-cv", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_single:
        sp = Path(args.single_report)
        if sp.is_file():
            plot_flat_report(sp, out)
        else:
            print(f"Skip single split report (missing): {sp}")

    if not args.skip_cv:
        cp = Path(args.cv_report)
        if cp.is_file():
            plot_flat_cv_report(cp, out)
        else:
            print(f"Skip CV report (missing): {cp}")


if __name__ == "__main__":
    main()
