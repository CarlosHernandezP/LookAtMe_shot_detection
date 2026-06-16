"""
Per-class precision / recall / F1 comparison: the pre-TCN deployed XGBoost vs
the current final TCN. 3 panels (P, R, F1), grouped bars per class.

NOTE: the old XGB was a single train/val split on the May `data_csv_only`
dataset (3 matches, pre-VFR-fix, n=2219); the TCN is 5-fold OOF on the current
clean 5-match dataset (n=6055). So this is an indicative model-vs-model
comparison across eras, not a controlled same-data benchmark.

Usage:
    uv run python -m shot_detector.tools.plot_xgb_vs_tcn \
        --xgb model_weights/flat_wall_shot_report.json \
        --tcn shot_detector/runs/tcn_final_v2/seq_tcn_cv_report.json \
        --out shot_detector/exports/xgb_vs_tcn_2026-06-16.jpg
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

XGB_C, TCN_C = "#7570b3", "#1b9e77"
CLASSES = ["idle", "forehand", "serve", "volley", "backhand", "smash", "bandeja", "vibora", "wall_shot"]


def cr_of(path, key):
    r = json.load(open(path))
    if "validation" in r:               # old XGB single-split report
        return r["validation"]["classification_report"], r["validation"]
    cv = r["cv"]["oof_aggregate"]       # TCN OOF
    return cv["classification_report"], cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xgb", required=True)
    ap.add_argument("--tcn", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    xgb, xagg = cr_of(args.xgb, "xgb")
    tcn, tagg = cr_of(args.tcn, "tcn")
    classes = [c for c in CLASSES if c in xgb and c in tcn]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    fig.suptitle(
        "Pre-TCN deployed XGBoost  vs  current final TCN — per-class\n"
        f"XGB f1_macro {xagg.get('f1_macro', float('nan')):.3f} (single split, May 3-match data)   |   "
        f"TCN f1_macro {tagg['f1_macro']:.3f} (5-fold OOF, current 5-match data)   "
        "— indicative across-era comparison", fontsize=12)

    for ax, metric, title in zip(axes, ["precision", "recall", "f1-score"],
                                 ["Precision", "Recall", "F1"]):
        xv = [xgb[c][metric] for c in classes]
        tv = [tcn[c][metric] for c in classes]
        x = np.arange(len(classes)); w = 0.4
        ax.bar(x - w / 2, xv, w, label="XGB (pre-TCN)", color=XGB_C, hatch="//")
        ax.bar(x + w / 2, tv, w, label="TCN (current)", color=TCN_C)
        for i, (a, b) in enumerate(zip(xv, tv)):
            ax.annotate(f"{a:.2f}", (i - w / 2, a), ha="center", va="bottom", fontsize=7)
            ax.annotate(f"{b:.2f}", (i + w / 2, b), ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_title(title); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(args.out, dpi=150, format="jpg")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
