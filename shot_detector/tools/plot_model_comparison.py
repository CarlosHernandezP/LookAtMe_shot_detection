"""
Comparison figure: XGB (K=160 aggregated features) vs TCN-big (raw sequences)
on the hard-negative dataset. Aggregate metrics, per-class F1, inference time.

Usage:
    uv run python -m shot_detector.tools.plot_model_comparison \
        --out shot_detector/exports/model_comparison.jpg
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

XGB_REPORT = "shot_detector/runs/xgb_hardneg/flat_wall_shot_cv_report.json"
TCN_REPORT = "shot_detector/runs/seq_tcn_hardneg_aug_big/seq_tcn_cv_report.json"

# measured on this host (L4 + 8 vCPU), batch=1 / batch=32, ms per sample
TIMING = {
    "XGB (feat+predict, CPU)": (5.50, 3.89),
    "TCN-big (CPU)": (1.76, 0.26),
    "TCN-big (GPU)": (0.57, 0.023),
}

XGB_COLOR, TCN_COLOR = "#d95f02", "#1b9e77"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    xgb = json.load(open(XGB_REPORT))["cv"]["oof_aggregate"]
    tcn = json.load(open(TCN_REPORT))["cv"]["oof_aggregate"]
    xgb_cr = xgb["classification_report"]
    tcn_cr = tcn["classification_report"]
    classes = [c for c in xgb_cr if isinstance(xgb_cr[c], dict) and "f1-score" in xgb_cr[c]
               and c not in ("macro avg", "weighted avg")]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    fig.suptitle(
        "Shot classifier: XGB (533 temporal aggregates, K=160) vs TCN-big (raw 30x41 sequence)\n"
        "5-fold CV out-of-fold predictions, hard-negative dataset (n=5689, 5 matches)",
        fontsize=13,
    )

    # ── Panel 1: aggregate metrics ──
    agg_keys = [("accuracy", "accuracy"), ("balanced_accuracy", "balanced\naccuracy"),
                ("f1_macro", "f1 macro"), ("f1_weighted", "f1 weighted"),
                ("idle_precision", "idle\nprecision"), ("idle_recall", "idle\nrecall")]
    xv = [xgb.get(k, np.nan) for k, _ in agg_keys]
    tv = [tcn.get(k, np.nan) for k, _ in agg_keys]
    x = np.arange(len(agg_keys))
    w = 0.38
    ax = axes[0]
    b1 = ax.bar(x - w / 2, xv, w, label="XGB", color=XGB_COLOR)
    b2 = ax.bar(x + w / 2, tv, w, label="TCN-big", color=TCN_COLOR)
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in agg_keys])
    ax.set_ylim(0, 1.05)
    ax.set_title("Aggregate (model level)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 2: per-class F1 ──
    order = sorted(classes, key=lambda c: -xgb_cr[c]["support"])
    xf = [xgb_cr[c]["f1-score"] for c in order]
    tf = [tcn_cr[c]["f1-score"] for c in order]
    sup = [int(xgb_cr[c]["support"]) for c in order]
    x = np.arange(len(order))
    ax = axes[1]
    ax.bar(x - w / 2, xf, w, label="XGB", color=XGB_COLOR)
    ax.bar(x + w / 2, tf, w, label="TCN-big", color=TCN_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={s})" for c, s in zip(order, sup)], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 3: inference time ──
    ax = axes[2]
    names = list(TIMING)
    bs1 = [TIMING[n][0] for n in names]
    bs32 = [TIMING[n][1] for n in names]
    x = np.arange(len(names))
    colors = [XGB_COLOR, TCN_COLOR, TCN_COLOR]
    ax.bar(x - w / 2, bs1, w, label="batch=1", color=colors, alpha=0.95)
    ax.bar(x + w / 2, bs32, w, label="batch=32", color=colors, alpha=0.55)
    for xi, (v1, v32) in enumerate(zip(bs1, bs32)):
        ax.annotate(f"{v1:g}", (xi - w / 2, v1), ha="center", va="bottom", fontsize=8)
        ax.annotate(f"{v32:g}", (xi + w / 2, v32), ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("ms / sample (log)")
    ax.set_title("Inference time\n(XGB time dominated by temporal-feature step)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(args.out, dpi=150, format="jpg")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
