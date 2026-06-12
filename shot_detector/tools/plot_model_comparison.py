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
# Deployed (old) production model: May 2026, data_csv_only dataset, single
# train/val split. Different (seek-misaligned) dataset -> reference only.
OLD_XGB_REPORT = "model_weights/flat_wall_shot_report.json"

# measured on this host (L4 + 8 vCPU), batch=1 / batch=32, ms per sample
TIMING = {
    "XGB (feat+predict, CPU)": (5.50, 3.89),
    "TCN-big (CPU)": (1.76, 0.26),
    "TCN-big (GPU)": (0.57, 0.023),
}

OLD_COLOR, XGB_COLOR, TCN_COLOR = "#7570b3", "#d95f02", "#1b9e77"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    xgb = json.load(open(XGB_REPORT))["cv"]["oof_aggregate"]
    tcn = json.load(open(TCN_REPORT))["cv"]["oof_aggregate"]
    old = json.load(open(OLD_XGB_REPORT))["validation"]
    xgb_cr = xgb["classification_report"]
    tcn_cr = tcn["classification_report"]
    old_cr = old["classification_report"]
    classes = [c for c in xgb_cr if isinstance(xgb_cr[c], dict) and "f1-score" in xgb_cr[c]
               and c not in ("macro avg", "weighted avg")]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    fig.suptitle(
        "Shot classifier: OLD deployed XGB vs retrained XGB vs TCN-big (raw 30x41 sequence)\n"
        "XGB/TCN: 5-fold CV OOF on hard-negative dataset (n=5689, 5 matches). "
        "OLD XGB: single val split on its own May dataset (3 matches, seek-misaligned) - reference only.",
        fontsize=12,
    )

    # ── Panel 1: aggregate metrics ──
    agg_keys = [("accuracy", "accuracy"), ("balanced_accuracy", "balanced\naccuracy"),
                ("f1_macro", "f1 macro"), ("f1_weighted", "f1 weighted"),
                ("idle_precision", "idle\nprecision"), ("idle_recall", "idle\nrecall")]
    ov = [old.get(k, np.nan) for k, _ in agg_keys]
    xv = [xgb.get(k, np.nan) for k, _ in agg_keys]
    tv = [tcn.get(k, np.nan) for k, _ in agg_keys]
    x = np.arange(len(agg_keys))
    w = 0.27
    ax = axes[0]
    b0 = ax.bar(x - w, ov, w, label="OLD XGB (deployed)", color=OLD_COLOR, hatch="//")
    b1 = ax.bar(x, xv, w, label="XGB retrained", color=XGB_COLOR)
    b2 = ax.bar(x + w, tv, w, label="TCN-big", color=TCN_COLOR)
    for bars in (b0, b1, b2):
        for b in bars:
            if np.isfinite(b.get_height()):
                ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in agg_keys])
    ax.set_ylim(0, 1.08)
    ax.set_title("Aggregate (model level)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 2: per-class F1 ──
    order = sorted(classes, key=lambda c: -xgb_cr[c]["support"])
    of = [old_cr[c]["f1-score"] if c in old_cr else np.nan for c in order]
    xf = [xgb_cr[c]["f1-score"] for c in order]
    tf = [tcn_cr[c]["f1-score"] for c in order]
    sup = [int(xgb_cr[c]["support"]) for c in order]
    x = np.arange(len(order))
    ax = axes[1]
    ax.bar(x - w, of, w, label="OLD XGB (its own data)", color=OLD_COLOR, hatch="//")
    ax.bar(x, xf, w, label="XGB retrained", color=XGB_COLOR)
    ax.bar(x + w, tf, w, label="TCN-big", color=TCN_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={s})" for c, s in zip(order, sup)], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class F1 (hatched = different dataset, indicative only)")
    ax.legend(fontsize=8)
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
