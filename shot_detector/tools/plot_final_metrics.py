"""
Shareable metrics figure for the final shot classifier:
  1) macro-F1 progression across the project (the improvement story)
  2) per-class F1 of the final model
  3) aggregate metrics (final model)

Usage:
    uv run python -m shot_detector.tools.plot_final_metrics \
        --report shot_detector/runs/tcn_final_v2/seq_tcn_cv_report.json \
        --out shot_detector/exports/final_metrics_2026-06-16.jpg
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREEN = "#1b9e77"
HISTORY = [
    ("deployed XGB\n(May, 3 matches)", 0.6127),
    ("TCN reid\n(Jun 12)", 0.6675),
    ("TCN hardneg\n(Jun 13)", 0.6801),
    ("LU fixed\n(Jun 16)", 0.7547),
    ("+idles +court\n(final)", None),  # filled from report
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rep = json.load(open(args.report))
    cv = rep["cv"]
    oof = cv["oof_aggregate"]
    cr = oof["classification_report"]
    f1_final = cv["f1_macro_mean"]
    classes = [c for c in cr if isinstance(cr[c], dict) and "f1-score" in cr[c]
               and c not in ("macro avg", "weighted avg")]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    fig.suptitle(
        "Padel shot classifier — final model (TCN, raw 30x43 pose+ball+court sequence, 5-fold CV)\n"
        f"macro-F1 {f1_final:.3f} ± {cv['f1_macro_std']:.3f}   accuracy {oof['accuracy']:.3f}   "
        f"(9 classes: idle + 8 shot types)", fontsize=13)

    # Panel 1: progression
    labels = [h[0] for h in HISTORY]
    vals = [h[1] if h[1] is not None else f1_final for h in HISTORY]
    ax = axes[0]
    colors = ["#999999"] * (len(vals) - 1) + [GREEN]
    bars = ax.bar(range(len(vals)), vals, color=colors)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=9,
                    fontweight=("bold" if i == len(vals) - 1 else "normal"))
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.55, 0.82)
    ax.set_ylabel("macro-F1 (5-fold CV)")
    ax.set_title("Improvement over the project")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: per-class F1
    order = sorted(classes, key=lambda c: -cr[c]["support"])
    f1s = [cr[c]["f1-score"] for c in order]
    sup = [int(cr[c]["support"]) for c in order]
    ax = axes[1]
    ax.bar(range(len(order)), f1s, color=GREEN)
    for i, v in enumerate(f1s):
        ax.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{c}\n(n={s})" for c, s in zip(order, sup)], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-class F1 (out-of-fold)")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: aggregate + key per-class recall
    ax = axes[2]
    keys = [("accuracy", oof["accuracy"]), ("balanced\naccuracy", oof.get("balanced_accuracy", np.nan)),
            ("f1_macro", oof["f1_macro"]), ("idle\nrecall", cr["idle"]["recall"]),
            ("serve\nrecall", cr["serve"]["recall"]), ("serve\nprecision", cr["serve"]["precision"])]
    ax.bar(range(len(keys)), [k[1] for k in keys], color=GREEN)
    for i, (_, v) in enumerate(keys):
        ax.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k[0] for k in keys], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Aggregate + serve (priority class)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(args.out, dpi=150, format="jpg")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
