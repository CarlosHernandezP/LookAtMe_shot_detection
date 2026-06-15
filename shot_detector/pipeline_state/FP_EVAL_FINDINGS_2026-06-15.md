# Production False-Positive Eval — Findings (2026-06-15)

Full-match, production-stride eval of the shot classifier on held-out match
`a145dd19` (BO-2222, 54000 frames, 146 annotated shots, both close players).
Harness: `stats-poc/tools/eval_shot_fp_fullmatch.py` (reuses production
`predict_shots` → `smooth_predictions` → `find_shot_segments`).

## The problem is real and reproduced

Running the **deployed May XGB** (leakage-free — a145dd19 not in its training)
over the whole match with the CURRENT segmenter:

- **276 shots emitted vs 146 annotated = 1.89x inflation**
- precision **0.49**, recall 0.93

Half of all reported shots are false. Matches the production complaint exactly.

## Root cause (confirmed in code)

`find_shot_segments` counts a shot for **any** smoothed window whose argmax is
non-idle — **no minimum duration, no confidence floor**. A real swing sustains
across several windows; a walking hand-wave is one window. Both count as +1.
Every brief misclassification becomes a counted shot.

## Fix shipped: hardened segmenter

`find_shot_segments_hardened` (in `stats-poc/src/utils/shot_segment_counter.py`)
keeps a segment only if it spans `>= min_windows` windows AND reaches
`min_peak_confidence`. Sweep on deployed XGB:

| config | emitted | inflation | precision | recall |
|---|---|---|---|---|
| current (mw=1, cf=0) | 276 | 1.89x | 0.49 | 0.93 |
| mw=2, cf=0 | 213 | 1.46x | 0.60 | 0.88 |
| mw=3, cf=0 | 124 | 0.85x | 0.73 | 0.62 |
| mw=1, cf=0.7 | 78 | 0.53x | 0.78 | 0.42 |

Hardening roughly halves false positives. But there's a **precision/recall
wall**: pushing precision past ~0.73 collapses recall, because the per-window
prediction stream itself is noisy.

## The bigger finding: training objective is misaligned with production

The hard-negative XGB (`xgb_hardneg`, idle-enriched) is actually **worse** on
raw count here (337 emitted, 2.31x, precision 0.35) than the old model. Cause:
we trained every model with `non_idle_weight_mult=1.918` (up-weights shots to
balance macro-F1). That bias is exactly wrong for "do not over-count shots" —
it makes the model eager to fire. The whole session optimized **macro-F1 on a
balanced split**, which is not the production objective.

Two structural fixes beyond segmentation (deferred earlier, now clearly needed):

1. **Retrain favoring idle**, not against it: drop `non_idle_weight_mult`,
   consider up-weighting idle, OR apply prior correction at inference
   (× production/train idle prior) — the base-rate fix we listed but skipped.
2. **Confidence floor is mandatory** at deploy: the hardneg model only behaves
   with `min_peak_confidence ~0.7` (precision 0.71, emitted 121 vs 146).

## Recommended immediate deploy (model-agnostic)

`find_shot_segments_hardened(min_windows=2, min_peak_confidence=0.6)` — cuts
inflation toward 1.0x without destroying recall. **Tune on YOUR deployed model**
with the harness; optimal floor is model-specific (old XGB likes mw=3/cf=0,
hardneg likes cf=0.7).

## Honest caveat

This proves the segmenter was a major bug and the hardening helps a lot, but it
does NOT fully solve over-counting on its own — the per-window model needs the
prior-correction / idle-favoring retrain to break the precision/recall wall.
Recall numbers also depend on event-match tolerance (±30 frames here).
