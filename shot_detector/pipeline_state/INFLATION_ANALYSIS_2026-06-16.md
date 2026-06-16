# Prediction Inflation Analysis (2026-06-16)

Goal: reduce production over-counting (down-time motions — high-fives, ball
pick-ups, ready stances — counted as shots). Final model: `tcn_final_v2`
(TCN, pose+ball+court, f1_macro 0.773).

## Per-window false-shot rate

"False-shot" = a confirmed down-time (idle) window predicted as a shot.

- **In-distribution** (matches the model trained on): ~1.9% at argmax. Low —
  the model recognizes familiar courts' down-time well.
- **Out-of-distribution** (held-out match 18dda9d2, the production-relevant
  case): **9.9% at argmax**. This is the honest number for a NEW match/court.

## Threshold lever (held-out 18dda9d2, unseen)

| confidence threshold | false-shot rate (down-time) | shot recall |
|---|---|---|
| argmax | 0.099 | 0.790 |
| 0.50 | 0.072 | 0.647 |
| 0.60 | 0.054 | 0.563 |
| 0.70 | 0.033 | 0.470 |

On an unseen match the threshold is NOT free — every cut in false-shots costs
shot recall (10%→7% costs 79%→65% recall). In-distribution it was nearly free;
out-of-distribution it isn't.

## Methods tried

| method | effect on unseen-match inflation | verdict |
|---|---|---|
| +300 random mined idles | neutral (helped in-dist f1 +1.6, not OOD inflation) | keep (f1), not an inflation fix |
| self-training FP hard-negatives (mine model's confident FPs → idle) | WORSE (9.9%→13.6%, likely noise; 52 negs too few) | does NOT generalize |
| confidence threshold | trades false-shots for recall | useful knob, recall cost |
| hardened segmenter (min_windows + conf floor) | filters isolated FPs | **the real lever** (below) |

## The real lever: the segmenter, not per-window tricks

The per-window 9.9% OVERSTATES production counted-shot inflation. Production
counts shots via `find_shot_segments` over a continuous prediction stream. A
real shot spans several consecutive windows (survives `min_windows>=2`); a
down-time false-shot is isolated and low-confidence (most FPs are <0.5 conf),
so it gets filtered. `find_shot_segments_hardened(min_windows=2,
min_peak_confidence~0.5)` cuts COUNTED inflation far more than per-window
thresholding, at much lower recall cost. (Built earlier, on stats-poc branch
`fix/shot-segment-fp-floor`.)

## The honest bottleneck

Out-of-distribution down-time inflation (~10% per window) resists training
tricks (random idles, FP-mining don't generalize from 5 matches/4 venues).
The model overfits to seen courts' down-time. **The #1 fix is more venue
diversity in training**, not more negatives on the same courts — consistent
with the earlier LOO generalization finding.

## Recommendations (ranked)

1. **Deploy `tcn_final_v2` + hardened segmenter** (min_windows=2,
   min_peak_confidence~0.5). Tune the floor per deployment with the harness.
2. **Add venue diversity** (more courts/matches) — the only thing that lowers
   the per-window OOD inflation floor.
3. Untried but promising (next): contact-frame-centered windows (tighter shot
   windows), a two-stage binary shot-detector + type-classifier (lets you tune
   inflation independently of type accuracy), 5-fold ensemble at inference.
