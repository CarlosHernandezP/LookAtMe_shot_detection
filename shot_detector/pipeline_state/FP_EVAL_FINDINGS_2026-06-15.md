# Production False-Positive Eval — Findings (2026-06-15)

Production-realistic, full-match eval of the shot classifier. Harness:
`stats-poc/tools/eval_shot_fp_fullmatch.py` (reuses production `predict_shots`
→ `smooth_predictions` → `find_shot_segments`), per-shot-type metrics,
leave-one-match-out (LOO) so each held-out match is leakage-free.

## TL;DR

1. The over-counting is **real and reproduces across all 5 matches** (2–4×
   inflation at the current segmenter, leakage-free).
2. The segmenter had a genuine bug (no duration/confidence floor) — **fixed**
   (`find_shot_segments_hardened`), model-agnostic, shipped to stats-poc.
3. Reverting `non_idle_weight_mult` (1.918 → 1.0) helps — confirmed directionally.
4. **BUT we cannot trust the precision / false-positive numbers**: the
   annotations are incomplete (~6–10 shots/min vs ~15–40/min for real active
   padel — only ~25–50% of shots labelled). Most "false positives" are real,
   unannotated shots. **The measurement instrument is inadequate for the FP
   question.** Threshold tuning for precision is therefore NOT reliable yet.

## Leakage matters

Earlier in-training eval (a145dd19 in the training set) suggested precision
~0.63. Leakage-free LOO across 5 matches: overall detection precision tops out
~0.44–0.52 — much worse. The model overfits to seen courts/players; each new
venue degrades it. **LOO is the production-relevant number.**

## Pooled LOO sweep (5 matches, 2246 annotated shots, 353 serves)

| pick | mw | cf | overall P | overall R | serve P | serve R |
|---|---|---|---|---|---|---|
| best overall F1 | 3 | 0.0 | 0.44 | 0.60 | 0.16 | 0.34 |
| best overall precision | 3 | 0.8 | 0.52 | 0.18 | 0.15 | 0.07 |
| best serve F1 | 2 | 0.0 | 0.36 | 0.74 | 0.15 | 0.46 |

Recall is the trustworthy axis (of annotated shots, how many found):
**overall 60–81%, serve 34–53%.** Serve recall is independently weak on
unseen matches — that part is not an annotation artifact.

## Serve false-positives are spurious, not confusion (15-11 fold)

Of 82 detected serves: 12 near a real serve, **0 confused with another shot
type, 70 in dead time**. Even at ±10 s tolerance, 51/82 are >10 s from ANY
annotation. So serve FPs are the model calling "serve" during
between-point / ready / walking moments — OR real serves the annotation
missed. With incomplete annotations we cannot separate these two.

## Why the annotations can't measure FP

Shots/min per annotation file: 5.7, 9.4, 6.5, 6.6, 9.9, 7.9, 9.6, 7.5.
Real active padel ≈ 15–40 shots/min. The labels capture a subset, so any
detection that doesn't hit a label is ambiguous: false positive OR real
unannotated shot. Precision is a lower bound, badly contaminated.

## What stands (independent of the annotation issue)

- **Segmenter fix** (`find_shot_segments_hardened`): a flicker is one window,
  a real swing sustains — requiring ≥`min_windows` and a confidence floor is
  correct regardless of how precision is measured. Ship it.
- **Revert `non_idle_weight_mult`**: training up-weighted shots, biasing toward
  firing — wrong for "don't over-count". Reverted in all current models.
- **Serve recall is genuinely low** on unseen courts → needs data/features.

## Real next steps (not threshold tuning)

1. **Get a properly measurable test set**: one match with EVERY shot annotated,
   OR human-review a random sample of the model's detections to estimate true
   precision. Until then FP cannot be quantified.
2. **Serve**: more serve training data + a court-position / serve-zone feature
   (serves start from behind the baseline; pose-only windows lack this).
   Likely needs rally/point structure, not just per-window pose.
3. **More venue diversity** in training (5 matches; LOO shows each new court
   breaks generalization).

## Artifacts

- Harness + per-class eval: `stats-poc/tools/eval_shot_fp_fullmatch.py`
- LOO drivers: `LookAtMe shot_detector/tools/run_loo_tcn.sh`,
  `stats-poc/tools/run_loo_eval.sh`, `tools/aggregate_loo_sweep.py`
- Hardened segmenter: `stats-poc/src/utils/shot_segment_counter.py`
  (branch `fix/shot-segment-fp-floor`)
- LOO + full-data no-weight TCN bundles: `shot_detector/runs/tcn_loo_*`,
  `shot_detector/runs/tcn_final_noweight`

## Court-position features for serves (2026-06-15, update)

Serves happen behind a baseline. Added court position from `players_reid.csv`
as TCN channels and ablated by held-out match (train on 4, test per-class
recall on the 5th's annotated shot windows — clean: classifying labelled
windows, not the contaminated sliding-window FP).

**Key geometric finding (Carlos's catch):** the LU-0002 camera watches the
OPPOSITE baseline from the BO cameras. Close players (and servers) sit at
`court_y ≈ 18` on BO cameras but `court_y ≈ 2` on LU. The calibration is
CORRECT — `court_y` is just camera-mounting-dependent, so raw `court_y` does
NOT transfer across mountings. `dist_to_near_baseline = min(court_y, 20-court_y)`
≈ 2.0 for serves on EVERY court → that is the camera-invariant signal.

Serve recall / precision by held-out match (TCN, 128ch):

| feature set | 22-11 (LU) | 18dda9d2 (BO) | 15-11 (BO) |
|---|---|---|---|
| base (41) | 0.085 / 0.114 | 0.68-0.73 / 0.49-0.53 | 0.957 / 0.721 |
| +court_y,court_x | 0.021 (hurt) | 0.798 / 0.635 | 0.870 (hurt recall) |
| +dist_to_baseline,court_x | **0.149 / 0.175** | 0.734 / 0.537 | (not run) |

- Raw `court_y` is mounting-dependent → inconsistent (helped 18dda9d2, hurt LU
  and 15-11). DO NOT use it.
- `dist_to_near_baseline + court_x` is camera-invariant → helps LU (0.085→0.149),
  helps 18dda9d2 precision, never hurts. **This is the feature set to adopt.**

**Caveats:** (1) absolute LU serve recall stays poor (0.149) — LU is the only
LU venue; features can't substitute for venue diversity. (2) run-to-run noise
~5 pts (stochastic training); only the LU delta and the geometric argument are
reliable. The dominant lever remains more venue diversity in training.

**Recommendation:** use `dist_to_near_baseline` + `court_x` (NOT court_y) as
two extra TCN channels in the next training. At inference, append them per
frame from `players_reid.csv` (court_x, and min(court_y, 20-court_y)). Treat
as a modest, robust improvement — not a generalization fix.
