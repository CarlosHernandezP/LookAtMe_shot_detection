# Handoff — re-run pipeline on CFR videos (2026-06-15)

## Why

LU-0002 (and the 0529 cameras) are VFR videos. Their pipeline `poses_raw`
decoded out of sync with the annotation timeline → shots extracted at the
wrong frames → garbage LU training data (serve recall 0.085). Root cause +
proof: `FP_EVAL_FINDINGS_2026-06-15.md`. NOT a court-position problem.

Fixed by converting the 3 VFR videos to true CFR 30. Alignment confirmed
visually (`tools/verify_cfr_alignment.py` montages: active player mid-swing at
the annotation frame, early/mid/late, all 3 videos).

## Re-run the MAIN pipeline on ONLY these 3 (CFR-corrected)

```
/home/ec2-user/data/matches/cfr/22-11-2025-18-10_rpi-LU-0002.mp4                              (match 22-11-2025-18-10_rpi-LU-0002, cam LU-0002)
/home/ec2-user/data/matches/cfr/0529b769-125d-4a22-bcee-b1707b87447e_1767949074894_rpi-BO-0001.mp4  (match 0529b769-..., cam BO-0001)
/home/ec2-user/data/matches/cfr/0529b769-125d-4a22-bcee-b1707b87447e_1767949074880_rpi-BO-0002.mp4  (match 0529b769-..., cam BO-0002)
```

All true CFR (r=30/1, ~161,997 frames). BO-0001 + BO-0002 are the SAME match.

DO NOT re-run: `15-11-2025-15-57_rpi-BO-0001`, `18dda9d2-...-BO-2226`,
`a145dd19-...-BO-2222` — already CFR-native and correct.

Pipeline run is per-camera (~140 min each). No code change — point it at the
CFR paths. The new intermediates (poses_raw/players_reid/ball_trajectories)
will land in the usual tree; if they overwrite the old VFR ones, that's the
intent (old ones are misaligned). Consider writing to a new match-id suffix or
backing up the old intermediates if you want to keep both.

## After the pipeline finishes (tomorrow-me)

1. Re-extract shot windows from the new intermediates:
   `uv run python -m shot_detector.extract_shots_from_intermediate --output-dir shot_detector/data/extract_intermediate_poses_court` (court columns included), then re-mine idles (`tools/mine_idle_negatives.py`), rebuild the merged dataset.
2. Retrain TCN (no non-idle weight) — expect a big LU jump, modest 0529 gain.
3. RE-EVALUATE the earlier conclusions on the now-clean data (they were
   confounded by LU corruption):
   - court features: `dist_to_near_baseline + court_x` (NOT raw court_y —
     mounting-dependent). Re-run the held-out ablation.
   - segmenter thresholds (`find_shot_segments_hardened`) — re-sweep.
   - Note: annotations are still INCOMPLETE (~6-10 shots/min vs ~15-40 real),
     so precision/FP still can't be measured against them; recall is the
     trustworthy axis. (FP_EVAL_FINDINGS.)
4. The labeling/clip datasets built from old VFR frames are misaligned —
   regenerate from CFR.

## Tools (this session, on branch feat/player-reid-pose-matching)

- `tools/vfr_to_cfr.sh` — the VFR→CFR conversion (ffmpeg -fps_mode cfr -r 30).
- `tools/verify_cfr_alignment.py` — montage: YOLO-pose on CFR frames, active
  player red, CONTACT frame marked. Use to confirm any new CFR video.
- `extract_shots_from_intermediate.py` (--court-features), `train_sequence_model.py`
  (--court-features, --heldout-token), `tools/mine_idle_negatives.py`.
