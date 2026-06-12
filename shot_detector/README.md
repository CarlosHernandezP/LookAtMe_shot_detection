# Shot detector pipeline

End-to-end flow: **annotation CSVs + ball trajectories → pose CSVs → XGBoost classifier**.

## What to run

| Step | Script | Purpose |
|------|--------|---------|
| 1–3 | `extract_shots.py` | Read shot CSVs, merge ball trajectories, run pose model, write `*_pose.csv` datasets. |
| 4 | `train_shot_model.py` | Build temporal features, CV or train/val metrics, **train on full data**, export `flat_wall_shot_xgb.joblib` + `flat_wall_shot_label_encoder.joblib`. |
| 5 | `plot_shot_metrics.py` | Figures from the JSON reports written by training. |
| — | `plot_track_metrics.py` | After extraction: aggregate `*_track_metrics.json`, CSV summaries, PNG histograms/scatter (see `--data-dir`). |
| — | `filter_clips_by_track_metrics.py` | Writes `include_pose_csvs.txt` / `exclude_pose_csvs.txt` using burst + switch + foot-jump rules (see `--help`). |

Supporting modules (not run directly): `shot_mapper.py` (label mapping), `ball_features.py`, `temporal_features.py`, `pose_io.py` (load pose CSV rows), `utils.py` (CSV / calibration helpers used by extraction).

## Pose backends

Extraction (`extract_shots.py`), single-clip debug (`test_single_clip.py`), and `predict_video.py` share the same pose API. Choose the backend with `--pose-backend`:

| Value | Model | Notes |
|-------|--------|--------|
| `ultralytics` (default) | YOLOv8m-pose | [`ultralytics_pose.py`](ultralytics_pose.py); weights resolved as `model_weights/yolov8m-pose.pt` (created on first run if missing). |
| `mmpose` | RTMO via MMPose | Legacy path; needs `configs/` + `model_weights/rtmo-s_*.pth`. |

Use the same backend for extraction and live inference so pose features match at train vs deploy time.

### Smoke test (~20 shots, MP4 overlays)

After annotation CSVs and videos are on disk (paths in `extract_shots.py`), validate active/idle overlays without colliding with an old RTMO run:

```bash
# From repository root (so package imports resolve):
uv run python -m shot_detector.extract_shots \
  --output-dir shot_detector/validate_yolov8m_pose \
  --resume-state shot_detector/validate_yolov8m_pose/extraction_state.json \
  --max-shots 20
```

Omit `--no-video` so each shot writes an MP4 with active (green) / idle (red) overlays. Use `--max-csv-files K` instead of/in addition to `--max-shots` if you want to cap annotation files rather than shot rows. Use `--random-csv-files N --random-seed S` to process a random subset of annotation CSVs (e.g. 10 videos).

## Training example

```bash
cd /path/to/pose_estimators
uv run python shot_detector/train_shot_model.py \
  --data-dir shot_detector/data_csv_only \
  --output-dir shot_detector/retrain_results_wall_flat \
  --cv-folds 5 --export-counts
```

Inference integration: see `export/INTEGRATION_GUIDE.md` and `export/shot_predictor.py`.

## Legacy

`train_model.py` only re-exports `pose_io` for old imports. Use `train_shot_model.py` for training.

## Frame alignment on VFR videos (fixed 2026-06)

Some rpi recordings are variable-frame-rate (`ffprobe r_frame_rate=250` vs
`avg_frame_rate ~30.005`): `22-11-2025 LU-0002` and both `0529b769` cameras.
On these, `cv2 cap.set(CAP_PROP_POS_FRAMES, N)` seeks by TIME and lands up to
±20 frames away from the Nth sequentially-decoded frame (offset varies along
the video, measured by pixel-exact match against a sequential decode).

All pipeline artifacts (`poses_raw.csv`, `players_reid.csv`,
`ball_trajectories.csv`) index frames by sequential decode order, and
annotations follow the same indexing — so anything that SEEKS is misaligned
with everything that reads SEQUENTIALLY.

**The fix:**

- Training data: use `extract_shots_from_intermediate.py` — poses come from
  `poses_raw.csv` (sequential by construction); no video decoding at all.
- Visualization: `tools/render_intermediate_pose_check.py` decodes each video
  in ONE sequential pass (`cap.grab()` to skip between shot windows,
  `retrieve()` + overlay inside them). Never seek.
- `extract_shots.py` (YOLO path) still seeks — warning at the seek site.
  Datasets built with it (`data_csv_only`, `extract_all_with_clips_v2*`)
  carry pose-vs-ball/annotation misalignment on the VFR videos.
- New video checklist: `ffprobe -show_entries stream=r_frame_rate,avg_frame_rate`;
  if they differ, the video is VFR — sequential reads only.
