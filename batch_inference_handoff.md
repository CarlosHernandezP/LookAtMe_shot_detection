# Batch Inference Handoff

Outputs from running `tools/run_batch_inference.py` over `shot_source_videos.txt`
on branch `feat/save-rackets-predictions` with model
`model_weights/object_detection/yolo26m-p2-with-shoes.pt`.

Repo root: `/home/ec2-user/carlos/LookAtMeProtoApp`. All paths below are relative
to that root.

## Matches processed

| match_id (`<match>`) | cameras (`<cam>`) | periods | notes |
|---|---|---|---|
| `0529b769-125d-4a22-bcee-b1707b87447e` | `BO-0001`, `BO-0002` | 3 (tempo-1..3) | two-camera merge fired |
| `15-11-2025-15-57_rpi-BO-0001` | `BO-0001` | 1 (tempo-1) | single camera; merge skipped |
| `18dda9d2-baba-4920-a642-d0a9838d01f3` | `BO-2226`, `BO-2227` | varies | both cams in tree but only `BO-2226` was inferred this round; `BO-2227` pre-existing from prior work |
| `22-11-2025-18-10_rpi-LU-0002` | `LU-0002` | 2 (tempo-1..2) | single camera; merge skipped |

## Output layout

### Per-camera intermediates — `data/intermediate/<match>/<cam>/`
- `detections.csv` — `frame_num,object_type,object_id,bbox,court_x,court_y,confidence`. `object_type ∈ {player, ball}`. `object_id` = track id for player, -1 for ball.
- `rackets_prediction.csv` — `frame_num,bbox,court_x,court_y,confidence`. **New sidecar from this branch**; raw racket detections, untracked.
- `poses_raw.csv` — `frame_num,track_id,keypoints`. YOLO-pose keypoints per player track.
- `ball_trajectories.csv` — `frame_number,trajectory_id,position_x,position_y,confidence,is_interpolated[,court_x,court_y]`.
- `players_reid.csv` — concatenated re-identified player positions across all periods.
- `video_metadata.json` — fps, dimensions, frame count.

### Per-period intermediates — `data/intermediate/<match>/<cam>/period_<P>/`
`<P>` is 0-indexed. **Maps to `tempo-<P+1>` in processed outputs.**
- `players_reid.csv` — frame range `[frame_number.min, frame_number.max]` is the **absolute video frame offset for that period**. Predictions in `_shots.csv` use frame numbers relative to that offset; downstream comparators must add it.
- `shot_detection/player_<id>_poses.csv` — pose features for the **2 close-to-camera players** only. Exists only for those local players; use their IDs as the local-player set for that period.

### Per-camera processed — `data/processed/<match>/<cam>/`
- `<match>_tempo-<N>_team.json` — team composition per period.
- `<match>_tempo-<N>_identification_frame.jpg` — frame used for team ID overlay.
- `<match>_tempo-<N>_player-<id>_statistics.json` — aggregated player stats.
- `<match>_tempo-<N>_player-<id>_heatmap.json` — court heatmap.
- `unused/<match>_tempo-<N>_player-<id>_shots.csv` — **per-frame** shot predictions, schema below.
- `unused/<match>_tempo-<N>_player-<id>_positions.csv` — per-frame positions.

#### `_shots.csv` schema
```
frame_num, shot_type, confidence,
prob_backhand, prob_bandeja, prob_forehand, prob_idle, prob_serve,
prob_smash, prob_vibora, prob_volley, prob_wall_shot
```
- `frame_num` is **relative to the period start** (offset = first frame of `period_<N-1>/players_reid.csv`).
- For shot-event comparison, smooth + segment via `src.utils.shot_segment_counter`:
  ```python
  from src.utils.shot_segment_counter import smooth_predictions, find_shot_segments
  smoothed = smooth_predictions(df, window_frames=30)
  segments = find_shot_segments(smoothed)  # {shot_type: [(start_frame, end_frame), ...]}
  ```
- After segmenting, add the period offset to get absolute video frame numbers.

### Match-level processed (only `0529b769-...`) — `data/processed/<match>/`
- Two-camera merge outputs. Format unchanged from prior pipeline runs.

## Annotation files

Source: `/home/ec2-user/data/shot_annotations/annotation_*.csv`. Schema:
`Shot,FrameId,Player` where `Player ∈ {right, left}`. Some files have repeated
headers mid-file — filter rows where `Shot == "Shot"` before parsing.

Filename conventions:
- `annotation_<match>_period<N>.csv` or `annotation_<match>_<N>.csv` — single period
- `annotation_<match>.csv` — entire video (no period split)

Period in annotation filename maps to pipeline `tempo-<N>` and to
`period_<N-1>` under `data/intermediate/`.

## Existing comparator

`tools/compare_shots_to_annotations.py` (in this repo). Already handles:
- Period frame offset (intermediate `period_<P>/players_reid.csv` min frame).
- Filtering to local players via `period_<P>/shot_detection/player_*_poses.csv`.
- Repeated annotation headers.
- Annotation label aliases (limited — extend if needed).

Run: `uv run python tools/compare_shots_to_annotations.py`

## Label mismatch (known caveat)

Annotation vocabulary (e.g. `lob`, `wall_lob`, `forehand_contrapared`,
`forehand_wall_exit`, `backhand_volley`, `bajada`, `flat_smash`,
`topspin_smash`, `drop_shot`) is finer than the shot model's classes
(`backhand, bandeja, forehand, idle, serve, smash, vibora, volley, wall_shot`).
A mapping table is needed for fair type-accuracy comparison. Suggested mappings:
- `flat_smash`, `topspin_smash` → `smash`
- `backhand_volley`, `forehand_volley` → `volley`
- `backhand_wall_exit`, `forehand_wall_exit`, `wall_lob` → `wall_shot`
- `lob`, `drop_shot`, `bajada`, `forehand_contrapared` → no direct match
  (treat as untyped match if frame matches)

## Batch run logs

`logs/batch_inference_20260519-092537/` contains per-video master logs (size
in MB) + per-pipeline sub-logs under `<video>_pipeline_logs/`. Each video
took ~140–145 min; total batch wall time ~12 hr.
