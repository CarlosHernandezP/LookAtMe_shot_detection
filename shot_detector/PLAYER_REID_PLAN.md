# Player Re-ID for Active/Idle Player Assignment

## Motivation

`extract_shots.py` currently picks the active and idle player per shot via
`identify_player()` in `shot_detector/utils.py`. It maps an annotation label
(`top`, `bottom`, `left`, `right`, `top_left`, ...) to a pose detection using
pure image-space heuristics on the current frame's pose centroids.

Empirically this misfires in two ways:

1. **Wrong player picked** when two players are close together, overlap, or
   when a pose detection drops out for a frame and a different player is
   the closest survivor.
2. **Identity swaps mid-clip** because we re-decide per frame instead of
   tracking a single identity through the clip.

We now have a much stronger upstream signal: `players_reid.csv` from the
new pipeline (`LookAtMeProtoApp`). It assigns a stable `player_id` to every
detected player across the entire video, with bbox and court coordinates.

## Data availability (5 source videos, 6 cams)

All processed matches have full per-camera + per-period `players_reid.csv`:

| match | cam | rows | frames covered | periods | reid per period |
|---|---|---|---|---|---|
| `0529b769-125d-4a22-bcee-b1707b87447e` | `BO-0001` | 542,722 | 0..148,692 | 3 | 3/3 |
| `0529b769-125d-4a22-bcee-b1707b87447e` | `BO-0002` | 546,028 | 0..148,716 | 3 | 3/3 |
| `15-11-2025-15-57_rpi-BO-0001` | `BO-0001` | 452,401 | 0..138,478 | 3 | 3/3 |
| `18dda9d2-baba-4920-a642-d0a9838d01f3` | `BO-2226` | 609,589 | 0..161,993 | 3 | 3/3 |
| `18dda9d2-baba-4920-a642-d0a9838d01f3` | `BO-2227` | 607,048 | 0..161,993 | 3 | 3/3 |
| `22-11-2025-18-10_rpi-LU-0002` | `LU-0002` | 453,374 | 0..119,731 | 3 | 3/3 |

Path: `LookAtMeProtoApp/data/intermediate/<match>/<cam>/players_reid.csv`
(camera-level) and `.../period_<P>/players_reid.csv` (per-period).

Schema:
```
frame_number, object_type, object_id, bbox, court_x, court_y, confidence, player_id
```

`bbox` is `[x1, y1, x2, y2]` (verify on first row). `player_id` is stable
across the period; `object_id` is the raw track id from the upstream tracker.

## Proposed approach (NOT implemented yet)

1. **Load `players_reid.csv`** for the (match, cam) of the shot's annotation
   file. Filter to the shot's frame window `[center-15, center+14]`.
2. **For each frame in the window**, run YOLO-pose as today and obtain N pose
   bboxes (from keypoint extents or the detector's bbox).
3. **Match each pose to a `player_id`** by IoU against the reid bboxes in
   that frame. Greedy max-IoU assignment; drop pose-reid pairs below an IoU
   threshold (start ~0.3).
4. **Pick the active player** for the shot by court position rule on the
   `player_id`'s court_x/court_y, not on pose centroids. Annotation label
   (`top` / `bottom` / `left` / `right`) maps to court-quadrant rule. Idle
   player is the other side of the same team / other-team mate, per the
   same court rule.
5. **Track-through-clip**: once `active_player_id` is chosen at the center
   frame, force every other frame's active pose to be the one matched to
   that `player_id`. This eliminates mid-clip identity swaps.
6. **Quality gates** (some already exist in `extract_shots.py`):
   - drop poses with IoU < threshold against any reid bbox (likely
     detector hallucination)
   - keep existing jitter / stationary / flickering / corner filters
   - re-evaluate `MAX_FORWARD_FILL` once reid is the source of truth — it
     can probably be lowered.

## Files involved

- `shot_detector/extract_shots.py` — `identify_player()` call sites,
  per-frame active/idle index resolution, the pose CSV writer.
- `shot_detector/utils.py` — `identify_player()` lives here, will gain a
  reid-aware variant.
- New: `shot_detector/players_reid_io.py` — loader + per-frame index by
  `(match, cam, frame)`.
- New: `shot_detector/pose_reid_match.py` — IoU-based pose↔reid matcher.

## Open questions

- IoU threshold (start 0.3, calibrate on a labeled subset).
- Fallback when no reid bbox in a frame: keep old heuristic, or drop the
  frame? Probably keep heuristic + flag in `track_metrics.json`.
- `match` key in `extract_shots.py` is video-name-derived; reid is
  ProtoApp-match-keyed. Need a map (parallel to `BALL_TRAJECTORY_MAP`).

## Where to start (next branch)

1. Stand up `players_reid_io.py` + a tiny CLI that prints reid bboxes for
   a given video/frame.
2. Compute IoU distribution between current `identify_player()` picks and
   reid bboxes across a sample of shots — to ground the threshold + show
   how often today's heuristic disagrees with reid.
3. Wire reid into `extract_shots.py` behind a `--use-reid` flag, keep the
   old path as fallback so we can A/B on track_metrics.
