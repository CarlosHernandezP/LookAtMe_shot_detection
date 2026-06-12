# Feature Research — Shot Classifier

Survey of additional features unlocked by the new pipeline outputs
(`players_reid.csv`, `rackets_prediction.csv`, updated `ball_trajectories.csv`).
**Research only — no implementation yet.** Pick what passes a smell test
before wiring into `extract_shots.py` and the training set.

Current feature set (see `pose_io.py` / `temporal_features.py`):
- per-keypoint body-relative coordinates
- absolute feet/shoulder positions
- elbow / shoulder joint angles
- ball position + velocity + acceleration in body-relative frame and absolute frame

## Candidate features

### A. Contact-frame estimation from ball trajectory

The contact frame is the most informative moment of a shot. Today we use
`center_frame` from the annotation, which is hand-marked and noisy by
several frames. Ball trajectory has the answer.

- **Ball-y inflection (`d²y/dt²` sign change)** during the shot window.
  Padel ball bounces and contacts both produce inflections; filter by
  proximity to the active player's wrist.
- **Ball-to-wrist minimum distance frame**: argmin over the window of
  `‖ball_xy − active_wrist_xy‖`. Strong proxy for contact.
- **Trajectory ID change** in `ball_trajectories.csv`. A new
  `trajectory_id` near the active player almost always means a contact
  event.

Use the estimated contact frame as a feature on its own (offset from
annotation frame) AND re-center the temporal window around it.

### B. Racket–wrist coupling

`rackets_prediction.csv` gives racket bboxes per frame. Combined with
pose wrists:

- **Racket center–wrist distance** (pixel + court-normalized).
  Distinguishes serve / smash (racket up + away from wrist on backswing)
  from volley (racket stays close).
- **Racket bbox area / aspect** as a swing-phase proxy. Larger blurred
  racket = mid-swing.
- **Racket vertical position vs head**: serve / lob have racket above
  head pre-contact; ground strokes don't.
- **Racket-velocity** (frame-to-frame center displacement). Smash >
  forehand > volley > idle. Strong signal.

### C. Court-relative motion (from reid)

reid provides `court_x`, `court_y` per player_id per frame. New features:

- **Active player court speed** = `‖Δcourt_xy‖ / Δt` over the window.
- **Court-x trajectory direction** (signed). Backhand vs forehand often
  correlates with lateral approach direction.
- **Distance to closer side wall** at contact. Wall shots imply the
  player is close to a wall; idle/volley typically not.
- **Distance to net** at contact. Volley / bandeja / vibora cluster near
  the net; serve / lob / wall shot far.
- **Court quadrant occupancy** as a one-hot (top-left/top-right/
  bottom-left/bottom-right).

### D. Opponent and partner geometry (also from reid)

- **Distance to nearest opponent** at contact, and its rate of change.
- **Partner position relative to active player** (vector). Common
  serve / volley formations have the partner near the net.
- **Average court-y of the 4 players** (team formation tilt).

### E. Pose temporal dynamics

The current set already has joint angles. Adding:

- **Angular velocity** of elbow and shoulder. Smash / serve peak much
  higher than forehand.
- **Trunk rotation** (shoulder-line angle) at contact and its rate of
  change pre-contact (windup speed).
- **Hip rotation** relative to trunk. Differentiates open/closed stance.
- **Knee flexion** at contact. Bandeja / vibora have distinctive bent
  knees pre-contact.
- **Pose stability variance** in the 5 frames before contact (how much
  the player moves vs sets).

### F. Handedness-aware features

We already mirror left-handed cases. With reid we can detect handedness
robustly:

- **Dominant wrist** = wrist closer to ball at contact across the
  player's history. Eliminates the need for the manual
  `HANDEDNESS_MIRROR_RULES` map.

### G. Shot direction (post-contact ball trajectory)

After the estimated contact frame, the ball trajectory direction is
a strong predictor of certain shot types (wall shots travel toward
walls, drops travel down).

- **Ball-y change in first 0.3s post-contact** — drops vs lobs.
- **Ball-x change in first 0.3s post-contact** — direction of placement.
- **Ball court-y at next bounce** — depth of placement.

## Quick wins to try first

Ranked by expected gain vs implementation cost:

1. **B (racket-wrist distance + racket velocity)** — directly available,
   tight code change in `extract_shots.py` / `temporal_features.py`.
2. **A (contact-frame from ball-to-wrist argmin)** — small change to
   the windowing logic; benefits every other feature.
3. **C (court speed, distance to net, distance to wall)** — needs reid
   loading already added by the player-reid branch; small additions.
4. **G (post-contact ball direction)** — easy with the existing ball
   trajectories.

## What NOT to add right now

- **Visual features from the clip** (e.g. raw image patches into a CNN
  head). Out of scope until we have labels for success/fail to justify
  end-to-end training.
- **Multi-camera fusion**. We have BO-2226 + BO-2227 for one match but
  haven't aligned timestamps cleanly — defer until the merge pipeline
  in ProtoApp is the source of truth.

## Validation plan

For each candidate feature:

1. Compute it on the existing labeled set (re-extracted with
   `--use-reid`).
2. Univariate metric: mutual information with shot class. Drop anything
   below baseline of a known-good feature (e.g. existing
   `right_elbow_angle_rad`).
3. Add the survivors to the XGB training run on top of the
   autoresearch K=160 subset (see `model_weights/flat_wall_shot_*`)
   and measure delta on the per-class F1 in
   `flat_wall_shot_report.json`.

## References in repo

- `shot_detector/temporal_features.py` — feature builders.
- `shot_detector/pose_io.py` — pose CSV reader/writer.
- `shot_detector/extract_shots.py` — point of insertion for new
  per-frame features.
- `shot_detector/PLAYER_REID_PLAN.md` — reid integration plan
  (this work is its successor).
- `model_weights/flat_wall_shot_hparams.yaml` — current best XGB
  hyperparams from autoresearch.
- `model_weights/flat_wall_shot_report.json` — current per-class F1
  baseline to beat.
