# Files that matter for the production pipeline

Rough mapping to your five goals:

1. **Shot labels from CSVs** — `utils.py` (`parse_shot_csv`, etc.), paths/config at top of `extract_shots.py`.
2. **Merge with ball trajectories** — `extract_shots.py` (ball maps + `ball_features.normalize_ball_features`).
3. **Dataset (`*_pose.csv`)** — output of `extract_shots.py` (pose + ball columns per frame).
4. **Train + export** — `train_shot_model.py` → `flat_wall_shot_xgb.joblib`, `flat_wall_shot_label_encoder.joblib`, `flat_wall_shot_model_meta.json`.
5. **Visualize** — `plot_shot_metrics.py` on `flat_wall_shot_cv_report.json` / `flat_wall_shot_report.json`.

**Aggregated class names** — `shot_mapper.py` (`DEFAULT_SHOT_MAPPING`).

**Features** — `temporal_features.py` (sequence → vector), `pose_io.py` (read CSV tensor).

Removed from this repo cleanup: cascade models, idle-threshold sweeps, UMAP experiments, duplicate trainers, old Random Forest `train_model` body.

## Data augmentation (removed)

The old `data_augmentation.py` applied horizontal mirroring, temporal warping, Gaussian noise, frame dropout, and sequence reversal. **Training no longer uses this module.** Reasons:

- Poses are **body-relative** and we only track **players near the camera**; treating left/right as interchangeable via **mirror** or **time reversal** can mislabel semantics (e.g. forehand vs backhand, shot direction).
- The current pipeline relies on **XGBoost + sample weights** for class imbalance instead of synthetic augmentations.

If you reintroduce augmentation later, prefer **light noise** or **temporal jitter** over mirror/reverse unless you explicitly model left/right invariance.
