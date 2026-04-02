# Shot detector pipeline

End-to-end flow: **annotation CSVs + ball trajectories → pose CSVs → XGBoost classifier**.

## What to run

| Step | Script | Purpose |
|------|--------|---------|
| 1–3 | `extract_shots.py` | Read shot CSVs, merge ball trajectories, run pose model, write `*_pose.csv` datasets. |
| 4 | `train_shot_model.py` | Build temporal features, CV or train/val metrics, **train on full data**, export `flat_wall_shot_xgb.joblib` + `flat_wall_shot_label_encoder.joblib`. |
| 5 | `plot_shot_metrics.py` | Figures from the JSON reports written by training. |

Supporting modules (not run directly): `shot_mapper.py` (label mapping), `ball_features.py`, `temporal_features.py`, `pose_io.py` (load pose CSV rows), `utils.py` (CSV / calibration helpers used by extraction).

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
