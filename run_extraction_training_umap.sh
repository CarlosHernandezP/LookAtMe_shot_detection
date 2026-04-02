#!/bin/bash
# After extraction finishes, train the flat XGBoost shot classifier.

set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):${PYTHONPATH}"

echo "Waiting for extract_shots to finish..."
while pgrep -f "python.*extract_shots" >/dev/null 2>&1; do
  echo "Extraction still running... (30s)"
  sleep 30
done

echo "Training (5-fold CV + full-data export)..."
uv run python shot_detector/train_shot_model.py \
  --data-dir shot_detector/data_csv_only \
  --output-dir shot_detector/retrain_results_wall_flat \
  --cv-folds 5 \
  --export-counts

echo "Done. Artifacts under shot_detector/retrain_results_wall_flat/"
