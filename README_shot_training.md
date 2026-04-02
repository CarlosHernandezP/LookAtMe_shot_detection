# Shot training (deprecated guide)

The pipeline is documented in **`shot_detector/README.md`** and **`shot_detector/PIPELINE.md`**.

Train and export:

```bash
uv run python shot_detector/train_shot_model.py \
  --data-dir shot_detector/data_csv_only \
  --output-dir shot_detector/retrain_results_wall_flat \
  --cv-folds 5 --export-counts
```
