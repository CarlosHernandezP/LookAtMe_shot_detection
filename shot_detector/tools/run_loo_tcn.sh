#!/usr/bin/env bash
# Train one leave-one-match-out TCN (no non-idle weight) per unique match,
# then a full-data deploy model. All use the corrected shot mapping
# (backhand_contrapared -> wall_shot). Each LOO model excludes that match's
# shot CSVs + mined idles, so the held-out match is a leakage-free FP test set.
set -euo pipefail
cd /home/ec2-user/carlos/LookAtMe_shot_detection

SRC=shot_detector/data/extract_intermediate_poses_hardneg
TOKENS=(0529b769 15-11-2025 18dda9d2 22-11-2025 a145dd19)

COMMON="--arch tcn --cv-folds 3 --epochs 200 --patience 35 --augment --label-smoothing 0.1 --channels 128 --kernel-size 5 --dropout 0.3 --save-model"

for tok in "${TOKENS[@]}"; do
  LOO="shot_detector/data/loo_${tok}_hardneg"
  rm -rf "$LOO"; mkdir -p "$LOO"
  find "$SRC" -maxdepth 1 -name "*_pose.csv" ! -name "*${tok}*" -exec ln -t "$LOO" {} +
  echo "[$tok] train on $(ls "$LOO"/*_pose.csv | wc -l), hold out $(ls "$SRC"/*${tok}*_pose.csv 2>/dev/null | wc -l)"
  uv run python shot_detector/train_sequence_model.py \
    --data-dir "$LOO" --output-dir "shot_detector/runs/tcn_loo_${tok}" $COMMON \
    > "logs/tcn_loo_${tok}.log" 2>&1
  echo "[$tok] done: $(grep -E 'OOF|Saved model' logs/tcn_loo_${tok}.log | tail -1)"
done

echo "[FULL] deploy model on all data"
uv run python shot_detector/train_sequence_model.py \
  --data-dir "$SRC" --output-dir shot_detector/runs/tcn_final_noweight $COMMON \
  > logs/tcn_final_noweight.log 2>&1
echo "[FULL] done: $(grep -E 'OOF|Saved model' logs/tcn_final_noweight.log | tail -1)"
echo ALL_TRAININGS_DONE
