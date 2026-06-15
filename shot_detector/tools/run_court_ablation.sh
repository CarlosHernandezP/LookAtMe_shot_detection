#!/usr/bin/env bash
set -uo pipefail
cd /home/ec2-user/carlos/LookAtMe_shot_detection
while ! grep -q REEXTRACT_DONE logs/reextract_court.log 2>/dev/null; do sleep 30; done
echo "extraction done: $(grep merged logs/reextract_court.log)"
DATA=shot_detector/data/court_hardneg
HELD=18dda9d2
COMMON="--arch tcn --epochs 200 --patience 35 --augment --label-smoothing 0.1 --channels 128 --kernel-size 5 --dropout 0.3 --heldout-token $HELD --output-dir shot_detector/runs/court_ablation"
mkdir -p shot_detector/runs/court_ablation
for cf in "" "court_y" "court_y,court_x" "court_y,court_x,dist_to_near_baseline"; do
  tag="${cf:-base}"
  echo "######## ABLATION court=[$tag] ########"
  uv run python shot_detector/train_sequence_model.py --data-dir $DATA $COMMON --court-features "$cf" 2>&1 | grep -E "HELD-OUT|serve|overall accuracy" 
done
echo COURT_ABLATION_DONE
