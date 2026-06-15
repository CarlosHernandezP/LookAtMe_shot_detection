#!/usr/bin/env bash
set -uo pipefail
SRC=/home/ec2-user/data/matches
OUT=/home/ec2-user/data/matches/cfr
# LU first (the critical verification), then the two 0529 cameras
VIDS=(
 "22-11-2025-18-10_rpi-LU-0002.mp4"
 "0529b769-125d-4a22-bcee-b1707b87447e_1767949074894_rpi-BO-0001.mp4"
 "0529b769-125d-4a22-bcee-b1707b87447e_1767949074880_rpi-BO-0002.mp4"
)
for v in "${VIDS[@]}"; do
  echo "=== converting $v ($(date +%H:%M:%S)) ==="
  ffmpeg -y -loglevel error -stats -i "$SRC/$v" \
    -fps_mode cfr -r 30 -c:v libx264 -crf 20 -preset fast \
    -c:a copy "$OUT/$v"
  r=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate,avg_frame_rate -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$OUT/$v" 2>/dev/null)
  echo "  DONE $v  r,avg,nframes=$r"
done
echo VFR2CFR_DONE
