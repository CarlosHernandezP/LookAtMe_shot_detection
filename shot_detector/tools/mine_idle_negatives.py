"""
Mine hard-negative idle windows from full-match intermediate artifacts.

Production over-predicts shots because training 'idle' only shows the
non-hitting player DURING shots. This tool samples 30-frame windows far away
from every annotated shot (walking, waiting, repositioning) for both
close-to-camera players and writes them as idle pose CSVs in the same format
as extract_shots_from_intermediate.py.

Output filenames: <video>_<center>_idle_<left|right>_pose.csv — the side is
derived from the player's bbox center at the window center frame.
shot_mapper.extract_shot_type_from_filename only accepts idle when the token
after 'idle' is a valid player position, so the side token is mandatory.
Collision with shot-extraction idle files is impossible: mined centers are
>= --min-gap-frames away from every annotated shot frame.

Usage:
    uv run python -m shot_detector.tools.mine_idle_negatives \
        --output-dir shot_detector/data/idle_hard_negatives \
        --per-video 180 --min-gap-frames 60
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shot_detector.extract_shots import PLAYERS_REID_MAP, PLAYERS_REID_ROOT, SHOTS_CSV_DIRS, save_pose_csv
from shot_detector.extract_shots_from_intermediate import (
    MAX_FORWARD_FILL,
    WINDOW_BEFORE,
    WINDOW_LEN,
    _court_xy,
    append_court_columns,
    load_poses_for_frames,
    resolve_match_cam,
)
from shot_detector.players_reid_io import load_players_reid_by_frame
from shot_detector.utils import parse_shot_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--per-video", type=int, default=180,
                    help="Idle windows sampled per (match, cam); each yields up to 2 samples (both close players)")
    ap.add_argument("--min-gap-frames", type=int, default=60,
                    help="Window center must be at least this far from every annotated shot frame (60 = 2 s)")
    ap.add_argument("--min-coverage", type=int, default=20,
                    help="Player must have pose rows in at least N of the 30 frames")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)

    # Group annotation CSVs by (match, cam)
    per_src = {}
    for d in SHOTS_CSV_DIRS:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.endswith(".csv"):
                continue
            p = os.path.join(d, f)
            mk, cam = resolve_match_cam(p)
            if mk:
                per_src.setdefault((mk, cam), []).append(p)

    totals = Counter()
    for (mk, cam), csv_paths in sorted(per_src.items()):
        reid_path = os.path.join(PLAYERS_REID_ROOT, PLAYERS_REID_MAP[mk][cam])
        inter_dir = os.path.dirname(reid_path)
        poses_raw_path = os.path.join(inter_dir, "poses_raw.csv")
        meta_path = os.path.join(inter_dir, "video_metadata.json")
        if not (os.path.exists(reid_path) and os.path.exists(poses_raw_path)):
            print(f"SKIP {mk}/{cam}: missing artifacts")
            continue
        meta = json.load(open(meta_path))
        img_w, img_h = int(meta["opencv_width"]), int(meta["opencv_height"])
        n_frames = int(meta["processed_frame_count"])

        shot_frames = []
        video_name = None
        for p in csv_paths:
            df = parse_shot_csv(p)
            shot_frames.extend(int(v) for v in df["FrameId"].tolist())
            vn = os.path.splitext(os.path.basename(p))[0]
            if vn.startswith("annotation_"):
                vn = vn[len("annotation_"):]
            video_name = vn  # same video for all periods of this (mk, cam)
        shot_frames.sort()

        def far_from_shots(center):
            return all(abs(center - sf) >= args.min_gap_frames for sf in shot_frames)

        # Sample candidate centers
        lo = WINDOW_BEFORE + 1
        hi = n_frames - WINDOW_LEN - 1
        centers = []
        attempts = 0
        while len(centers) < args.per_video and attempts < args.per_video * 50:
            attempts += 1
            c = rng.randint(lo, hi)
            if far_from_shots(c) and all(abs(c - e) >= WINDOW_LEN for e in centers):
                centers.append(c)
        centers.sort()
        print(f"{mk}/{cam}: {len(centers)} idle windows (video={video_name}, shots={len(shot_frames)})")

        frames_needed = set()
        for c in centers:
            s = max(0, c - WINDOW_BEFORE)
            frames_needed.update(range(s, s + WINDOW_LEN))
        poses_by_frame = load_poses_for_frames(poses_raw_path, frames_needed)
        reid_by_frame = load_players_reid_by_frame(reid_path)

        for c in centers:
            start = max(0, c - WINDOW_BEFORE)
            pid2track = {}
            for fr in range(start, start + WINDOW_LEN):
                for r in reid_by_frame.get(fr, []):
                    pid2track[(fr, r["player_id"])] = r["object_id"]

            def pose_at(fr, pid):
                tid = pid2track.get((fr, pid))
                return poses_by_frame.get(fr, {}).get(tid) if tid is not None else None

            all_pids = {r["player_id"] for fr in range(start, start + WINDOW_LEN)
                        for r in reid_by_frame.get(fr, [])}
            pid_cov = {pid: sum(1 for fr in range(start, start + WINDOW_LEN) if pose_at(fr, pid))
                       for pid in all_pids}
            candidates = [pid for pid, cov in sorted(pid_cov.items(), key=lambda kv: -kv[1])
                          if cov >= args.min_coverage][:2]
            for pid in candidates:
                seq, court = [], []
                last, miss = None, 0
                for fr in range(start, start + WINDOW_LEN):
                    court.append(_court_xy(reid_by_frame, fr, pid))
                    p = pose_at(fr, pid)
                    if p is not None:
                        last, miss = p, 0
                    elif last is not None and miss < MAX_FORWARD_FILL:
                        p, miss = copy.deepcopy(last), miss + 1
                    else:
                        p, miss = None, miss + 1
                    seq.append(p)
                # side token from bbox center x (parser requires a valid position)
                side = "left"
                for r in reid_by_frame.get(c, []):
                    if r["player_id"] == pid:
                        side = "left" if 0.5 * (r["bbox"][0] + r["bbox"][2]) < img_w / 2 else "right"
                        break
                out = os.path.join(args.output_dir, f"{video_name}_{c}_idle_{side}_pose.csv")
                if os.path.exists(out):  # both pids same side token at same center
                    out = os.path.join(args.output_dir, f"{video_name}_{c + 1}_idle_{side}_pose.csv")
                save_pose_csv(seq, out, image_width=img_w, image_height=img_h,
                              ball_positions=None, start_frame=start)
                append_court_columns(out, court)
                totals["samples"] += 1
        totals["windows"] += len(centers)

    print("\n=== Summary ===")
    for k, v in sorted(totals.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
