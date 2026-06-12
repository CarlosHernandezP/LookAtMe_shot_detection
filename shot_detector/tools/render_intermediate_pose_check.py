"""
Render verification clips for the intermediate-poses extraction: N clips per
shot class, with the chosen ACTIVE (green) and IDLE (blue) players overlaid
(reid bbox + poses_raw skeleton). Purpose: visually confirm active/idle
assignment of the close-to-camera pair.

Writes at 15 fps (half speed) for easier inspection.

Usage:
    uv run python -m shot_detector.tools.render_intermediate_pose_check \
        --out-dir shot_detector/exports/intermediate_pose_check \
        --per-class 3
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
from collections import defaultdict

import cv2
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shot_detector.extract_shots import PLAYERS_REID_MAP, PLAYERS_REID_ROOT, SHOTS_CSV_DIRS
from shot_detector.extract_shots_from_intermediate import (
    WINDOW_BEFORE,
    WINDOW_LEN,
    load_poses_for_frames,
    pick_pid_by_label,
    resolve_match_cam,
)
from shot_detector.players_reid_io import load_players_reid_by_frame
from shot_detector.utils import get_video_path, parse_shot_csv

VIDEOS_DIRS = ["/home/ec2-user/data/matches/"]

COCO_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]


def draw_player(frame, inst, bbox, color, tag):
    if bbox:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if inst:
        kpts = inst["keypoints"]
        for a, b in COCO_EDGES:
            xa, ya = int(kpts[a][0]), int(kpts[a][1])
            xb, yb = int(kpts[b][0]), int(kpts[b][1])
            if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                cv2.line(frame, (xa, ya), (xb, yb), color, 2)
        for k in kpts:
            if k[0] > 0 and k[1] > 0:
                cv2.circle(frame, (int(k[0]), int(k[1])), 3, color, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    # Gather all (csv_path, shot_row) per shot class
    by_class = defaultdict(list)
    csv_files = {}
    for d in SHOTS_CSV_DIRS:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".csv"):
                    csv_files[f] = os.path.join(d, f)
    for csv_file, csv_path in sorted(csv_files.items()):
        mk, cam = resolve_match_cam(csv_path)
        if not mk:
            continue
        df = parse_shot_csv(csv_path)
        for _, row in df.iterrows():
            if "top" in str(row["Player"]).lower():
                continue
            by_class[str(row["Shot"])].append((csv_path, mk, cam, int(row["FrameId"]), str(row["Player"])))

    picks = []
    for cls, items in sorted(by_class.items()):
        rng.shuffle(items)
        picks.extend((cls, *it) for it in items[: args.per_class])
    print(f"{len(by_class)} classes, {len(picks)} clips to render")

    # Group picks per (match, cam) to load artifacts once
    per_src = defaultdict(list)
    for p in picks:
        per_src[(p[2], p[3])].append(p)

    manifest = []
    for (mk, cam), items in per_src.items():
        reid_path = os.path.join(PLAYERS_REID_ROOT, PLAYERS_REID_MAP[mk][cam])
        inter_dir = os.path.dirname(reid_path)
        poses_raw_path = os.path.join(inter_dir, "poses_raw.csv")
        frames_needed = set()
        for _, _, _, _, center, _ in items:
            s = max(0, center - WINDOW_BEFORE)
            frames_needed.update(range(s, s + WINDOW_LEN))
        print(f"loading artifacts for {mk}/{cam} ({len(frames_needed)} frames) ...")
        poses_by_frame = load_poses_for_frames(poses_raw_path, frames_needed)
        reid_by_frame = load_players_reid_by_frame(reid_path)

        video_path = None
        for _, csv_path, _, _, _, _ in items:
            video_path = get_video_path(csv_path, VIDEOS_DIRS)
            if video_path and os.path.exists(video_path):
                break
        if not video_path or not os.path.exists(video_path):
            print(f"  WARN no video for {mk}/{cam}, skipping {len(items)} clips")
            continue

        # Plan all windows up-front, then make ONE sequential decode pass.
        # cv2 CAP_PROP_POS_FRAMES seeking is frame-INACCURATE on the VFR videos
        # (r_frame_rate=250 vs avg ~30.005): poses_raw frame_num is a sequential
        # decode index, so only sequential reading lines the overlays up.
        windows = []  # (start, end, clip dict)
        for cls, csv_path, _, _, center, label in items:
            start = max(0, center - WINDOW_BEFORE)
            pid2track = {}
            for fr in range(start, start + WINDOW_LEN):
                for r in reid_by_frame.get(fr, []):
                    pid2track[(fr, r["player_id"])] = r["object_id"]

            def pose_at(fr, pid, _p2t=pid2track):
                tid = _p2t.get((fr, pid))
                return poses_by_frame.get(fr, {}).get(tid) if tid is not None else None

            all_pids = {r["player_id"] for fr in range(start, start + WINDOW_LEN)
                        for r in reid_by_frame.get(fr, [])}
            pid_cov = {pid: sum(1 for fr in range(start, start + WINDOW_LEN) if pose_at(fr, pid))
                       for pid in all_pids}
            candidates = [pid for pid, cov in sorted(pid_cov.items(), key=lambda kv: -kv[1]) if cov > 0][:2]
            recs_c = [r for r in reid_by_frame.get(center, []) if r["player_id"] in candidates]
            active_pid = pick_pid_by_label(recs_c, label)
            idle_pid = next((p for p in candidates if p != active_pid), None)
            if active_pid is None:
                print(f"  WARN no active pid for {cls}@{center}, skipping")
                continue
            out_name = f"{cls}_{os.path.basename(csv_path)[:-4]}_{center}_{label}.mp4"
            windows.append({
                "start": start, "end": start + WINDOW_LEN, "cls": cls, "label": label,
                "active_pid": active_pid, "idle_pid": idle_pid, "pose_at": pose_at,
                "out_name": out_name, "writer": None,
            })

        if not windows:
            continue
        last_needed = max(wd["end"] for wd in windows)
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fr = 0
        while fr < last_needed:
            active_windows = [wd for wd in windows if wd["start"] <= fr < wd["end"]]
            if active_windows:
                ok, frame = cap.read()
            else:
                ok = cap.grab()
                frame = None
            if not ok:
                break
            for wd in active_windows:
                if wd["writer"] is None:
                    wd["writer"] = cv2.VideoWriter(
                        os.path.join(args.out_dir, wd["out_name"]),
                        cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                img = frame.copy()
                reid_recs = {r["player_id"]: r for r in reid_by_frame.get(fr, [])}
                a_rec = reid_recs.get(wd["active_pid"])
                draw_player(img, wd["pose_at"](fr, wd["active_pid"]),
                            a_rec["bbox"] if a_rec else None, (0, 255, 0), f"ACTIVE pid={wd['active_pid']}")
                if wd["idle_pid"] is not None:
                    i_rec = reid_recs.get(wd["idle_pid"])
                    draw_player(img, wd["pose_at"](fr, wd["idle_pid"]),
                                i_rec["bbox"] if i_rec else None, (255, 128, 0), f"IDLE pid={wd['idle_pid']}")
                cv2.rectangle(img, (0, 0), (w, 40), (0, 0, 0), -1)
                cv2.putText(img, f"{wd['cls']}  label={wd['label']}  frame={fr}  ACTIVE=green IDLE=blue",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                wd["writer"].write(img)
                if fr == wd["end"] - 1:
                    wd["writer"].release()
                    manifest.append({"class": wd["cls"], "clip": wd["out_name"], "label": wd["label"],
                                     "active_pid": int(wd["active_pid"]),
                                     "idle_pid": int(wd["idle_pid"]) if wd["idle_pid"] is not None else None})
                    print(f"  OK {wd['out_name']}")
            fr += 1
        for wd in windows:
            if wd["writer"] is not None:
                try:
                    wd["writer"].release()
                except Exception:
                    pass
        cap.release()

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{len(manifest)} clips -> {args.out_dir}")


if __name__ == "__main__":
    main()
