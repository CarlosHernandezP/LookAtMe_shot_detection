"""
Render verification clips for mined idle windows (mine_idle_negatives.py).

Picks N mined idle CSVs per video, renders each 30-frame window with the
mined player's reid bbox + poses_raw skeleton in ORANGE ("MINED IDLE").
Sequential decode only (VFR-safe), half-speed playback.

Usage:
    uv run python -m shot_detector.tools.render_mined_idle_check \
        --mined-dir shot_detector/data/idle_hard_negatives \
        --out-dir shot_detector/exports/mined_idle_check \
        --per-video 3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shot_detector.extract_shots import PLAYERS_REID_MAP, PLAYERS_REID_ROOT
from shot_detector.extract_shots_from_intermediate import (
    WINDOW_BEFORE,
    WINDOW_LEN,
    load_poses_for_frames,
    resolve_match_cam,
)
from shot_detector.players_reid_io import load_players_reid_by_frame
from shot_detector.tools.render_intermediate_pose_check import draw_player

VIDEOS_DIR = "/home/ec2-user/data/matches"
NAME_RE = re.compile(r"^(?P<video>.+)_(?P<center>\d+)_idle_(?P<side>left|right)_pose\.csv$")


def video_file_for(video_name: str):
    """Mined names carry the annotation stem (may include _period<N> / _<N> suffix)."""
    base = re.sub(r"_period\d+$", "", video_name)
    base = re.sub(r"_\d$", "", base)
    for ext in (".mp4", ".MP4", ".mov", ".avi"):
        p = os.path.join(VIDEOS_DIR, base + ext)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-video", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    by_video = defaultdict(list)
    for f in sorted(os.listdir(args.mined_dir)):
        m = NAME_RE.match(f)
        if m:
            by_video[m.group("video")].append((int(m.group("center")), m.group("side"), f))

    manifest = []
    for video_name, items in sorted(by_video.items()):
        rng.shuffle(items)
        picks = sorted(items[: args.per_video])
        mk, cam = resolve_match_cam(video_name)
        if not mk:
            print(f"WARN no map for {video_name}")
            continue
        video_path = video_file_for(video_name)
        if not video_path:
            print(f"WARN no video for {video_name}")
            continue
        reid_path = os.path.join(PLAYERS_REID_ROOT, PLAYERS_REID_MAP[mk][cam])
        poses_raw_path = os.path.join(os.path.dirname(reid_path), "poses_raw.csv")
        frames_needed = set()
        for c, _, _ in picks:
            s = max(0, c - WINDOW_BEFORE)
            frames_needed.update(range(s, s + WINDOW_LEN))
        print(f"{video_name}: {len(picks)} windows")
        poses_by_frame = load_poses_for_frames(poses_raw_path, frames_needed)
        reid_by_frame = load_players_reid_by_frame(reid_path)

        windows = []
        for c, side, fname in picks:
            start = max(0, c - WINDOW_BEFORE)
            # mined player = candidate whose bbox-center-x side matches the token
            mined_pid = None
            for r in reid_by_frame.get(c, []):
                cx = 0.5 * (r["bbox"][0] + r["bbox"][2])
                r_side = "left" if cx < 960 else "right"
                tid_pose = poses_by_frame.get(c, {}).get(r["object_id"])
                if r_side == side and tid_pose is not None:
                    mined_pid = r["player_id"]
                    break
            windows.append({"start": start, "end": start + WINDOW_LEN, "center": c,
                            "side": side, "pid": mined_pid, "fname": fname, "writer": None})

        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        last_needed = max(wd["end"] for wd in windows)
        fr = 0
        while fr < last_needed:
            act = [wd for wd in windows if wd["start"] <= fr < wd["end"]]
            if act:
                ok, frame = cap.read()
            else:
                ok = cap.grab()
                frame = None
            if not ok:
                break
            for wd in act:
                if wd["writer"] is None:
                    out_name = wd["fname"].replace("_pose.csv", "_check.mp4")
                    wd["out_name"] = out_name
                    wd["writer"] = cv2.VideoWriter(
                        os.path.join(args.out_dir, out_name),
                        cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                img = frame.copy()
                if wd["pid"] is not None:
                    rec = next((r for r in reid_by_frame.get(fr, []) if r["player_id"] == wd["pid"]), None)
                    tid = rec["object_id"] if rec else None
                    pose = poses_by_frame.get(fr, {}).get(tid) if tid is not None else None
                    draw_player(img, pose, rec["bbox"] if rec else None,
                                (0, 165, 255), f"MINED IDLE pid={wd['pid']}")
                cv2.rectangle(img, (0, 0), (w, 40), (0, 0, 0), -1)
                cv2.putText(img, f"mined idle  side={wd['side']}  frame={fr}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                wd["writer"].write(img)
                if fr == wd["end"] - 1:
                    wd["writer"].release()
                    manifest.append({"clip": wd["out_name"], "video": video_name,
                                     "center": wd["center"], "side": wd["side"],
                                     "pid": wd["pid"]})
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
