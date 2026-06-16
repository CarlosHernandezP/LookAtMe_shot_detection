"""
Verify the RE-PROCESSED pipeline data aligns with annotations on the CFR video.

Unlike verify_cfr_alignment.py (live YOLO), this overlays the pipeline's OWN
poses_raw skeletons + players_reid bboxes at each annotated shot, so it checks
the production data that will actually be used for extraction/training.

Join: poses_raw.track_id == players_reid.object_id. Sequential decode (CFR ->
frame N == poses_raw frame N). Active player (annotation label, foot rule on
reid bbox) drawn RED; annotation frame marked CONTACT.

Usage:
    uv run python -m shot_detector.tools.verify_pipeline_alignment \
        --video /home/ec2-user/data/matches/cfr/22-11-2025-18-10_rpi-LU-0002.mp4 \
        --intermediate /home/ec2-user/carlos/stats-poc/data/intermediate/22-11-2025-18-10_rpi-LU-0002/LU-0002 \
        --annotation '/home/ec2-user/data/shot_annotations/annotation_22-11-2025-18-10_rpi-LU-0002_*.csv' \
        --out /tmp/pipeverify_LU.mp4 --n-shots 20
"""
from __future__ import annotations

import argparse
import ast
import glob
import subprocess

import cv2
import numpy as np
import pandas as pd

from shot_detector.players_reid_io import load_players_reid_by_frame

COCO_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
              (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
PRE, POST = 15, 15


def load_poses_for_frames(poses_raw_path, frames):
    out = {}
    for chunk in pd.read_csv(poses_raw_path, chunksize=200_000):
        sub = chunk[chunk["frame_num"].isin(frames)]
        for r in sub.itertuples(index=False):
            try:
                kp = ast.literal_eval(r.keypoints)
            except (ValueError, SyntaxError):
                continue
            out.setdefault(int(r.frame_num), {})[int(r.track_id)] = kp
    return out


def pick_active(recs, label):
    label = (label or "").lower()
    best, bs = None, -1e9
    for r in recs:
        b = r["bbox"]
        fx, fy = 0.5 * (b[0] + b[2]), b[3]
        s = 0.0
        if "left" in label:
            s -= fx
        elif "right" in label:
            s += fx
        if "top" in label:
            s -= fy
        elif "bottom" in label:
            s += fy
        if s > bs:
            bs, best = s, r["player_id"]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--intermediate", required=True, help="<match>/<cam> dir with poses_raw.csv, players_reid.csv")
    ap.add_argument("--annotation", required=True, help="glob ok")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-shots", type=int, default=20)
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    ann = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(args.annotation))], ignore_index=True)
    ann = ann[ann.Shot != "Shot"].copy()
    ann["FrameId"] = ann.FrameId.astype(int)
    ann = ann.sort_values("FrameId").reset_index(drop=True)
    idx = np.linspace(0, len(ann) - 1, min(args.n_shots, len(ann))).astype(int)
    shots = ann.iloc[idx].reset_index(drop=True)
    windows = [(int(r.FrameId) - PRE, int(r.FrameId) + POST, int(r.FrameId), str(r.Shot), str(r.Player))
               for r in shots.itertuples()]
    last = max(w[1] for w in windows)
    frames_needed = set()
    for w in windows:
        frames_needed.update(range(w[0], w[1] + 1))

    print(f"{len(windows)} shots; loading pipeline data ...")
    reid_by_frame = load_players_reid_by_frame(f"{args.intermediate}/players_reid.csv")
    poses = load_poses_for_frames(f"{args.intermediate}/poses_raw.csv", frames_needed)

    # active player_id locked per shot at its contact frame
    active_pid_for = {}
    for w in windows:
        active_pid_for[w[2]] = pick_active(reid_by_frame.get(w[2], []), w[4])

    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    fn = -1
    while fn < last:
        if not cap.grab():
            break
        fn += 1
        aw = [w for w in windows if w[0] <= fn <= w[1]]
        if not aw:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        w = aw[0]
        act_pid = active_pid_for[w[2]]
        for r in reid_by_frame.get(fn, []):
            is_act = r["player_id"] == act_pid
            col = (0, 0, 255) if is_act else (0, 200, 0)
            th = 3 if is_act else 1
            b = [int(v) for v in r["bbox"]]
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), col, th)
            kp = poses.get(fn, {}).get(r["object_id"])
            if kp:
                for a, bb in COCO_EDGES:
                    xa, ya = int(kp[a][0]), int(kp[a][1])
                    xb, yb = int(kp[bb][0]), int(kp[bb][1])
                    if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                        cv2.line(frame, (xa, ya), (xb, yb), col, th)
                for p in kp:
                    if p[0] > 0 and p[1] > 0:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 3, col, -1)
        cv2.rectangle(frame, (0, 0), (W, 46), (0, 0, 0), -1)
        cv2.putText(frame, f"{w[3]} ({w[4]})  frame {fn}  off {fn - w[2]:+d}  pid={act_pid}",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        if fn == w[2]:
            cv2.putText(frame, "<<< CONTACT >>>", (W - 430, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            cv2.rectangle(frame, (2, 2), (W - 3, H - 3), (0, 0, 255), 6)
        vw.write(frame)
    cap.release()
    vw.release()
    h264 = args.out.replace(".mp4", "_h264.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.out, "-c:v", "libx264",
                    "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", h264], check=False)
    print(f"wrote {h264}")


if __name__ == "__main__":
    main()
