"""
Verify CFR alignment: for N annotated shots, sequentially decode the CFR video
to each shot window, run YOLO-pose on the frames, draw every player's bbox +
skeleton, highlight the ACTIVE player (the annotation label's player, by foot
position) in RED. The annotation center frame is marked CONTACT.

Self-contained: needs only the CFR video + annotations + YOLO-pose. Does NOT
use the pipeline intermediates (which are still VFR-indexed), so the overlay is
aligned to the CFR frames by construction. If the active player is mid-swing at
the CONTACT frame, the CFR conversion fixed the alignment.

Usage:
    uv run python -m shot_detector.tools.verify_cfr_alignment \
        --video /home/ec2-user/data/matches/cfr/22-11-2025-18-10_rpi-LU-0002.mp4 \
        --annotation '/home/ec2-user/data/shot_annotations/annotation_22-11-2025-18-10_rpi-LU-0002_*.csv' \
        --out /tmp/cfr_verify_LU.mp4 --n-shots 20
"""
from __future__ import annotations

import argparse
import glob

import cv2
import numpy as np
import pandas as pd

from shot_detector.ultralytics_pose import init_ultralytics_pose

COCO_EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
              (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
PRE, POST = 15, 15


def bbox_from_kpts(kp):
    xs = [p[0] for p in kp if p[0] > 0]
    ys = [p[1] for p in kp if p[1] > 0]
    if not xs:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def pick_active(boxes, label):
    label = (label or "").lower()
    best, bs = -1, -1e9
    for i, b in enumerate(boxes):
        if b is None:
            continue
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
            bs, best = s, i
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--annotation", required=True, help="glob ok")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-shots", type=int, default=20)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ann = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(args.annotation))], ignore_index=True)
    ann = ann[ann.Shot != "Shot"].copy()
    ann["FrameId"] = ann.FrameId.astype(int)
    ann = ann.sort_values("FrameId").reset_index(drop=True)
    # spread N shots across the whole match (so drift at the end is also tested)
    idx = np.linspace(0, len(ann) - 1, min(args.n_shots, len(ann))).astype(int)
    shots = ann.iloc[idx].reset_index(drop=True)
    windows = [(int(r.FrameId) - PRE, int(r.FrameId) + POST, int(r.FrameId), str(r.Shot), str(r.Player))
               for r in shots.itertuples()]
    last = max(w[1] for w in windows)
    print(f"{len(windows)} shots, frames up to {last}")

    inf = init_ultralytics_pose()
    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    fn = -1
    while fn < last:
        if not cap.grab():
            break
        fn += 1
        active_w = [w for w in windows if w[0] <= fn <= w[1]]
        if not active_w:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        w = active_w[0]
        res = next(inf(frame, return_vis=False))
        preds = res["predictions"][0] or []
        boxes, kpts = [], []
        for inst in preds:
            kp = inst["keypoints"]
            kpts.append(kp)
            bb = inst.get("bbox")
            bb = bb[0] if bb and isinstance(bb[0], (list, tuple)) else (bbox_from_kpts(kp))
            boxes.append(list(bb) if bb is not None else None)
        act = pick_active(boxes, w[4]) if fn == w[2] or True else -1
        for i, (bb, kp) in enumerate(zip(boxes, kpts)):
            col = (0, 0, 255) if i == act else (0, 200, 0)
            th = 3 if i == act else 1
            if bb:
                cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), col, th)
            for a, b in COCO_EDGES:
                xa, ya = int(kp[a][0]), int(kp[a][1])
                xb, yb = int(kp[b][0]), int(kp[b][1])
                if xa > 0 and ya > 0 and xb > 0 and yb > 0:
                    cv2.line(frame, (xa, ya), (xb, yb), col, th)
            for p in kp:
                if p[0] > 0 and p[1] > 0:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 3, col, -1)
        contact = fn == w[2]
        cv2.rectangle(frame, (0, 0), (W, 46), (0, 0, 0), -1)
        msg = f"{w[3]} ({w[4]})  frame {fn}  off {fn - w[2]:+d}"
        cv2.putText(frame, msg, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 255), 2)
        if contact:
            cv2.putText(frame, "<<< CONTACT (annotation frame) >>>", (W - 760, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            cv2.rectangle(frame, (2, 2), (W - 3, H - 3), (0, 0, 255), 6)
        vw.write(frame)
    cap.release()
    vw.release()
    # h264 for easy playback
    import subprocess
    h264 = args.out.replace(".mp4", "_h264.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.out, "-c:v", "libx264",
                    "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", h264], check=False)
    print(f"wrote {h264}")


if __name__ == "__main__":
    main()
