#!/usr/bin/env python3
"""
Binary success/fail labeling tool for shot clips.

Usage:
    python label_shot_success.py --clips-dir ./clips --labels-csv ./labels.csv

Keys (while clip plays in loop):
    y / 1  -> success
    n / 0  -> fail
    space  -> skip (no label written, revisit later)
    b      -> back (re-label previous clip)
    +/-    -> faster / slower playback
    r      -> restart current clip
    q / ESC -> quit (progress saved)

Labels are written incrementally to labels.csv with columns:
    clip_filename, label, timestamp
where label in {success, fail}. Already-labeled clips are skipped on resume.
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime

import cv2


def load_existing_labels(labels_csv):
    labeled = {}
    order = []
    if not os.path.exists(labels_csv):
        return labeled, order
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("clip_filename")
            label = row.get("label")
            if name:
                if name not in labeled:
                    order.append(name)
                labeled[name] = label
    return labeled, order


def append_label(labels_csv, clip_filename, label):
    is_new = not os.path.exists(labels_csv)
    with open(labels_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["clip_filename", "label", "timestamp"])
        writer.writerow([clip_filename, label, datetime.utcnow().isoformat(timespec="seconds") + "Z"])


def rewrite_labels(labels_csv, ordered_pairs):
    with open(labels_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_filename", "label", "timestamp"])
        for name, label in ordered_pairs:
            writer.writerow([name, label, datetime.utcnow().isoformat(timespec="seconds") + "Z"])


def list_clips(clips_dir):
    clips = []
    for fn in sorted(os.listdir(clips_dir)):
        if fn.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            clips.append(fn)
    return clips


def play_clip_until_key(clip_path, base_delay_ms, header_text):
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"[WARN] cannot open {clip_path}", file=sys.stderr)
        return ord("n"), base_delay_ms  # auto-mark as fail? safer: skip
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = max(1, base_delay_ms)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        return ord(" "), delay_ms
    h, w = frames[0].shape[:2]
    window = "label_shot_success"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    i = 0
    while True:
        fr = frames[i % len(frames)].copy()
        cv2.rectangle(fr, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(fr, header_text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(fr, "y=success  n=fail  space=skip  b=back  r=restart  +/- speed  q=quit",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(window, fr)
        k = cv2.waitKey(delay_ms) & 0xFF
        if k == 255:
            i += 1
            continue
        if k in (ord("y"), ord("1"), ord("n"), ord("0"), ord(" "), ord("b"), ord("q"), 27):
            return k, delay_ms
        if k == ord("r"):
            i = 0
            continue
        if k in (ord("+"), ord("=")):
            delay_ms = max(1, int(delay_ms * 0.7))
            continue
        if k in (ord("-"), ord("_")):
            delay_ms = min(500, int(delay_ms * 1.4) + 1)
            continue
        i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-dir", required=True, help="Directory containing .mp4 shot clips")
    ap.add_argument("--labels-csv", default="labels.csv", help="Output CSV (incremental, resumable)")
    ap.add_argument("--base-delay-ms", type=int, default=33, help="Initial frame delay (33 ~= 30fps)")
    ap.add_argument("--include-idle", action="store_true",
                    help="Also show clips containing '_idle_' in name (default: skip them)")
    args = ap.parse_args()

    if not os.path.isdir(args.clips_dir):
        print(f"clips-dir not found: {args.clips_dir}", file=sys.stderr)
        sys.exit(1)

    clips = list_clips(args.clips_dir)
    if not args.include_idle:
        clips = [c for c in clips if "_idle_" not in c]
    if not clips:
        print("No clips found.")
        return

    labeled, order = load_existing_labels(args.labels_csv)
    pending = [c for c in clips if c not in labeled]
    print(f"Total clips: {len(clips)}  already labeled: {len(labeled)}  pending: {len(pending)}")

    delay_ms = args.base_delay_ms
    idx = 0
    while idx < len(pending):
        name = pending[idx]
        path = os.path.join(args.clips_dir, name)
        header = f"[{idx+1}/{len(pending)}] {name}"
        key, delay_ms = play_clip_until_key(path, delay_ms, header)

        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            print(f"skip: {name}")
            idx += 1
            continue
        if key == ord("b"):
            # back one — drop last appended label from CSV
            if order:
                last = order.pop()
                labeled.pop(last, None)
                rewrite_labels(args.labels_csv, [(n, labeled[n]) for n in order])
                # put `last` back at the front of pending
                pending.insert(idx, last)
                print(f"back: re-label {last}")
            continue
        label = "success" if key in (ord("y"), ord("1")) else "fail"
        append_label(args.labels_csv, name, label)
        labeled[name] = label
        order.append(name)
        print(f"{label}: {name}")
        idx += 1

    cv2.destroyAllWindows()
    n_success = sum(1 for v in labeled.values() if v == "success")
    n_fail = sum(1 for v in labeled.values() if v == "fail")
    print(f"Done. labeled={len(labeled)} success={n_success} fail={n_fail}  csv={args.labels_csv}")


if __name__ == "__main__":
    main()
