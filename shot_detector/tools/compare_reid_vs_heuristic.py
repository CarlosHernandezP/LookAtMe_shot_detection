"""
Compare old (heuristic) vs new (reid) active-player picks per shot.

For each shot present in BOTH:
    shot_detector/data/extract_all_with_clips_v2/<base>_pose.csv
    shot_detector/data/extract_all_with_clips_v2_reid/<base>_pose.csv
compute mean Euclidean delta between (feet_mid_x_abs, feet_mid_y_abs) across
the shot window. Rank shots by mean delta. Pick the top-N (default 10) where
the mean delta exceeds the threshold (default 50 px).

For each selected shot, render a single MP4 overlay:
    - backdrop = old MP4 in extract_all_with_clips_v2/<base>.mp4
    - red dot + label "OLD"  at feet_mid for the heuristic CSV
    - green dot + label "NEW" at feet_mid for the reid CSV
    - HUD with shot id, mean delta, fallback frames (from track_metrics.json)

Outputs:
    <out_dir>/compare_clips/<base>_compare.mp4
    <out_dir>/compare_summary.csv

Usage:
    uv run python shot_detector/tools/compare_reid_vs_heuristic.py \
        --old-dir shot_detector/data/extract_all_with_clips_v2 \
        --new-dir shot_detector/data/extract_all_with_clips_v2_reid \
        --out-dir shot_detector/exports/compare_reid_2026-06-05 \
        --top-n 10 --threshold 50
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import pandas as pd


def list_active_pose_csvs(directory: str) -> List[str]:
    out = []
    for f in os.listdir(directory):
        if not f.endswith("_pose.csv"):
            continue
        if "_idle_" in f:
            continue
        out.append(f)
    return sorted(out)


def base_from_pose_csv(name: str) -> str:
    return name[: -len("_pose.csv")]


def mean_centroid_delta(old_csv: str, new_csv: str) -> Tuple[Optional[float], int]:
    if not (os.path.exists(old_csv) and os.path.exists(new_csv)):
        return None, 0
    try:
        a = pd.read_csv(old_csv)
        b = pd.read_csv(new_csv)
    except Exception:
        return None, 0
    if "frame_num" not in a.columns or "frame_num" not in b.columns:
        return None, 0
    merged = a[["frame_num", "feet_mid_x_abs", "feet_mid_y_abs"]].merge(
        b[["frame_num", "feet_mid_x_abs", "feet_mid_y_abs"]],
        on="frame_num",
        suffixes=("_old", "_new"),
        how="inner",
    )
    merged = merged.dropna(
        subset=["feet_mid_x_abs_old", "feet_mid_y_abs_old", "feet_mid_x_abs_new", "feet_mid_y_abs_new"]
    )
    if merged.empty:
        return None, 0
    dx = merged["feet_mid_x_abs_old"] - merged["feet_mid_x_abs_new"]
    dy = merged["feet_mid_y_abs_old"] - merged["feet_mid_y_abs_new"]
    d = (dx ** 2 + dy ** 2).pow(0.5)
    return float(d.mean()), int(len(merged))


def load_track_metrics(dirpath: str, base: str) -> dict:
    p = os.path.join(dirpath, f"{base}_track_metrics.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def load_centroid_series(csv_path: str) -> Dict[int, Tuple[float, float]]:
    out: Dict[int, Tuple[float, float]] = {}
    if not os.path.exists(csv_path):
        return out
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return out
    for r in df.itertuples(index=False):
        try:
            fr = int(r.frame_num)
            x = float(r.feet_mid_x_abs)
            y = float(r.feet_mid_y_abs)
        except Exception:
            continue
        if math.isnan(x) or math.isnan(y):
            continue
        out[fr] = (x, y)
    return out


def render_compare_clip(
    mp4_old: str,
    centroids_old: Dict[int, Tuple[float, float]],
    centroids_new: Dict[int, Tuple[float, float]],
    out_path: str,
    header_lines: List[str],
) -> bool:
    if not os.path.exists(mp4_old):
        return False
    cap = cv2.VideoCapture(mp4_old)
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sorted_old = sorted(centroids_old.keys())
    base_frame = sorted_old[0] if sorted_old else 0
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fr_num = base_frame + idx
        if fr_num in centroids_old:
            x, y = centroids_old[fr_num]
            cv2.circle(frame, (int(x), int(y)), 14, (0, 0, 255), 3)
            cv2.putText(frame, "OLD", (int(x) + 18, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if fr_num in centroids_new:
            x, y = centroids_new[fr_num]
            cv2.circle(frame, (int(x), int(y)), 14, (0, 255, 0), 3)
            cv2.putText(frame, "NEW", (int(x) + 18, int(y) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.rectangle(frame, (0, 0), (w, 28 * len(header_lines) + 12), (0, 0, 0), -1)
        for li, line in enumerate(header_lines):
            cv2.putText(frame, line, (10, 22 + li * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        out.write(frame)
        idx += 1
    cap.release()
    out.release()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-dir", required=True)
    ap.add_argument("--new-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=50.0, help="Min mean centroid delta px to qualify")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "compare_clips"), exist_ok=True)

    old_csvs = list_active_pose_csvs(args.old_dir)
    new_set = set(list_active_pose_csvs(args.new_dir))
    common = [c for c in old_csvs if c in new_set]
    print(f"Active-pose CSVs: old={len(old_csvs)}  new={len(new_set)}  common={len(common)}")

    rows = []
    for cname in common:
        base = base_from_pose_csv(cname)
        delta, n = mean_centroid_delta(
            os.path.join(args.old_dir, cname),
            os.path.join(args.new_dir, cname),
        )
        if delta is None:
            continue
        rows.append({"base": base, "mean_delta_px": delta, "matched_frames": n})

    rows.sort(key=lambda r: r["mean_delta_px"], reverse=True)
    qualifying = [r for r in rows if r["mean_delta_px"] >= args.threshold]
    print(f"Shots with mean delta >= {args.threshold} px: {len(qualifying)} / {len(rows)}")
    picks = qualifying[: args.top_n]

    summary_path = os.path.join(args.out_dir, "compare_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "base", "mean_delta_px", "matched_frames",
            "old_active_player_label", "new_active_player_id", "new_fallback_frames_active"
        ])
        for r in picks:
            base = r["base"]
            old_tm = load_track_metrics(args.old_dir, base)
            new_tm = load_track_metrics(args.new_dir, base)
            w.writerow([
                base,
                f"{r['mean_delta_px']:.2f}",
                r["matched_frames"],
                old_tm.get("player_label", ""),
                (new_tm.get("reid") or {}).get("active_player_id", ""),
                (new_tm.get("reid") or {}).get("fallback_frames_active", ""),
            ])

    print(f"Wrote summary: {summary_path}")

    # Render compare clips
    for r in picks:
        base = r["base"]
        mp4 = os.path.join(args.old_dir, f"{base}.mp4")
        c_old = load_centroid_series(os.path.join(args.old_dir, f"{base}_pose.csv"))
        c_new = load_centroid_series(os.path.join(args.new_dir, f"{base}_pose.csv"))
        new_tm = load_track_metrics(args.new_dir, base)
        reid = new_tm.get("reid") or {}
        hdr = [
            f"{base}",
            f"mean_delta={r['mean_delta_px']:.1f}px  matched={r['matched_frames']}  "
            f"new_pid={reid.get('active_player_id')}  fallback={reid.get('fallback_frames_active')}",
        ]
        out_path = os.path.join(args.out_dir, "compare_clips", f"{base}_compare.mp4")
        ok = render_compare_clip(mp4, c_old, c_new, out_path, hdr)
        print(("OK   " if ok else "FAIL ") + out_path)

    print(f"Done. {len(picks)} compare clips in {os.path.join(args.out_dir, 'compare_clips')}")


if __name__ == "__main__":
    main()
