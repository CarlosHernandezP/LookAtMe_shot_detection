"""
Extract per-shot pose CSVs from the upstream pipeline's intermediate artifacts
(poses_raw.csv + players_reid.csv) instead of re-running YOLO on the video.

Motivation: identical pose source to production inference, and no GPU needed.

Join: poses_raw.track_id == players_reid.object_id (verified).

poses_raw only contains the two CLOSE-TO-CAMERA players (by upstream design).
Candidates are therefore the two player_ids with pose coverage in the shot
window; the annotation label (left/right/bottom) picks the active one via the
image-space bbox foot rule on their reid bboxes, and the other candidate is
idle. The active player_id is locked for the whole 30-frame window. Shots
labeled 'top*' belong to far-court players (8 across all annotations) and are
skipped. No jitter / stationary / flickering filters are applied — the reid
identity lock replaces them.

Output format identical to extract_shots.py (save_pose_csv), so
train_shot_model.py consumes it unchanged.

Usage:
    uv run python -m shot_detector.extract_shots_from_intermediate \
        --output-dir shot_detector/data/extract_intermediate_poses
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from collections import Counter, defaultdict

import pandas as pd

from shot_detector.extract_shots import (
    BALL_TRAJECTORY_DIR,
    BALL_TRAJECTORY_MAP,
    FRAME_OFFSET,
    MIN_TRAJECTORY_LENGTH,
    PLAYERS_REID_MAP,
    PLAYERS_REID_ROOT,
    SHOTS_CSV_DIRS,
    find_closest_trajectory_to_player,
    handedness_mirror_active_idle,
    load_ball_trajectories,
    mirror_coco_pose_horizontal,
    save_pose_csv,
)
from shot_detector.players_reid_io import load_players_reid_by_frame
from shot_detector.utils import parse_shot_csv

WINDOW_BEFORE = 15
WINDOW_LEN = 30
MAX_FORWARD_FILL = 5


def resolve_match_cam(csv_path: str):
    name = os.path.basename(csv_path)
    for key, cam_map in PLAYERS_REID_MAP.items():
        if key in name:
            for cam in cam_map:
                if cam in name:
                    return key, cam
            if len(cam_map) == 1:
                return key, next(iter(cam_map))
    return None, None


def pick_pid_by_label(recs, label):
    """Image-space bbox foot rule (same semantics as identify_player)."""
    label = (label or "").lower().strip()
    best_pid, best_score = None, float("-inf")
    for r in recs:
        b = r["bbox"]
        fx, fy = 0.5 * (b[0] + b[2]), b[3]
        score = 0.0
        if "left" in label:
            score -= fx
        elif "right" in label:
            score += fx
        if "top" in label:
            score -= fy
        elif "bottom" in label:
            score += fy
        if score > best_score:
            best_score, best_pid = score, r["player_id"]
    return best_pid


def load_poses_for_frames(poses_raw_path: str, frames_needed: set):
    """{frame: {track_id: instance_dict}} for the requested frames only."""
    out = defaultdict(dict)
    for chunk in pd.read_csv(poses_raw_path, chunksize=200_000):
        sub = chunk[chunk["frame_num"].isin(frames_needed)]
        for row in sub.itertuples(index=False):
            try:
                kpts = ast.literal_eval(row.keypoints)
            except (ValueError, SyntaxError):
                continue
            if not kpts or len(kpts) < 17:
                continue
            keypoints = [[float(k[0]), float(k[1]), 1.0] for k in kpts[:17]]
            xs = [k[0] for k in keypoints]
            ys = [k[1] for k in keypoints]
            out[int(row.frame_num)][int(row.track_id)] = {
                "bbox": [[min(xs), min(ys), max(xs), max(ys)]],
                "keypoints": keypoints,
                "keypoint_scores": [1.0] * 17,
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-active-coverage", type=int, default=15,
                    help="Skip shot if active player has pose rows in fewer than N of 30 frames (pre forward-fill)")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    csv_files = {}
    for d in SHOTS_CSV_DIRS:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".csv"):
                    csv_files[f] = os.path.join(d, f)

    totals = Counter()
    for csv_file in sorted(csv_files):
        csv_path = csv_files[csv_file]
        match_key, cam = resolve_match_cam(csv_path)
        if not match_key:
            print(f"SKIP {csv_file}: no PLAYERS_REID_MAP entry")
            continue
        inter_dir = os.path.join(PLAYERS_REID_ROOT, os.path.dirname(PLAYERS_REID_MAP[match_key][cam]))
        reid_path = os.path.join(PLAYERS_REID_ROOT, PLAYERS_REID_MAP[match_key][cam])
        poses_raw_path = os.path.join(inter_dir, "poses_raw.csv")
        meta_path = os.path.join(inter_dir, "video_metadata.json")
        if not (os.path.exists(reid_path) and os.path.exists(poses_raw_path)):
            print(f"SKIP {csv_file}: missing intermediate artifacts in {inter_dir}")
            continue

        df = parse_shot_csv(csv_path)
        if df.empty:
            continue
        meta = json.load(open(meta_path))
        img_w, img_h = int(meta["opencv_width"]), int(meta["opencv_height"])
        video_name = os.path.splitext(csv_file)[0]
        if video_name.startswith("annotation_"):
            video_name = video_name[len("annotation_"):]

        frames_needed = set()
        for _, row in df.iterrows():
            s = max(0, int(row["FrameId"]) - WINDOW_BEFORE)
            frames_needed.update(range(s, s + WINDOW_LEN))

        print(f"{csv_file}: {len(df)} shots, loading poses for {len(frames_needed)} frames ...")
        poses_by_frame = load_poses_for_frames(poses_raw_path, frames_needed)
        reid_by_frame = load_players_reid_by_frame(reid_path)

        ball_csv = os.path.join(BALL_TRAJECTORY_DIR, BALL_TRAJECTORY_MAP[match_key][cam])
        trajectories = load_ball_trajectories(ball_csv, camera_id=cam) if os.path.exists(ball_csv) else {}

        for _, row in df.iterrows():
            shot_type = row["Shot"]
            center = int(row["FrameId"])
            label = str(row["Player"])
            start = max(0, center - WINDOW_BEFORE)
            totals["shots"] += 1

            if "top" in label.lower():
                totals["far_player_skipped"] += 1
                continue

            # object_id per (frame, pid) — reid is the id bridge
            pid2track = {}
            for fr in range(start, start + WINDOW_LEN):
                for r in reid_by_frame.get(fr, []):
                    pid2track[(fr, r["player_id"])] = r["object_id"]

            def pose_at(fr, pid):
                tid = pid2track.get((fr, pid))
                if tid is None:
                    return None
                return poses_by_frame.get(fr, {}).get(tid)

            # Candidates = the (max 2) close-to-camera pids: those with pose
            # coverage in the window, ranked by coverage.
            all_pids = {r["player_id"] for fr in range(start, start + WINDOW_LEN)
                        for r in reid_by_frame.get(fr, [])}
            pid_cov = {pid: sum(1 for fr in range(start, start + WINDOW_LEN) if pose_at(fr, pid))
                       for pid in all_pids}
            candidates = [pid for pid, cov in sorted(pid_cov.items(), key=lambda kv: -kv[1]) if cov > 0][:2]
            if not candidates:
                totals["no_pose_candidates"] += 1
                continue

            recs_c = [r for r in reid_by_frame.get(center, []) if r["player_id"] in candidates]
            if not recs_c:
                # no reid row at exact center for the candidates — use nearest frame
                for off in range(1, WINDOW_BEFORE + 1):
                    for fr in (center - off, center + off):
                        recs_c = [r for r in reid_by_frame.get(fr, []) if r["player_id"] in candidates]
                        if recs_c:
                            break
                    if recs_c:
                        break
            active_pid = pick_pid_by_label(recs_c, label)
            if active_pid is None:
                totals["no_reid_at_center"] += 1
                continue

            coverage = pid_cov.get(active_pid, 0)
            if coverage < args.min_active_coverage:
                totals["low_coverage_skipped"] += 1
                continue

            idle_pid = next((p for p in candidates if p != active_pid), None)

            active_seq, idle_seq = [], []
            last_a = last_i = None
            miss_a = miss_i = 0
            for i, fr in enumerate(range(start, start + WINDOW_LEN)):
                ma, mi = handedness_mirror_active_idle(csv_path, fr, label)

                pa = pose_at(fr, active_pid)
                if pa is not None:
                    last_a, miss_a = pa, 0
                elif last_a is not None and miss_a < MAX_FORWARD_FILL:
                    pa, miss_a = copy.deepcopy(last_a), miss_a + 1
                else:
                    pa, miss_a = None, miss_a + 1
                active_seq.append(mirror_coco_pose_horizontal(copy.deepcopy(pa), img_w) if (pa and ma) else pa)

                pi = pose_at(fr, idle_pid) if idle_pid is not None else None
                if pi is not None:
                    last_i, miss_i = pi, 0
                elif last_i is not None and miss_i < MAX_FORWARD_FILL:
                    pi, miss_i = copy.deepcopy(last_i), miss_i + 1
                else:
                    pi, miss_i = None, miss_i + 1
                idle_seq.append(mirror_coco_pose_horizontal(copy.deepcopy(pi), img_w) if (pi and mi) else pi)

            ball_positions = {}
            if trajectories:
                best = find_closest_trajectory_to_player(
                    trajectories, active_seq, start, WINDOW_LEN,
                    min_trajectory_length=MIN_TRAJECTORY_LENGTH, frame_offset=FRAME_OFFSET,
                )
                if best is not None:
                    for _, trow in trajectories[best].iterrows():
                        vf = int(trow["frame_number"]) - FRAME_OFFSET
                        if start <= vf < start + WINDOW_LEN:
                            ball_positions[vf] = (
                                float(trow["position_x"]), float(trow["position_y"]),
                                float(trow.get("confidence", 1.0)), bool(trow.get("is_interpolated", False)),
                            )

            base = f"{video_name}_{center}_{shot_type}_{label}"
            save_pose_csv(active_seq, os.path.join(args.output_dir, f"{base}_pose.csv"),
                          image_width=img_w, image_height=img_h,
                          ball_positions=ball_positions or None, start_frame=start)
            if any(p is not None for p in idle_seq):
                save_pose_csv(idle_seq, os.path.join(args.output_dir, f"{video_name}_{center}_idle_{label}_pose.csv"),
                              image_width=img_w, image_height=img_h,
                              ball_positions=ball_positions or None, start_frame=start)
            totals["exported"] += 1
            totals["with_ball"] += bool(ball_positions)

    print("\n=== Summary ===")
    for k, v in sorted(totals.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
