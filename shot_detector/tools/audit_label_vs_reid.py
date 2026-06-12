"""
Audit: does the heuristic identify_player()'s pick agree with the court-rule
applied directly to players_reid.csv?

For each shot:
- Read track_metrics.json -> reid.active_player_id (the player_id the
  heuristic-picked pose was matched to at the center frame).
- Load reid records at the center frame, apply the same label rule
  ('bottom'/'top'/'left'/'right') but on court_x / court_y of the
  reid candidates. That is what the player should be.
- Disagreement = the two player_ids differ.

Output:
    <out_dir>/label_vs_reid_disagreements.csv
columns: base, player_label, heuristic_pid, court_rule_pid, n_reid_at_center

This is the dataset for the "10 visualization examples".
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

import pandas as pd

# Reuse the loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shot_detector.players_reid_io import load_players_reid_by_frame  # noqa: E402


PLAYERS_REID_ROOT = "/home/ec2-user/carlos/LookAtMeProtoApp/data/intermediate"
PLAYERS_REID_MAP = {
    "0529b769-125d-4a22-bcee-b1707b87447e": {
        "BO-0001": "0529b769-125d-4a22-bcee-b1707b87447e/BO-0001/players_reid.csv",
        "BO-0002": "0529b769-125d-4a22-bcee-b1707b87447e/BO-0002/players_reid.csv",
    },
    "15-11-2025-15-57_rpi-BO-0001": {
        "BO-0001": "15-11-2025-15-57_rpi-BO-0001/BO-0001/players_reid.csv",
    },
    "22-11-2025-18-10_rpi-LU-0002": {
        "LU-0002": "22-11-2025-18-10_rpi-LU-0002/LU-0002/players_reid.csv",
    },
    "18dda9d2-baba-4920-a642-d0a9838d01f3": {
        "BO-2226": "18dda9d2-baba-4920-a642-d0a9838d01f3/BO-2226/players_reid.csv",
    },
}


def resolve_reid_path(video_name: str) -> Optional[str]:
    for k, cam_map in PLAYERS_REID_MAP.items():
        if k in video_name:
            cam = None
            for cam_id in cam_map:
                if cam_id in video_name:
                    cam = cam_id
                    break
            if cam is None and len(cam_map) == 1:
                cam = next(iter(cam_map))
            if cam:
                return os.path.join(PLAYERS_REID_ROOT, cam_map[cam])
    return None


def court_rule_pid(reid_recs: List[dict], label: str) -> Optional[int]:
    """
    Apply the same heuristic identify_player() uses, but on reid BBOX
    image-space coordinates (foot midpoint). Matches identify_player()'s
    own logic: image x=0 left, image y=0 top.
    """
    if not reid_recs:
        return None
    label = (label or "").lower().strip()
    best_pid = None
    best_score = float("-inf")
    for r in reid_recs:
        b = r["bbox"]
        foot_x = 0.5 * (b[0] + b[2])
        foot_y = b[3]
        score = 0.0
        if "left" in label:
            score -= foot_x
        elif "right" in label:
            score += foot_x
        if "top" in label:
            score -= foot_y
        elif "bottom" in label:
            score += foot_y
        if score > best_score:
            best_score = score
            best_pid = r["player_id"]
    return best_pid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reid-extract-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Cache reid by (match, cam) to avoid reloading per shot
    reid_cache: Dict[str, Dict[int, List[dict]]] = {}

    out_rows = []
    n_total = 0
    n_agree = 0
    n_no_reid = 0
    for fn in sorted(os.listdir(args.reid_extract_dir)):
        if not fn.endswith("_track_metrics.json"):
            continue
        if "_idle_" in fn:
            continue
        with open(os.path.join(args.reid_extract_dir, fn)) as f:
            tm = json.load(f)
        video_name = tm.get("video_name")
        center_frame = tm.get("center_frame")
        label = tm.get("player_label", "")
        reid = tm.get("reid") or {}
        heur_pid = reid.get("active_player_id")
        if not (video_name and center_frame is not None):
            continue
        n_total += 1
        reid_path = resolve_reid_path(video_name)
        if not reid_path:
            n_no_reid += 1
            continue
        if reid_path not in reid_cache:
            reid_cache[reid_path] = load_players_reid_by_frame(reid_path)
        recs = reid_cache[reid_path].get(int(center_frame), [])
        cr_pid = court_rule_pid(recs, label)
        if cr_pid is None or heur_pid is None:
            n_no_reid += 1
            continue
        if cr_pid == heur_pid:
            n_agree += 1
            continue
        base = fn[: -len("_track_metrics.json")]
        out_rows.append({
            "base": base,
            "video_name": video_name,
            "center_frame": int(center_frame),
            "player_label": label,
            "heuristic_pid": int(heur_pid),
            "court_rule_pid": int(cr_pid),
            "n_reid_at_center": len(recs),
        })

    summary_path = os.path.join(args.out_dir, "label_vs_reid_disagreements.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "base", "video_name", "center_frame", "player_label",
                "heuristic_pid", "court_rule_pid", "n_reid_at_center",
            ],
        )
        w.writeheader()
        w.writerows(out_rows)

    print(f"shots evaluated: {n_total}  agree: {n_agree}  "
          f"disagree: {len(out_rows)}  no-reid: {n_no_reid}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
