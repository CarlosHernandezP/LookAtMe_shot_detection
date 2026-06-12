"""
Render N comparison MP4s for shots where the heuristic identify_player() pick
disagrees with the court-rule applied on players_reid.csv coords.

For each picked shot:
- backdrop = MP4 from <old_dir>/<base>.mp4 (first 30 frames are the shot window)
- RED bbox/dot  -> the player_id the HEURISTIC selected
- GREEN bbox/dot -> the player_id the COURT-RULE selected
HUD shows base, label, heuristic_pid, court_rule_pid.

Picks top-N disagreements ranked by image-space separation between the two
players' bbox centers at the center frame (largest first).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2

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


def bbox_center(b: List[float]) -> Tuple[float, float]:
    return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))


def find_rec_by_pid(recs: List[dict], pid: int) -> Optional[dict]:
    for r in recs:
        if r["player_id"] == pid:
            return r
    return None


def render_one(
    mp4_path: str,
    base_frame: int,
    reid_by_frame: Dict[int, List[dict]],
    heur_pid: int,
    rule_pid: int,
    header: List[str],
    out_path: str,
    n_frames: int = 30,
) -> bool:
    if not os.path.exists(mp4_path):
        print(f"  missing mp4: {mp4_path}")
        return False
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fr_num = base_frame + idx
        recs = reid_by_frame.get(fr_num, []) if idx < n_frames else []
        rh = find_rec_by_pid(recs, heur_pid)
        rr = find_rec_by_pid(recs, rule_pid)
        if rh:
            x1, y1, x2, y2 = (int(v) for v in rh["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"OLD pid={heur_pid}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if rr:
            x1, y1, x2, y2 = (int(v) for v in rr["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"NEW pid={rule_pid}", (x1, y2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.rectangle(frame, (0, 0), (w, 28 * len(header) + 12), (0, 0, 0), -1)
        for li, line in enumerate(header):
            cv2.putText(frame, line, (10, 22 + li * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        out.write(frame)
        idx += 1
    cap.release()
    out.release()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disagreements-csv", required=True)
    ap.add_argument("--old-clips-dir", required=True, help="extract_all_with_clips_v2/")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    with open(args.disagreements_csv, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    print(f"total disagreements: {len(rows)}")

    # Score each by image-space separation at center frame
    reid_cache: Dict[str, Dict[int, List[dict]]] = {}
    scored = []
    for r in rows:
        video_name = r["video_name"]
        reid_path = resolve_reid_path(video_name)
        if not reid_path:
            continue
        if reid_path not in reid_cache:
            print(f"loading reid {reid_path} ...")
            reid_cache[reid_path] = load_players_reid_by_frame(reid_path)
        recs = reid_cache[reid_path].get(int(r["center_frame"]), [])
        rh = find_rec_by_pid(recs, int(r["heuristic_pid"]))
        rr = find_rec_by_pid(recs, int(r["court_rule_pid"]))
        if not (rh and rr):
            continue
        ch = bbox_center(rh["bbox"])
        cr = bbox_center(rr["bbox"])
        sep = math.hypot(ch[0] - cr[0], ch[1] - cr[1])
        r["_sep_px"] = sep
        r["_reid_path"] = reid_path
        scored.append(r)

    scored.sort(key=lambda r: r["_sep_px"], reverse=True)
    picks = scored[: args.top_n]
    print(f"rendering {len(picks)} compare clips ...")

    out_summary = os.path.join(args.out_dir, "picks_summary.csv")
    with open(out_summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "base", "player_label", "heuristic_pid", "court_rule_pid",
            "image_separation_px", "center_frame"
        ])
        for r in picks:
            w.writerow([
                r["base"], r["player_label"], r["heuristic_pid"],
                r["court_rule_pid"], f"{r['_sep_px']:.1f}", r["center_frame"]
            ])

    # base contains the original video name; the clip filename is <base>.mp4
    for r in picks:
        base = r["base"]
        mp4 = os.path.join(args.old_clips_dir, f"{base}.mp4")
        center_frame = int(r["center_frame"])
        base_frame = center_frame - 15  # extract_shots uses this offset
        header = [
            f"{base}",
            f"label={r['player_label']}  OLD pid={r['heuristic_pid']}  "
            f"NEW pid={r['court_rule_pid']}  sep={r['_sep_px']:.1f}px",
        ]
        out_path = os.path.join(args.out_dir, f"{base}_disagree.mp4")
        ok = render_one(
            mp4, base_frame, reid_cache[r["_reid_path"]],
            int(r["heuristic_pid"]), int(r["court_rule_pid"]),
            header, out_path,
        )
        print(("OK   " if ok else "FAIL ") + out_path)


if __name__ == "__main__":
    main()
