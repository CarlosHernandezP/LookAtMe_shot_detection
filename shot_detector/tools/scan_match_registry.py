"""
Scan disk for shot-annotation matches and report pipeline readiness.

Checks, per match-id found in /home/ec2-user/data/shot_annotations:
- annotation CSVs (+ column schema, flags extra columns like 'Good')
- video in /home/ec2-user/data/matches
- upstream intermediate artifacts (players_reid.csv, poses_raw.csv,
  ball_trajectories.csv) in LookAtMeProtoApp or stats-poc
- extracted per-shot pose CSVs in the shot_detector dataset dirs
- registration in extract_shots.py maps (BALL_TRAJECTORY_MAP, PLAYERS_REID_MAP)

Usage:
    uv run python -m shot_detector.tools.scan_match_registry            # report
    uv run python -m shot_detector.tools.scan_match_registry --update  # also refresh auto fields in match_registry.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_DIR = Path("/home/ec2-user/data/shot_annotations")
MATCHES_DIR = Path("/home/ec2-user/data/matches")
INTERMEDIATE_ROOTS = [
    Path("/home/ec2-user/carlos/LookAtMeProtoApp/data/intermediate"),
    Path("/home/ec2-user/carlos/stats-poc/data/intermediate"),
]
DATASET_DIRS = [
    REPO_ROOT / "shot_detector" / "data_csv_only",
    REPO_ROOT / "shot_detector" / "data" / "extract_all_with_clips_v2",
    REPO_ROOT / "shot_detector" / "data" / "extract_all_with_clips_v2_reid",
    REPO_ROOT / "shot_detector" / "data" / "extract_intermediate_poses",
]
# Canonical training dataset (sequential-read intermediate poses, VFR-safe)
CANONICAL_DATASET = "extract_intermediate_poses"
REGISTRY_PATH = REPO_ROOT / "shot_detector" / "pipeline_state" / "match_registry.json"
EXTRACT_SHOTS = REPO_ROOT / "shot_detector" / "extract_shots.py"

CAMERA_RE = re.compile(r"((?:BO|LU|JTS|MAV)-\d{4})", re.IGNORECASE)


def match_id_from_annotation(filename: str) -> str:
    base = filename[len("annotation_"):] if filename.startswith("annotation_") else filename
    base = os.path.splitext(base)[0]
    # strip _period<N> or trailing _<N>
    base = re.sub(r"_period\d+$", "", base)
    base = re.sub(r"_\d+$", "", base)
    # strip _<epoch>_rpi-<CAM> tail when present
    base = re.sub(r"_\d{10,13}_rpi-[A-Za-z]+-\d+$", "", base)
    return base


def scan() -> dict:
    out: dict = {}
    for f in sorted(ANNOTATIONS_DIR.glob("annotation_*.csv")):
        mid = match_id_from_annotation(f.name)
        rec = out.setdefault(mid, {
            "annotation_csvs": [], "cameras": set(), "extra_columns": set(),
            "videos": [], "intermediate_root": None, "intermediate_cams": [],
            "extracted_datasets": [], "registered_ball_map": False,
            "registered_reid_map": False,
        })
        rec["annotation_csvs"].append(f.name)
        m = CAMERA_RE.search(f.name)
        if m:
            rec["cameras"].add(m.group(1).upper())
        with open(f, newline="") as fh:
            header = next(csv.reader(fh), [])
        extras = [c for c in header if c not in ("Shot", "FrameId", "Player")]
        rec["extra_columns"].update(extras)

    extract_src = EXTRACT_SHOTS.read_text() if EXTRACT_SHOTS.exists() else ""

    for mid, rec in out.items():
        rec["videos"] = [p.name for p in MATCHES_DIR.glob(f"*{mid}*")
                         if p.suffix.lower() in (".mp4", ".mov", ".avi")]
        for root in INTERMEDIATE_ROOTS:
            d = root / mid
            if d.is_dir():
                rec["intermediate_root"] = str(root)
                rec["intermediate_cams"] = sorted(
                    c.name for c in d.iterdir()
                    if c.is_dir() and (c / "players_reid.csv").exists()
                )
                break
        for ds in DATASET_DIRS:
            if ds.is_dir() and glob.glob(str(ds / f"*{mid}*_pose.csv")):
                rec["extracted_datasets"].append(ds.name)
        # crude but effective: substring presence in the two maps' source
        if mid in extract_src:
            ball_sec = extract_src.split("BALL_TRAJECTORY_MAP", 1)[-1].split("PLAYERS_REID_MAP")[0]
            reid_sec = extract_src.split("PLAYERS_REID_MAP", 1)[-1]
            rec["registered_ball_map"] = mid in ball_sec
            rec["registered_reid_map"] = mid in reid_sec
        rec["cameras"] = sorted(rec["cameras"])
        rec["extra_columns"] = sorted(rec["extra_columns"])
    return out


def print_report(scanned: dict, registry: dict) -> None:
    deployed = {m for m, r in registry.get("matches", {}).items() if r.get("in_deployed_model")}
    cols = ["match_id", "ann", "video", "intermediate", "canonical_ds", "ball_map", "reid_map", "deployed"]
    print(f"{cols[0]:<42} {cols[1]:>3} {cols[2]:>5} {cols[3]:>12} {cols[4]:>12} {cols[5]:>8} {cols[6]:>8} {cols[7]:>8}")
    for mid, r in sorted(scanned.items()):
        print(
            f"{mid:<42} {len(r['annotation_csvs']):>3} "
            f"{'yes' if r['videos'] else 'NO':>5} "
            f"{(Path(r['intermediate_root']).parts[-3] if r['intermediate_root'] else 'NO'):>12} "
            f"{'yes' if CANONICAL_DATASET in r['extracted_datasets'] else 'NO':>12} "
            f"{'yes' if r['registered_ball_map'] else 'NO':>8} "
            f"{'yes' if r['registered_reid_map'] else 'NO':>8} "
            f"{'yes' if mid in deployed else 'no':>8}"
        )
        if r["extra_columns"]:
            print(f"  ^ extra annotation columns: {r['extra_columns']}")


def update_registry(scanned: dict, registry: dict) -> dict:
    matches = registry.setdefault("matches", {})
    for mid, r in scanned.items():
        entry = matches.setdefault(mid, {"in_deployed_model": False})
        entry["annotation_csvs"] = r["annotation_csvs"]
        entry["cameras"] = r["cameras"] or entry.get("cameras", [])
        entry["intermediate_root"] = r["intermediate_root"]
        entry["videos_present"] = bool(r["videos"])
        entry["extracted_datasets"] = r["extracted_datasets"]
        if r["extra_columns"]:
            entry["extra_annotation_columns"] = r["extra_columns"]
    return registry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="refresh auto fields in match_registry.json")
    args = ap.parse_args()

    scanned = scan()
    registry = json.loads(REGISTRY_PATH.read_text()) if REGISTRY_PATH.exists() else {"matches": {}}
    print_report(scanned, registry)
    if args.update:
        REGISTRY_PATH.write_text(json.dumps(update_registry(scanned, registry), indent=2) + "\n")
        print(f"\nupdated {REGISTRY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
