"""
Aggregates raw shot labels from filenames into model classes.
"""

from typing import Dict, List, Optional, Set
import os
import glob
import pandas as pd

# Raw shot labels (from filenames) -> model class. ``wall_shot`` is only ``forehand_contrapared``;
# wall exits / wall lobs map to forehand or backhand.
DEFAULT_SHOT_MAPPING: Dict[str, List[str]] = {
    "forehand": [
        "forehand",
        "lob",
        "forehand_wall_exit",
        "wall_lob",
    ],
    "backhand": [
        "backhand",
        "backhand_wall_exit",
    ],
    "serve": ["serve"],
    "smash": [
        "flat_smash",
        "topspin_smash",
        "smash",
        "bajada",
    ],
    "volley": [
        "backhand_volley",
        "forehand_volley",
        "drop_shot",
    ],
    "bandeja": ["bandeja"],
    "vibora": ["vibora"],
    "idle": ["idle"],
    "wall_shot": ["forehand_contrapared"],
}


def map_shot_to_class(shot_type: str, mapping: Optional[Dict[str, List[str]]] = None) -> Optional[str]:
    if mapping is None:
        mapping = DEFAULT_SHOT_MAPPING

    if shot_type is None or str(shot_type).strip().lower() == "shot":
        return None

    shot_type = str(shot_type).strip().lower()

    for class_name, shot_list in mapping.items():
        if shot_type in [s.lower() for s in shot_list]:
            return class_name

    return None


def get_all_shot_types(csv_dir: str) -> Set[str]:
    shot_types = set()
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if "Shot" in df.columns:
                for shot in df["Shot"].unique():
                    if pd.notna(shot) and str(shot).strip().lower() != "shot":
                        shot_types.add(str(shot).strip())
        except Exception as e:
            print(f"Warning: Could not read {csv_file}: {e}")
            continue

    return shot_types


def get_all_shot_types_from_pose_files(data_dir: str) -> Set[str]:
    shot_types = set()
    pose_files = glob.glob(os.path.join(data_dir, "*_pose.csv"))

    for pose_file in pose_files:
        basename = os.path.basename(pose_file)
        parts = basename.replace("_pose.csv", "").split("_")
        player_positions = {"left", "right", "top", "bottom"}
        shot_type_parts = []

        for i, part in enumerate(parts):
            if part.lower() in player_positions:
                if i > 0:
                    shot_type_parts = parts[2:i]
                break

        if shot_type_parts:
            shot_type = "_".join(shot_type_parts)
            if shot_type and shot_type.lower() != "shot":
                shot_types.add(shot_type)

    return shot_types


def create_mapping_from_config(config_dict: Dict) -> Dict[str, List[str]]:
    if "shot_mapping" in config_dict:
        return config_dict["shot_mapping"]
    return config_dict


def validate_mapping(mapping: Dict[str, List[str]], discovered_shots: Set[str]) -> Dict[str, List[str]]:
    all_mapped_shots = set()
    for shot_list in mapping.values():
        all_mapped_shots.update([s.lower() for s in shot_list])

    discovered_lower = {s.lower() for s in discovered_shots}
    unmapped = discovered_lower - all_mapped_shots

    result = {
        "mapped": len(discovered_shots) - len(unmapped),
        "unmapped": list(unmapped),
        "warnings": [],
    }

    if unmapped:
        result["warnings"].append(
            f"Found {len(unmapped)} unmapped shot types: {', '.join(unmapped)}"
        )

    return result


def extract_shot_type_from_filename(filename: str) -> Optional[str]:
    basename = os.path.basename(filename)
    if not basename.endswith("_pose.csv"):
        return None

    name = basename.replace("_pose.csv", "")
    parts = name.split("_")

    if "idle" in [p.lower() for p in parts]:
        idle_idx = -1
        for i, part in enumerate(parts):
            if part.lower() == "idle":
                idle_idx = i
                break

        if idle_idx != -1:
            player_positions = {"left", "right", "top", "bottom"}
            if idle_idx + 1 < len(parts) and parts[idle_idx + 1].lower() in player_positions:
                return "idle"

    player_positions = {"left", "right", "top", "bottom"}
    player_idx = -1

    for i, part in enumerate(parts):
        if part.lower() in player_positions:
            player_idx = i
            break

    if player_idx == -1 or player_idx < 1:
        return None

    frame_idx = -1
    for i in range(player_idx - 1, -1, -1):
        try:
            int(parts[i])
            frame_idx = i
            break
        except ValueError:
            continue

    if frame_idx == -1:
        shot_type = parts[player_idx - 1]
    else:
        if player_idx - frame_idx > 1:
            shot_type_parts = parts[frame_idx + 1 : player_idx]
            shot_type = "_".join(shot_type_parts)
        else:
            shot_type = parts[player_idx - 1]

    if shot_type and shot_type.lower() != "shot":
        return shot_type

    return None
