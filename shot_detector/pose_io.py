"""Load pose + ball rows from extracted ``*_pose.csv`` files (training pipeline)."""

from typing import List, Optional

import numpy as np
import pandas as pd

from shot_detector.ball_features import get_ball_feature_names

BODY_KEYPOINT_NAMES = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

EXPECTED_FEATURE_COLS: List[str] = []
for kp_name in BODY_KEYPOINT_NAMES:
    EXPECTED_FEATURE_COLS.extend([f"{kp_name}_x_body_rel", f"{kp_name}_y_body_rel"])
EXPECTED_FEATURE_COLS.extend(["hip_y_abs", "hip_x_abs", "shoulder_center_y_abs"])

BALL_FEATURE_COLS = get_ball_feature_names()
EXPECTED_FEATURE_COLS.extend(BALL_FEATURE_COLS)

SEQUENCE_LENGTH = 30
N_FEATURES_RAW = 27 + len(BALL_FEATURE_COLS)


def load_pose_csv(pose_csv_path: str) -> Optional[np.ndarray]:
    """Return array (T, N_FEATURES_RAW) or None if invalid."""
    try:
        df = pd.read_csv(pose_csv_path)
        if "frame_num" not in df.columns:
            return None
        missing_cols = [col for col in EXPECTED_FEATURE_COLS if col not in df.columns]
        if missing_cols:
            return None

        available_cols = [col for col in EXPECTED_FEATURE_COLS if col in df.columns]
        missing_cols = [col for col in EXPECTED_FEATURE_COLS if col not in df.columns]
        if missing_cols:
            features_df = df[available_cols].copy()
            for col in missing_cols:
                features_df[col] = np.nan
            features_df = features_df[EXPECTED_FEATURE_COLS]
        else:
            features_df = df[EXPECTED_FEATURE_COLS].copy()

        features = features_df.values
        pose_features = features[:, :27]
        ball_features = (
            features[:, 27:]
            if features.shape[1] > 27
            else np.full((features.shape[0], len(BALL_FEATURE_COLS)), np.nan)
        )

        pose_df = pd.DataFrame(pose_features)
        pose_df = pose_df.ffill().bfill().fillna(0)
        pose_features = pose_df.values

        if features.shape[1] > 27:
            features = np.hstack([pose_features, ball_features])
        else:
            features = np.hstack(
                [pose_features, np.full((features.shape[0], len(BALL_FEATURE_COLS)), np.nan)]
            )

        nan_ratio = np.isnan(features).sum() / features.size
        if nan_ratio > 0.5:
            return None
        if len(features) < 10:
            return None
        return features
    except Exception as e:
        print(f"Warning: Could not load {pose_csv_path}: {e}")
        return None


def pad_or_trim_sequence(features: np.ndarray) -> np.ndarray:
    if len(features) < SEQUENCE_LENGTH:
        padding = np.tile(features[-1:], (SEQUENCE_LENGTH - len(features), 1))
        features = np.vstack([features, padding])
    elif len(features) > SEQUENCE_LENGTH:
        features = features[:SEQUENCE_LENGTH]
    return features
