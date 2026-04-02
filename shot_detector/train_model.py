"""
Legacy compatibility: pose CSV loading lives in ``pose_io``.
Training: use ``train_shot_model.py``.
"""

from shot_detector.pose_io import (
    BALL_FEATURE_COLS,
    EXPECTED_FEATURE_COLS,
    N_FEATURES_RAW,
    SEQUENCE_LENGTH,
    load_pose_csv,
    pad_or_trim_sequence,
)

__all__ = [
    "BALL_FEATURE_COLS",
    "EXPECTED_FEATURE_COLS",
    "N_FEATURES_RAW",
    "SEQUENCE_LENGTH",
    "load_pose_csv",
    "pad_or_trim_sequence",
]
