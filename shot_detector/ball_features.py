"""
Ball feature extraction for shot detection.
Extracts ball position relative to player body and computes velocity/acceleration.
Uses NaN for missing ball data (XGBoost can handle NaN).
"""
import numpy as np
from typing import Optional, Tuple


def normalize_ball_features(ball_pos, ball_prev_pos, hip_center, shoulder_width, 
                           image_width, image_height, ball_visible=True, ball_confidence=1.0,
                           ball_prev_prev_pos=None):
    """
    Normalize ball features following the same pattern as pose features.
    Returns NaN for missing ball data (XGBoost can handle NaN).
    
    Parameters
    ----------
    ball_pos : np.ndarray or None
        Ball position [x, y] in pixel coordinates, or None if missing
    ball_prev_pos : np.ndarray or None
        Previous frame ball position for velocity calculation
    hip_center : np.ndarray
        Hip center [x, y] from pose
    shoulder_width : float
        Shoulder width for normalization (same as used for keypoints)
    image_width : int
        Image width in pixels
    image_height : int
        Image height in pixels
    ball_visible : bool
        Whether ball was detected (True) or interpolated (False)
    ball_confidence : float
        Detection confidence (0.0-1.0)
    ball_prev_prev_pos : np.ndarray or None
        Previous previous frame ball position for acceleration calculation
        
    Returns
    -------
    np.ndarray
        Array of 10 ball features (with NaN for missing data):
        [ball_x_body_rel, ball_y_body_rel, ball_x_abs, ball_y_abs,
         ball_vx_body_rel, ball_vy_body_rel, ball_ax_body_rel, ball_ay_body_rel,
         ball_visible, ball_confidence]
    """
    if ball_pos is None or np.any(np.isnan(ball_pos)):
        # Missing ball - return NaN (XGBoost can handle this)
        return np.array([
            np.nan, np.nan,  # ball_x_body_rel, ball_y_body_rel
            np.nan, np.nan,  # ball_x_abs, ball_y_abs
            np.nan, np.nan,  # ball_vx_body_rel, ball_vy_body_rel
            np.nan, np.nan,  # ball_ax_body_rel, ball_ay_body_rel
            0.0,             # ball_visible (0 = not visible)
            ball_confidence  # ball_confidence (might be 0.0)
        ])
    
    # 1. Body-relative position (same normalization as keypoints)
    ball_body_rel = (ball_pos - hip_center) / shoulder_width
    
    # 2. Absolute position (normalized by image size, like hip_x_abs)
    ball_x_abs = ball_pos[0] / image_width
    ball_y_abs = ball_pos[1] / image_height
    
    # 3. Body-relative velocity (in pixels per frame, normalized by shoulder_width)
    if ball_prev_pos is not None and not np.any(np.isnan(ball_prev_pos)):
        ball_velocity_pixels = ball_pos - ball_prev_pos
        ball_velocity_body_rel = ball_velocity_pixels / shoulder_width
    else:
        ball_velocity_body_rel = np.array([np.nan, np.nan])
    
    # 4. Body-relative acceleration (change in velocity, normalized by shoulder_width)
    if (ball_prev_pos is not None and not np.any(np.isnan(ball_prev_pos)) and
        ball_prev_prev_pos is not None and not np.any(np.isnan(ball_prev_prev_pos))):
        # Acceleration = change in velocity
        prev_velocity_pixels = ball_prev_pos - ball_prev_prev_pos
        prev_velocity_body_rel = prev_velocity_pixels / shoulder_width
        ball_acceleration_body_rel = ball_velocity_body_rel - prev_velocity_body_rel
    else:
        ball_acceleration_body_rel = np.array([np.nan, np.nan])
    
    return np.array([
        ball_body_rel[0],      # ball_x_body_rel
        ball_body_rel[1],      # ball_y_body_rel
        ball_x_abs,            # ball_x_abs
        ball_y_abs,            # ball_y_abs
        ball_velocity_body_rel[0],  # ball_vx_body_rel (speed in x)
        ball_velocity_body_rel[1],  # ball_vy_body_rel (speed in y)
        ball_acceleration_body_rel[0],  # ball_ax_body_rel (acceleration in x)
        ball_acceleration_body_rel[1],  # ball_ay_body_rel (acceleration in y)
        1.0 if ball_visible else 0.0,  # ball_visible
        ball_confidence        # ball_confidence
    ])


def get_ball_feature_names():
    """
    Get names of ball features in order.
    
    Returns
    -------
    list
        List of 10 feature names
    """
    return [
        'ball_x_body_rel',
        'ball_y_body_rel',
        'ball_x_abs',
        'ball_y_abs',
        'ball_vx_body_rel',
        'ball_vy_body_rel',
        'ball_ax_body_rel',
        'ball_ay_body_rel',
        'ball_visible',
        'ball_confidence'
    ]
