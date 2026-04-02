import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import re
import argparse
import json
from mmpose.apis import MMPoseInferencer
from shot_detector.utils import parse_shot_csv, get_video_path, identify_player, get_idle_player
from shot_detector.utils import load_fisheye_params, load_perspective_matrix, transform_points, get_foot_position
from shot_detector.ball_features import normalize_ball_features, get_ball_feature_names

# Configuration
SHOTS_CSV_DIRS = [
    'shot_csvs/shots_csvs/',
    '/home/daniele/shots_csvs/'
]
VIDEOS_DIRS = [
    'videos/',
    '/home/daniele/videos/'
]
OUTPUT_DIR = 'shot_detector/data/'
MODEL_CONFIG = 'configs/rtmo-s_8xb32-600e_coco-640x640.py'
MODEL_CHECKPOINT = 'model_weights/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ball trajectory configuration
BALL_TRAJECTORY_DIR = '/home/carlos/pose_estimators/'
FRAME_OFFSET = 0
MIN_TRAJECTORY_LENGTH = 5

# Map: video name substring -> (trajectory CSV filename, camera_id or None)
# camera_id=None means the trajectory file already contains only one camera
BALL_TRAJECTORY_MAP = {
    '0529b769-125d-4a22-bcee-b1707b87447e': {
        'BO-0001': '0529b769-125d-4a22-bcee-b1707b87447e_BO-0001_ball_trajectories.csv',
        'BO-0002': '0529b769-125d-4a22-bcee-b1707b87447e_BO-0002_ball_trajectories.csv',
    },
    '15-11-2025-15-57_rpi-BO-0001': {
        'BO-0001': '15-11-2025_ball_trajectories.csv',
    },
    '22-11-2025-18-10_rpi-LU-0002': {
        'LU-0002': '22-11-2025_ball_trajectories.csv',
    },
}

# Which annotation CSVs to process (None = all available)
VIDEO_FILTER = None

# Calibration constants
PARAM_DIR = 'parameters'
FISHEYE_FILE = 'fishcam-fisheye.txt'

# Debug mode: Set to True to see all detected poses with indices
DEBUG_MODE = True

# COCO Keypoint mapping for normalization
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

COCO_TO_FEATURE_IDX = {
    5: 0,   # left_shoulder
    6: 1,   # right_shoulder
    7: 2,   # left_elbow
    8: 3,   # right_elbow
    9: 4,   # left_wrist
    10: 5,  # right_wrist
    11: 6,  # left_hip
    12: 7,  # right_hip
    13: 8,  # left_knee
    14: 9,  # right_knee
    15: 10, # left_ankle
    16: 11  # right_ankle
}

BODY_KEYPOINT_NAMES = [
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle'
]

# Tracking thresholds
DISTANCE_THRESHOLD = 150 * 150  # 150px squared (pixel space)
COURT_DISTANCE_THRESHOLD = 2.0 * 2.0  # 2 meters in court space

def init_mmpose():
    if not os.path.exists(MODEL_CONFIG):
        raise FileNotFoundError(f"Config not found: {MODEL_CONFIG}")
    if not os.path.exists(MODEL_CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_CHECKPOINT}")
    
    print(f"Initializing MMPose with {MODEL_CONFIG} on {DEVICE}...")
    print(f"Detection score threshold: 0.05 (lowered from 0.1 in config)")
    return MMPoseInferencer(
        pose2d=MODEL_CONFIG,
        pose2d_weights=MODEL_CHECKPOINT,
        device=DEVICE
    )

def match_player_by_position(poses, prev_bbox, prev_court_pos, K, D, H, exclude_idx=-1):
    """Match a player by pixel or court position. Returns (match_idx, new_bbox, new_court_pos)."""
    if not poses:
        return -1, None, None
    
    match_idx = -1
    best_pixel_dist = float('inf')
    best_court_dist = float('inf')
    
    # Try pixel-based matching first
    if prev_bbox is not None:
        cx_prev = (prev_bbox[0] + prev_bbox[2]) / 2
        cy_prev = (prev_bbox[1] + prev_bbox[3]) / 2
        
        for p_idx, p in enumerate(poses):
            if p_idx == exclude_idx: continue
            bbox = unwrap_bbox(p['bbox'])
            if len(bbox) < 4: continue
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            dist = (cx - cx_prev)**2 + (cy - cy_prev)**2
            if dist < best_pixel_dist:
                best_pixel_dist = dist
                match_idx = p_idx
    
    # Use pixel match if close enough
    if match_idx != -1 and best_pixel_dist < DISTANCE_THRESHOLD:
        new_bbox = unwrap_bbox(poses[match_idx]['bbox'])
        foot_pos = get_foot_position(new_bbox)
        new_court_pos = None
        if H is not None:
            transformed = transform_points([foot_pos], K, D, H)
            if len(transformed) > 0:
                new_court_pos = transformed[0]
        return match_idx, new_bbox, new_court_pos
    
    # Try court-based recovery
    if prev_court_pos is not None and H is not None:
        match_idx = -1
        for p_idx, p in enumerate(poses):
            if p_idx == exclude_idx: continue
            bbox = unwrap_bbox(p['bbox'])
            if len(bbox) < 4: continue
            foot_pos = get_foot_position(bbox)
            transformed = transform_points([foot_pos], K, D, H)
            if len(transformed) > 0:
                court_pos = transformed[0]
                court_dist = ((court_pos[0] - prev_court_pos[0])**2 + 
                             (court_pos[1] - prev_court_pos[1])**2)
                if court_dist < best_court_dist:
                    best_court_dist = court_dist
                    match_idx = p_idx
        
        if match_idx != -1 and best_court_dist < COURT_DISTANCE_THRESHOLD:
            new_bbox = unwrap_bbox(poses[match_idx]['bbox'])
            foot_pos = get_foot_position(new_bbox)
            new_court_pos = None
            if H is not None:
                transformed = transform_points([foot_pos], K, D, H)
                if len(transformed) > 0:
                    new_court_pos = transformed[0]
            return match_idx, new_bbox, new_court_pos
    
    # No match found
    return -1, None, prev_court_pos  # Keep court position for next frame

def get_calibration(video_name):
    """
    Infers camera parameters based on video name.
    """
    # Common fisheye params
    fisheye_path = os.path.join(PARAM_DIR, FISHEYE_FILE)
    K, D = load_fisheye_params(fisheye_path)
    
    # Perspective matrix based on camera name
    camera_name = None
    
    # Check standard names BO01, BO02...
    for cam in ['BO01', 'BO02', 'LU01', 'LU02']:
        if cam in video_name: 
            camera_name = cam
            break
            
    # Try with hyphenated version (e.g. BO-01, BO-0001)
    if not camera_name:
        for prefix in ['BO', 'LU']:
            # Search for pattern PREFIX-NUMBER
            match = re.search(f"{prefix}-(\d+)", video_name)
            if match:
                num = int(match.group(1))
                # Normalize to 2 digits: BO01
                camera_name = f"{prefix}{num:02d}"
                break

    H = None
    if camera_name:
        perspective_path = os.path.join(PARAM_DIR, f"{camera_name}-perspective.txt")
        H = load_perspective_matrix(perspective_path)
    else:
        # Fallback or specific logic for other names?
        pass

    return K, D, H

def is_pose_near_bottom(bbox, img_height, bottom_margin=0.05):
    """
    Checks if a pose is near the bottom of the image.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        img_height: Image height
        bottom_margin: Margin from bottom as fraction of image height (default 5%)
    
    Returns:
        True if pose is near bottom, False otherwise
    """
    if len(bbox) < 4:
        return False
    
    foot_pos = get_foot_position(bbox)
    foot_y = foot_pos[1]
    
    # Check if near bottom (within bottom_margin of bottom edge)
    return foot_y > img_height * (1 - bottom_margin)

def is_pose_near_corners_or_bottom(bbox, img_width, img_height, corner_margin=0.05, bottom_margin=0.05):
    """
    Checks if a pose is near the corners or bottom of the image.
    These are often false positives from reflections or static objects.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        img_width: Image width
        img_height: Image height
        corner_margin: Margin as fraction of image size (default 5%)
        bottom_margin: Margin from bottom as fraction of image height (default 5%)
    
    Returns:
        True if pose is near corners or bottom, False otherwise
    """
    if len(bbox) < 4:
        return False
    
    x1, y1, x2, y2 = bbox[:4]
    foot_pos = get_foot_position(bbox)
    foot_x, foot_y = foot_pos
    
    # Check if near bottom (within bottom_margin of bottom edge)
    if foot_y > img_height * (1 - bottom_margin):
        return True
    
    # Check if near corners
    corner_threshold_x = img_width * corner_margin
    corner_threshold_y = img_height * corner_margin
    
    # Bottom-left corner
    if foot_x < corner_threshold_x and foot_y > img_height * (1 - corner_margin):
        return True
    
    # Bottom-right corner
    if foot_x > img_width * (1 - corner_margin) and foot_y > img_height * (1 - corner_margin):
        return True
    
    # Top-left corner (less common but possible)
    if foot_x < corner_threshold_x and foot_y < corner_threshold_y:
        return True
    
    # Top-right corner
    if foot_x > img_width * (1 - corner_margin) and foot_y < corner_threshold_y:
        return True
    
    return False

def filter_stationary_poses(poses_per_frame, movement_threshold=20.0, min_frames=5, 
                           img_height=None, bottom_margin=0.05, filter_bottom_stationary_only=False):
    """
    Filters poses that barely move across frames (likely false positives).
    If filter_bottom_stationary_only is True, only filters poses that are BOTH near bottom AND stationary.
    
    Args:
        poses_per_frame: List of lists, where each inner list contains poses for a frame
        movement_threshold: Minimum pixel movement required (default 20 pixels)
        min_frames: Minimum number of frames the pose must appear in to be considered stationary (default 5)
        img_height: Image height (required if filter_bottom_stationary_only is True)
        bottom_margin: Margin from bottom as fraction of image height (default 5%)
        filter_bottom_stationary_only: If True, only filter poses that are near bottom AND stationary
    
    Returns:
        List of lists with stationary poses removed
    """
    if not poses_per_frame or len(poses_per_frame) < min_frames:
        return poses_per_frame
    
    # Build a list of all poses with their frame indices and positions
    pose_tracks = []  # List of (frame_idx, foot_pos, pose, is_near_bottom)
    
    for frame_idx, frame_poses in enumerate(poses_per_frame):
        for pose in frame_poses:
            bbox = unwrap_bbox(pose['bbox'])
            if len(bbox) < 4:
                continue
            foot_pos = get_foot_position(bbox)
            # Check if near bottom (if filtering bottom+stationary only)
            is_near_bottom = False
            if filter_bottom_stationary_only and img_height is not None:
                is_near_bottom = is_pose_near_bottom(bbox, img_height, bottom_margin)
            pose_tracks.append((frame_idx, foot_pos, pose, is_near_bottom))
    
    # Group poses into tracks by proximity
    # Simple approach: poses in nearby frames with similar positions are the same track
    tracks = []  # List of lists of (frame_idx, foot_pos, pose, is_near_bottom)
    match_distance = 100  # Pixels
    
    for frame_idx, foot_pos, pose, is_near_bottom in pose_tracks:
        matched = False
        for track in tracks:
            # Check if this pose matches any pose in the track
            for track_frame_idx, track_foot_pos, _, _ in track:
                dist = np.sqrt((foot_pos[0] - track_foot_pos[0])**2 + 
                              (foot_pos[1] - track_foot_pos[1])**2)
                # Match if close in position and within reasonable frame distance
                if dist < match_distance and abs(frame_idx - track_frame_idx) <= min_frames:
                    track.append((frame_idx, foot_pos, pose, is_near_bottom))
                    matched = True
                    break
            if matched:
                break
        
        if not matched:
            tracks.append([(frame_idx, foot_pos, pose, is_near_bottom)])
    
    # Filter out stationary tracks
    filtered_poses_per_frame = [[] for _ in range(len(poses_per_frame))]
    filtered_count = 0
    
    for track in tracks:
        if len(track) < min_frames:
            # Too few appearances, keep it (might be valid but brief)
            for frame_idx, _, pose, _ in track:
                filtered_poses_per_frame[frame_idx].append(pose)
            continue
        
        # Calculate maximum movement in this track
        positions = [foot_pos for _, foot_pos, _, _ in track]
        max_movement = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                movement = np.sqrt((positions[i][0] - positions[j][0])**2 +
                                  (positions[i][1] - positions[j][1])**2)
                max_movement = max(max_movement, movement)
        
        # Check if track is near bottom (if any pose in track is near bottom, consider it near bottom)
        track_is_near_bottom = any(is_near_bottom for _, _, _, is_near_bottom in track)
        
        # Decide whether to filter:
        # - If filter_bottom_stationary_only: only filter if BOTH stationary AND near bottom
        # - Otherwise: filter if stationary
        should_filter = False
        if filter_bottom_stationary_only:
            # Only filter if stationary AND near bottom
            if max_movement < movement_threshold and track_is_near_bottom:
                should_filter = True
        else:
            # Filter if stationary (regardless of position)
            if max_movement < movement_threshold:
                should_filter = True
        
        # Keep track if it moves enough or doesn't meet filter criteria
        if not should_filter:
            for frame_idx, _, pose, _ in track:
                filtered_poses_per_frame[frame_idx].append(pose)
        else:
            # Track is filtered out
            filtered_count += len(track)
    
    return filtered_poses_per_frame

def filter_flickering_poses(poses_per_frame, min_presence_ratio=0.8):
    """
    Filters poses that flicker (don't appear in at least min_presence_ratio of frames).
    
    Args:
        poses_per_frame: List of lists, where each inner list contains poses for a frame
        min_presence_ratio: Minimum ratio of frames the pose must appear in (default 0.8 = 80%)
    
    Returns:
        Tuple of (filtered_poses_per_frame, flickering_tracks)
        flickering_tracks: List of track info for poses that were filtered as flickering
    """
    if not poses_per_frame or len(poses_per_frame) == 0:
        return poses_per_frame, []
    
    total_frames = len(poses_per_frame)
    min_frames_required = int(total_frames * min_presence_ratio)
    
    # Build tracks (same as in filter_stationary_poses)
    pose_tracks = []  # List of (frame_idx, foot_pos, pose)
    
    for frame_idx, frame_poses in enumerate(poses_per_frame):
        for pose in frame_poses:
            bbox = unwrap_bbox(pose['bbox'])
            if len(bbox) < 4:
                continue
            foot_pos = get_foot_position(bbox)
            pose_tracks.append((frame_idx, foot_pos, pose))
    
    # Group poses into tracks by proximity
    tracks = []
    match_distance = 100  # Pixels
    
    for frame_idx, foot_pos, pose in pose_tracks:
        matched = False
        for track in tracks:
            for track_frame_idx, track_foot_pos, _ in track:
                dist = np.sqrt((foot_pos[0] - track_foot_pos[0])**2 + 
                              (foot_pos[1] - track_foot_pos[1])**2)
                if dist < match_distance:
                    track.append((frame_idx, foot_pos, pose))
                    matched = True
                    break
            if matched:
                break
        
        if not matched:
            tracks.append([(frame_idx, foot_pos, pose)])
    
    # Filter out flickering tracks
    filtered_poses_per_frame = [[] for _ in range(len(poses_per_frame))]
    flickering_tracks = []
    
    for track in tracks:
        presence_ratio = len(track) / total_frames
        
        if presence_ratio >= min_presence_ratio:
            # Keep track - appears in enough frames
            for frame_idx, _, pose in track:
                filtered_poses_per_frame[frame_idx].append(pose)
        else:
            # Flickering - mark for visualization but don't include in filtered output
            flickering_tracks.append({
                'frames': [f_idx for f_idx, _, _ in track],
                'presence_ratio': presence_ratio
            })
    
    return filtered_poses_per_frame, flickering_tracks

def is_pose_valid(pose, K, D, H, img_height=None, img_width=None):
    """
    Checks if a pose is within the imaginary field (0-10, 0-20).
    Also filters background players (y < 30% of image height).
    Filters poses near corners/bottom of image.
    """
    bbox = unwrap_bbox(pose['bbox'])
    if len(bbox) < 4:
        return False
    
    # 1. Pixel-based filtering (y > 30% of height)
    if img_height is not None:
        foot_y = get_foot_position(bbox)[1]
        if foot_y < 0.25 * img_height:
             return False # Too far up (background players)
        
        # 2. Filter poses near corners (bottom filtering is done in combination with stationary)
        if img_width is not None:
            # Only filter top corners, not bottom (bottom is handled with stationary filtering)
            corner_margin = 0.05
            corner_threshold_x = img_width * corner_margin
            corner_threshold_y = img_height * corner_margin
            foot_pos = get_foot_position(bbox)
            foot_x, foot_y = foot_pos
            
            # Check top corners only (not bottom corners)
            # Top-left corner
            if foot_x < corner_threshold_x and foot_y < corner_threshold_y:
                return False
            # Top-right corner
            if foot_x > img_width * (1 - corner_margin) and foot_y < corner_threshold_y:
                return False

    if H is None:
        return True # No calibration, assume valid or handle differently
        
    foot_pos = get_foot_position(bbox)
    
    # Transform
    # transform_points expects list of points or (N,2)
    # It returns (N,2)
    transformed = transform_points([foot_pos], K, D, H)
    if len(transformed) == 0:
        return False
        
    tx, ty = transformed[0]
    
    # Filter -0.3 to 10.3 x (0.3m margin), -1.3 to 21.3 y (1.3m margin)
    if -1.1 <= tx <= 10.7 and -0.7 <= ty <= 20.7:
        return True
        
    return False

def extract_clip_and_pose(video_path, start_frame, duration, inferencer, K=None, D=None, H=None, return_all_poses=False):
    """
    Extracts frames, runs pose estimation, and returns data.
    Filters poses based on calibration if provided.
    Also filters poses near corners/bottom and stationary poses.
    
    Args:
        return_all_poses: If True, returns all poses (filtered and unfiltered) for debugging
    
    Returns:
        If return_all_poses: (frames, poses_per_frame, all_poses_per_frame, filtering_info)
        filtering_info: Dict with 'filtered_per_frame' and 'flickering_per_frame' lists
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        if return_all_poses:
            return None, None, None, None
        return None, None
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    frames = []
    poses_per_frame = []
    all_poses_per_frame = [] if return_all_poses else None
    
    for _ in range(duration):
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        
        # Run inference on single frame
        result_generator = inferencer(frame, return_vis=False)
        result = next(result_generator)
        
        predictions = result['predictions'][0]
        
        frame_poses = []
        frame_all_poses = []
        
        if predictions:
            for instance in predictions:
                if return_all_poses:
                    # Store all poses with validity flag
                    is_valid = is_pose_valid(instance, K, D, H, img_height=img_height, img_width=img_width)
                    frame_all_poses.append((instance, is_valid))
                
                # Filter here (corner/bottom filtering included)
                if is_pose_valid(instance, K, D, H, img_height=img_height, img_width=img_width):
                    frame_poses.append(instance)
                
        poses_per_frame.append(frame_poses)
        if return_all_poses:
            all_poses_per_frame.append(frame_all_poses)

    cap.release()
    
    # Debug: count poses before filtering
    total_poses_before = sum(len(p) for p in poses_per_frame)
    
    # Save original poses before filtering for visualization
    original_poses_per_frame = [p.copy() for p in poses_per_frame]
    
    # Apply stationary pose filtering (removes poses that are near bottom AND barely move)
    poses_per_frame_after_stationary = filter_stationary_poses(poses_per_frame, movement_threshold=20.0, min_frames=5,
                                             img_height=img_height, bottom_margin=0.05, 
                                             filter_bottom_stationary_only=True)
    
    # Debug: count poses after stationary filtering
    total_poses_after_stationary = sum(len(p) for p in poses_per_frame_after_stationary)
    
    # Apply flickering pose filtering (removes poses that appear in < 80% of frames)
    poses_per_frame_after_flickering, flickering_tracks = filter_flickering_poses(poses_per_frame_after_stationary, min_presence_ratio=0.8)
    
    # Debug: count poses after flickering filtering
    total_poses_after_flickering = sum(len(p) for p in poses_per_frame_after_flickering)
    
    # Debug output to verify filtering is working
    if DEBUG_MODE and (total_poses_before > total_poses_after_flickering):
        print(f"  Filtering applied: {total_poses_before} -> {total_poses_after_stationary} (after stationary) -> {total_poses_after_flickering} (after flickering)")
        print(f"    Removed {total_poses_before - total_poses_after_stationary} poses (stationary+bottom)")
        print(f"    Removed {total_poses_after_stationary - total_poses_after_flickering} poses (flickering)")
    
    # Build filtering info for visualization: track which poses were filtered
    filtering_info = {
        'filtered_per_frame': [set() for _ in range(len(original_poses_per_frame))],
        'flickering_per_frame': [set() for _ in range(len(original_poses_per_frame))]
    }
    
    # Map filtered poses back to original indices by matching bboxes
    for frame_idx in range(len(original_poses_per_frame)):
        original_poses = original_poses_per_frame[frame_idx]
        after_stationary = poses_per_frame_after_stationary[frame_idx] if frame_idx < len(poses_per_frame_after_stationary) else []
        after_flickering = poses_per_frame_after_flickering[frame_idx] if frame_idx < len(poses_per_frame_after_flickering) else []
        
        # Find poses filtered by stationary
        for orig_idx, orig_pose in enumerate(original_poses):
            orig_bbox = unwrap_bbox(orig_pose['bbox'])
            if len(orig_bbox) < 4:
                continue
            orig_center = ((orig_bbox[0] + orig_bbox[2]) / 2, (orig_bbox[1] + orig_bbox[3]) / 2)
            
            found_in_stationary = False
            for filtered_pose in after_stationary:
                filtered_bbox = unwrap_bbox(filtered_pose['bbox'])
                if len(filtered_bbox) < 4:
                    continue
                filtered_center = ((filtered_bbox[0] + filtered_bbox[2]) / 2, 
                                  (filtered_bbox[1] + filtered_bbox[3]) / 2)
                dist = np.sqrt((orig_center[0] - filtered_center[0])**2 + 
                              (orig_center[1] - filtered_center[1])**2)
                if dist < 50:
                    found_in_stationary = True
                    break
            
            if not found_in_stationary:
                filtering_info['filtered_per_frame'][frame_idx].add(orig_idx)
            else:
                # Check if it's in flickering (was in stationary but not in flickering)
                found_in_flickering = False
                for flickering_pose in after_flickering:
                    flickering_bbox = unwrap_bbox(flickering_pose['bbox'])
                    if len(flickering_bbox) < 4:
                        continue
                    flickering_center = ((flickering_bbox[0] + flickering_bbox[2]) / 2, 
                                        (flickering_bbox[1] + flickering_bbox[3]) / 2)
                    dist = np.sqrt((orig_center[0] - flickering_center[0])**2 + 
                                  (orig_center[1] - flickering_center[1])**2)
                    if dist < 50:
                        found_in_flickering = True
                        break
                
                if not found_in_flickering:
                    filtering_info['flickering_per_frame'][frame_idx].add(orig_idx)
    
    # Use filtered poses for actual processing
    poses_per_frame = poses_per_frame_after_flickering
    
    if return_all_poses:
        return frames, poses_per_frame, all_poses_per_frame, filtering_info
    return frames, poses_per_frame

def unwrap_bbox(bbox):
    """
    Helper to unwrap nested bbox structure ([x1,y1,x2,y2],) -> [x1,y1,x2,y2]
    """
    if len(bbox) == 1 and isinstance(bbox[0], (list, tuple, np.ndarray)):
        return bbox[0]
    return bbox

def draw_overlay(frame, poses, active_idx, idle_idx, debug=False, all_poses_info=None, 
                is_forward_filled=False, forward_filled_pose=None, filtering_info=None,
                ball_positions=None, frame_idx_in_clip=None, start_frame=0):
    """
    Draws bounding boxes and labels on the frame.
    
    Args:
        debug: If True, shows all poses with indices and debug info
        all_poses_info: List of (pose, is_valid) tuples for all detected poses (including filtered ones)
        is_forward_filled: If True, active player is forward-filled (show in blue)
        forward_filled_pose: The pose to draw when forward-filling (if None, uses last pose in poses)
        filtering_info: Dict with 'filtered_per_frame' and 'flickering_per_frame' sets of pose indices
        ball_positions: Dict {frame_number: (x, y, confidence, is_interpolated)} for ball positions
        frame_idx_in_clip: Current frame index in clip (0-based)
        start_frame: Starting frame number in video
    """
    img = frame.copy()
    
    # Only draw poses that are being tracked
    tracked_indices = set()
    if active_idx != -1:
        tracked_indices.add(active_idx)
    if idle_idx != -1:
        tracked_indices.add(idle_idx)
    
    # If all_poses_info is provided, draw ALL poses (filtered and unfiltered)
    if all_poses_info is not None:
        fill_status = "FILLED" if is_forward_filled else ""
        debug_text = f"All detected: {len(all_poses_info)} | Valid: {len(poses)} | Active: {active_idx} | Idle: {idle_idx} {fill_status}"
        cv2.putText(img, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw all poses
        valid_count = 0
        for i, (pose, is_valid) in enumerate(all_poses_info):
            bbox = unwrap_bbox(pose['bbox'])
            if len(bbox) < 4: continue
            
            x1, y1, x2, y2 = map(int, bbox[:4])
            
            # Color coding:
            # - Green: Valid AND tracked as Active
            # - Red: Valid AND tracked as Idle  
            # - Yellow: Valid but not tracked
            # - Magenta: Invalid (filtered out)
            
            # Check filtering status
            is_filtered = False
            is_flickering = False
            if filtering_info is not None:
                # Note: i is the index in all_poses_info, which should match the original pose index
                # We need to get the frame index - but we don't have it here. 
                # For now, we'll check if this pose matches any filtered pose by bbox
                # This is approximate but should work
                this_bbox = unwrap_bbox(pose['bbox'])
                if len(this_bbox) >= 4:
                    # Try to match with filtered/flickering poses by checking all frames
                    # This is a simplified approach - ideally we'd pass frame_idx
                    pass  # Will handle below with bbox matching
            
            if is_valid:
                valid_count += 1
                # Check if this pose is in the filtered list and tracked
                # Compare by bbox to find matching pose
                pose_in_filtered = False
                tracked_as = None
                this_bbox = unwrap_bbox(pose['bbox'])
                for j, filtered_pose in enumerate(poses):
                    filtered_bbox = unwrap_bbox(filtered_pose['bbox'])
                    # Compare bboxes (allow small tolerance for floating point)
                    if len(this_bbox) >= 4 and len(filtered_bbox) >= 4:
                        if (abs(this_bbox[0] - filtered_bbox[0]) < 1 and
                            abs(this_bbox[1] - filtered_bbox[1]) < 1 and
                            abs(this_bbox[2] - filtered_bbox[2]) < 1 and
                            abs(this_bbox[3] - filtered_bbox[3]) < 1):
                            pose_in_filtered = True
                            if j == active_idx:
                                tracked_as = 'active'
                            elif j == idle_idx:
                                tracked_as = 'idle'
                            break
                
                # Check if this pose was filtered (stationary/bottom) or flickering
                if filtering_info is not None:
                    # filtering_info is now per-frame: {'filtered': set(), 'flickering': set()}
                    if i in filtering_info.get('flickering', set()):
                        is_flickering = True
                    elif i in filtering_info.get('filtered', set()):
                        is_filtered = True
                
                # Determine color based on status
                if is_flickering:
                    color = (0, 0, 0)  # Black - flickering
                    label = f"P{i} FLICKERING"
                elif is_filtered:
                    color = (0, 165, 255)  # Orange - filtered (stationary/bottom)
                    label = f"P{i} FILTERED"
                elif tracked_as == 'active':
                    if is_forward_filled:
                        color = (255, 0, 0)  # Blue - forward-filled
                        label = f"P{i} ACTIVE (FILLED)"
                    else:
                        color = (0, 255, 0)  # Green
                        label = f"P{i} ACTIVE"
                elif tracked_as == 'idle':
                    color = (0, 0, 255)  # Red
                    label = f"P{i} IDLE"
                elif pose_in_filtered:
                    color = (0, 255, 255)  # Yellow - valid but not tracked
                    label = f"P{i} VALID"
                else:
                    color = (0, 255, 255)  # Yellow - should not happen
                    label = f"P{i} VALID?"
            else:
                color = (255, 0, 255)  # Magenta - invalid (initial validation failed)
                label = f"P{i} INVALID"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw keypoints
            if 'keypoints' in pose:
                kpts = pose['keypoints']
                for kp in kpts:
                    kx, ky = int(kp[0]), int(kp[1])
                    if kx > 0 and ky > 0:
                        cv2.circle(img, (kx, ky), 3, color, -1)
        
        cv2.putText(img, f"Valid: {valid_count}/{len(all_poses_info)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw forward-filled pose if needed (draw on top of everything)
        if is_forward_filled and forward_filled_pose is not None:
            bbox = unwrap_bbox(forward_filled_pose['bbox'])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                color = (255, 0, 0)  # Blue (BGR format)
                label = "ACTIVE (FILLED)"
                # Draw with thicker line to make it more visible
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
                
                # Draw keypoints
                if 'keypoints' in forward_filled_pose:
                    kpts = forward_filled_pose['keypoints']
                    for kp in kpts:
                        kx, ky = int(kp[0]), int(kp[1])
                        if kx > 0 and ky > 0:
                            cv2.circle(img, (kx, ky), 4, color, -1)
            else:
                print(f"WARNING: Forward-filled pose has invalid bbox: {bbox}")
        
        # Draw ball trajectory (MUST be before return!)
        if ball_positions is not None and len(ball_positions) > 0 and frame_idx_in_clip is not None:
            draw_ball_trajectory(img, ball_positions, frame_idx_in_clip, start_frame)
        
        return img
    
    # Original behavior (only filtered poses)
    if debug:
        debug_text = f"Detected: {len(poses)} poses | Active: {active_idx} | Idle: {idle_idx}"
        cv2.putText(img, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if len(poses) > 0:
            all_indices = ", ".join([str(i) for i in range(len(poses))])
            cv2.putText(img, f"Indices: [{all_indices}]", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    for i, pose in enumerate(poses):
        bbox = unwrap_bbox(pose['bbox'])
        if len(bbox) < 4: continue
        
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        if debug:
            color = (128, 128, 128) # Gray for untracked
            label = f"P{i}"
            
            if i == active_idx:
                if is_forward_filled:
                    color = (255, 0, 0)  # Blue - forward-filled
                    label = f"P{i} ACTIVE (FILLED)"
                else:
                    color = (0, 255, 0)  # Green for Active
                    label = f"P{i} ACTIVE"
            elif i == idle_idx:
                color = (0, 0, 255) # Red for Idle
                label = f"P{i} IDLE"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            if i not in tracked_indices:
                continue
                
            color = (255, 255, 255)
            label = ""
            
            if i == active_idx:
                color = (0, 255, 0)
                label = "ACTIVE"
            elif i == idle_idx:
                color = (0, 0, 255)
                label = "IDLE"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            if label:
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
        if 'keypoints' in pose:
            kpts = pose['keypoints']
            for kp in kpts:
                kx, ky = int(kp[0]), int(kp[1])
                if kx > 0 and ky > 0:
                    cv2.circle(img, (kx, ky), 3, color, -1)
    
    # Draw forward-filled pose if needed (draw on top of everything)
    if is_forward_filled and forward_filled_pose is not None:
        bbox = unwrap_bbox(forward_filled_pose['bbox'])
        if len(bbox) >= 4:
            x1, y1, x2, y2 = map(int, bbox[:4])
            color = (255, 0, 0)  # Blue (BGR format)
            label = "ACTIVE (FILLED)"
            # Draw with thicker line to make it more visible
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
            
            # Draw keypoints
            if 'keypoints' in forward_filled_pose:
                kpts = forward_filled_pose['keypoints']
                for kp in kpts:
                    kx, ky = int(kp[0]), int(kp[1])
                    if kx > 0 and ky > 0:
                        cv2.circle(img, (kx, ky), 4, color, -1)
        else:
            print(f"WARNING: Forward-filled pose has invalid bbox: {bbox}")
    
    # Draw ball trajectory if available (using same approach as extract_clips_with_ball.py)
    if ball_positions is not None and len(ball_positions) > 0 and frame_idx_in_clip is not None and start_frame is not None:
        draw_ball_trajectory(img, ball_positions, frame_idx_in_clip, start_frame)

    return img

def draw_ball_trajectory(frame, ball_positions, frame_idx_in_clip, start_frame, 
                        trajectory_color=(255, 255, 0), point_radius=5, trail_length=10):
    """
    Draw ball trajectory on frame.
    IDENTICAL to extract_clips_with_ball.py implementation.
    
    Args:
        frame: Frame to draw on
        ball_positions: Dict {frame_number: (x, y)} for ball positions
        frame_idx_in_clip: Current frame index in clip (0-29)
        start_frame: Starting frame number in video
        trajectory_color: BGR color for trajectory
        point_radius: Radius of ball point
        trail_length: Number of previous frames to show in trail
    """
    current_frame_num = start_frame + frame_idx_in_clip
    
    # Draw trail (previous frames)
    trail_frames = []
    for i in range(max(0, frame_idx_in_clip - trail_length), frame_idx_in_clip):
        trail_frame_num = start_frame + i
        if trail_frame_num in ball_positions:
            trail_frames.append(ball_positions[trail_frame_num])
    
    # Draw trail lines
    for i in range(len(trail_frames) - 1):
        pt1 = (int(trail_frames[i][0]), int(trail_frames[i][1]))
        pt2 = (int(trail_frames[i+1][0]), int(trail_frames[i+1][1]))
        # Fade trail (more recent = brighter)
        alpha = (i + 1) / len(trail_frames) if len(trail_frames) > 0 else 1.0
        color = tuple(int(c * alpha) for c in trajectory_color)
        cv2.line(frame, pt1, pt2, color, 2)
    
    # Draw current ball position
    if current_frame_num in ball_positions:
        ball_pos = ball_positions[current_frame_num]
        center = (int(ball_pos[0]), int(ball_pos[1]))
        # Verify coordinates are within frame bounds
        h, w = frame.shape[:2]
        if 0 <= center[0] < w and 0 <= center[1] < h:
            cv2.circle(frame, center, point_radius, trajectory_color, -1)
            cv2.circle(frame, center, point_radius + 2, (255, 255, 255), 2)  # White outline
        else:
            # Ball is outside frame - draw at edge
            x = max(0, min(w-1, center[0]))
            y = max(0, min(h-1, center[1]))
            cv2.circle(frame, (x, y), point_radius, (0, 0, 255), -1)  # Red if out of bounds

def load_ball_trajectories(ball_csv_path, camera_id=None):
    """
    Load ball trajectories from CSV.
    
    Args:
        ball_csv_path: Path to ball trajectory CSV
        camera_id: Optional camera ID to filter trajectories (e.g., 'BO-0001', 'BO-0002')
    
    Returns:
        dict: {trajectory_id: DataFrame with columns [frame_number, position_x, position_y, confidence, is_interpolated]}
    """
    if not os.path.exists(ball_csv_path):
        return {}
    
    df = pd.read_csv(ball_csv_path)
    
    # Filter by camera_id if provided
    if camera_id and 'camera_id' in df.columns:
        df = df[df['camera_id'] == camera_id].copy()
        if len(df) == 0:
            return {}
    
    # Group by trajectory_id
    trajectories = {}
    for traj_id, group in df.groupby('trajectory_id'):
        trajectories[traj_id] = group[['frame_number', 'position_x', 'position_y', 'confidence', 'is_interpolated']].copy()
        trajectories[traj_id] = trajectories[traj_id].sort_values('frame_number')
    
    return trajectories

def find_closest_trajectory_to_player(trajectories, player_poses_per_frame, start_frame, duration, 
                                     min_trajectory_length=15, frame_offset=-7):
    """
    Find the ball trajectory that is closest to the active player across the clip.
    Prefers longest trajectories that are close to the player.
    """
    if not trajectories or not player_poses_per_frame:
        return None
    
    # Apply frame offset to start_frame for trajectory matching
    # If trajectories are ahead (frame_offset negative), we need to look earlier
    adjusted_start_frame = start_frame + frame_offset
    adjusted_end_frame = adjusted_start_frame + duration
    
    candidate_trajectories = []
    
    # Debug: check trajectory frame ranges
    all_traj_frames = []
    for traj_id, traj_df in trajectories.items():
        all_traj_frames.extend(traj_df['frame_number'].tolist())
    
    if all_traj_frames:
        min_traj_frame = min(all_traj_frames)
        max_traj_frame = max(all_traj_frames)
        # Only print once per call
        if len(candidate_trajectories) == 0:  # First iteration
            pass  # Will print after loop if needed
    
    for traj_id, traj_df in trajectories.items():
        # Filter trajectory to clip frame range (with offset)
        traj_in_range = traj_df[
            (traj_df['frame_number'] >= adjusted_start_frame) & 
            (traj_df['frame_number'] < adjusted_end_frame)
        ]
        
        # Skip if no overlap
        if len(traj_in_range) == 0:
            continue
        
        total_distance = 0.0
        valid_frames = 0
        
        for i in range(duration):
            # Video frame number
            video_frame_num = start_frame + i
            # Trajectory frame number (with offset)
            traj_frame_num = adjusted_start_frame + i
            
            player_pose = player_poses_per_frame[i] if i < len(player_poses_per_frame) else None
            
            if player_pose is None:
                continue
            
            # Get ball position for this trajectory frame (with offset)
            ball_frame = traj_df[traj_df['frame_number'] == traj_frame_num]
            if ball_frame.empty:
                continue
            
            ball_pos = np.array([ball_frame.iloc[0]['position_x'], ball_frame.iloc[0]['position_y']])
            
            # Get player position (use bbox center)
            if isinstance(player_pose, dict):
                bbox = unwrap_bbox(player_pose.get('bbox', []))
            else:
                continue
            
            if len(bbox) < 4:
                continue
            
            # Use bbox center as player position
            player_pos = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            
            # Calculate distance
            distance = np.linalg.norm(ball_pos - player_pos)
            total_distance += distance
            valid_frames += 1
        
        # Only consider trajectories with at least some valid frames
        if valid_frames > 0:
            avg_distance = total_distance / valid_frames
            candidate_trajectories.append({
                'traj_id': traj_id,
                'avg_distance': avg_distance,
                'traj_length': len(traj_in_range),
                'valid_frames': valid_frames
            })
        elif len(traj_in_range) > 0:
            # Even if we couldn't calculate distance (no player poses), still consider it if it overlaps
            # Use a large default distance
            candidate_trajectories.append({
                'traj_id': traj_id,
                'avg_distance': 10000.0,  # Large default distance
                'traj_length': len(traj_in_range),
                'valid_frames': 0
            })
    
    if not candidate_trajectories:
        return None
    
    # Sort by: 1) trajectory length (longer is better), 2) distance (closer is better)
    # Prefer longest trajectories, then closest ones
    candidate_trajectories.sort(key=lambda x: (-x['traj_length'], x['avg_distance']))
    
    # Return the best trajectory (longest, then closest)
    return candidate_trajectories[0]['traj_id']

def normalize_keypoints_body_relative(keypoints, image_width, image_height):
    """
    Normalize keypoints using body-relative normalization.
    Returns 27 features: 24 body-relative + 3 absolute position features.
    
    keypoints: Array of shape (12, 2) containing x, y coordinates for the 12 body parts
    """
    # Extract body keypoints (12 keypoints)
    body_keypoints = keypoints
    
    # Calculate hip center (average of left and right hip)
    left_hip = body_keypoints[6]  # left_hip index in feature order
    right_hip = body_keypoints[7]  # right_hip index in feature order
    
    # Check if hips are valid (not NaN)
    if np.any(np.isnan(left_hip)) or np.any(np.isnan(right_hip)):
        hip_center = np.array([0.0, 0.0])
    else:
        hip_center = (left_hip + right_hip) / 2.0
    
    # Calculate shoulder width (distance between left and right shoulder)
    left_shoulder = body_keypoints[0]  # left_shoulder index
    right_shoulder = body_keypoints[1]  # right_shoulder index
    
    # Check if shoulders are valid
    if np.any(np.isnan(left_shoulder)) or np.any(np.isnan(right_shoulder)):
        shoulder_width = 1.0
    else:
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        if shoulder_width < 1.0:
            shoulder_width = 1.0
    
    # Normalize body keypoints relative to hip center and shoulder width
    body_relative = (body_keypoints - hip_center) / shoulder_width  # (12, 2)
    body_relative_flat = body_relative.flatten()  # (24,)
    
    # Calculate absolute position features (normalized by image size)
    if np.any(np.isnan(left_shoulder)) or np.any(np.isnan(right_shoulder)):
        shoulder_center = np.array([0.0, 0.0])
    else:
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
    
    # Normalize absolute positions by image size
    hip_y_abs = hip_center[1] / image_height if not np.isnan(hip_center[1]) else 0.0
    hip_x_abs = hip_center[0] / image_width if not np.isnan(hip_center[0]) else 0.0
    shoulder_center_y_abs = shoulder_center[1] / image_height if not np.isnan(shoulder_center[1]) else 0.0
    
    # Combine: 24 body-relative features + 3 absolute features = 27 features
    features = np.concatenate([
        body_relative_flat,  # (24,)
        np.array([hip_y_abs, hip_x_abs, shoulder_center_y_abs])  # (3,)
    ])
    
    return features

def save_pose_csv(poses_sequence, output_path, image_width=None, image_height=None, 
                 ball_positions=None, start_frame=0):
    """
    Saves the pose sequence to a CSV with normalized features.
    Format: frame_num, left_shoulder_x_body_rel, left_shoulder_y_body_rel, ..., hip_y_abs, hip_x_abs, shoulder_center_y_abs,
            ball_x_body_rel, ball_y_body_rel, ball_x_abs, ball_y_abs, ball_vx_body_rel, ball_vy_body_rel, ball_visible, ball_confidence
    
    If image_width/height are not provided, falls back to raw keypoint format.
    
    Args:
        ball_positions: Dict {frame_number: (x, y, confidence, is_interpolated)} for ball positions
        start_frame: Starting frame number in video (for matching ball positions)
    """
    data = []
    
    # If no image dimensions, use old format (for backward compatibility)
    if image_width is None or image_height is None:
        for frame_idx, pose in enumerate(poses_sequence):
            row = {'frame_num': frame_idx}
            if pose:
                kpts = pose['keypoints'] # usually list of [x, y] or [x, y, score]
                # keypoint_scores might be separate
                kp_scores = pose.get('keypoint_scores', [1.0]*len(kpts))
                
                for i, (kp, score) in enumerate(zip(kpts, kp_scores)):
                    row[f'kpt{i}_x'] = kp[0]
                    row[f'kpt{i}_y'] = kp[1]
                    row[f'kpt{i}_score'] = score
            else:
                # Handle missing pose? Fill 0?
                pass
            data.append(row)
    else:
        # Normalized format
        for frame_idx, pose in enumerate(poses_sequence):
            row = {'frame_num': frame_idx}
            if pose and 'keypoints' in pose:
                try:
                    kpts = np.array(pose['keypoints'])  # COCO format: 17 keypoints
                    
                    # Handle different keypoint formats: [17, 2] or [17, 3] or list of [x,y] or [x,y,score]
                    if len(kpts.shape) == 1:
                        # Flattened array, reshape if possible
                        if len(kpts) >= 34:  # At least 17 keypoints * 2
                            kpts = kpts.reshape(-1, 2)[:17]
                        else:
                            raise ValueError(f"Unexpected keypoint format: shape {kpts.shape}")
                    elif len(kpts.shape) == 2:
                        # Already in [17, 2] or [17, 3] format
                        if kpts.shape[0] < 17:
                            # Pad with zeros if we have fewer than 17 keypoints
                            padded = np.zeros((17, 2))
                            padded[:kpts.shape[0], :2] = kpts[:, :2]
                            kpts = padded
                    else:
                        raise ValueError(f"Unexpected keypoint format: shape {kpts.shape}")
                    
                    # Extract 12 body keypoints in the correct order
                    body_kpts_12 = np.zeros((12, 2))
                    for coco_idx, feat_idx in COCO_TO_FEATURE_IDX.items():
                        if coco_idx < len(kpts):
                            kp = kpts[coco_idx]
                            if len(kp) >= 2:
                                body_kpts_12[feat_idx] = kp[:2]  # Take x,y
                    
                    # Normalize
                    features = normalize_keypoints_body_relative(body_kpts_12, image_width, image_height)
                    
                    # Add normalized features to row
                    for i, kp_name in enumerate(BODY_KEYPOINT_NAMES):
                        row[f'{kp_name}_x_body_rel'] = features[i * 2]
                        row[f'{kp_name}_y_body_rel'] = features[i * 2 + 1]
                    row['hip_y_abs'] = features[24]
                    row['hip_x_abs'] = features[25]
                    row['shoulder_center_y_abs'] = features[26]
                    
                    # Extract ball features if available
                    if ball_positions is not None:
                        video_frame_num = start_frame + frame_idx
                        ball_data = ball_positions.get(video_frame_num)
                        
                        # Get hip center and shoulder width for ball normalization
                        left_hip = body_kpts_12[6]
                        right_hip = body_kpts_12[7]
                        left_shoulder = body_kpts_12[0]
                        right_shoulder = body_kpts_12[1]
                        
                        if not (np.any(np.isnan(left_hip)) or np.any(np.isnan(right_hip))):
                            hip_center = (left_hip + right_hip) / 2.0
                        else:
                            hip_center = np.array([0.0, 0.0])
                        
                        if not (np.any(np.isnan(left_shoulder)) or np.any(np.isnan(right_shoulder))):
                            shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
                            if shoulder_width < 1.0:
                                shoulder_width = 1.0
                        else:
                            shoulder_width = 1.0
                        
                        # Get previous ball positions for velocity and acceleration
                        prev_video_frame = start_frame + frame_idx - 1
                        prev_prev_video_frame = start_frame + frame_idx - 2
                        prev_ball_data = ball_positions.get(prev_video_frame) if ball_positions else None
                        prev_prev_ball_data = ball_positions.get(prev_prev_video_frame) if ball_positions else None
                        
                        ball_prev_pos = None
                        ball_prev_prev_pos = None
                        if prev_ball_data:
                            ball_prev_pos = np.array([prev_ball_data[0], prev_ball_data[1]])
                        if prev_prev_ball_data:
                            ball_prev_prev_pos = np.array([prev_prev_ball_data[0], prev_prev_ball_data[1]])
                        
                        # Extract ball features
                        # ball_positions stores only (x, y) — same as extract_clips_with_ball.py
                        if ball_data:
                            ball_pos = np.array([ball_data[0], ball_data[1]])
                            ball_confidence = 1.0
                            ball_visible = True
                        else:
                            ball_pos = None
                            ball_confidence = 0.0
                            ball_visible = False
                        
                        ball_features = normalize_ball_features(
                            ball_pos, ball_prev_pos, hip_center, shoulder_width,
                            image_width, image_height, ball_visible, ball_confidence,
                            ball_prev_prev_pos=ball_prev_prev_pos
                        )
                        
                        # Add ball features to row
                        ball_feature_names = get_ball_feature_names()
                        for i, feat_name in enumerate(ball_feature_names):
                            row[feat_name] = ball_features[i]
                    else:
                        # No ball data - fill with NaN
                        ball_feature_names = get_ball_feature_names()
                        for feat_name in ball_feature_names:
                            row[feat_name] = np.nan
                except Exception as e:
                    # If normalization fails, fill with NaN
                    print(f"Warning: Failed to normalize pose at frame {frame_idx}: {e}")
                    for kp_name in BODY_KEYPOINT_NAMES:
                        row[f'{kp_name}_x_body_rel'] = np.nan
                        row[f'{kp_name}_y_body_rel'] = np.nan
                    row['hip_y_abs'] = np.nan
                    row['hip_x_abs'] = np.nan
                    row['shoulder_center_y_abs'] = np.nan
                    # Ball features also NaN
                    ball_feature_names = get_ball_feature_names()
                    for feat_name in ball_feature_names:
                        row[feat_name] = np.nan
            else:
                # Missing pose - fill with NaN
                for kp_name in BODY_KEYPOINT_NAMES:
                    row[f'{kp_name}_x_body_rel'] = np.nan
                    row[f'{kp_name}_y_body_rel'] = np.nan
                row['hip_y_abs'] = np.nan
                row['hip_x_abs'] = np.nan
                row['shoulder_center_y_abs'] = np.nan
                # Ball features also NaN
                ball_feature_names = get_ball_feature_names()
                for feat_name in ball_feature_names:
                    row[feat_name] = np.nan
            data.append(row)
        
    df = pd.DataFrame(data)
    
    # Ensure column order matches shot_predictor.py expectations
    if image_width is not None and image_height is not None:
        column_order = ['frame_num']
        for kp_name in BODY_KEYPOINT_NAMES:
            column_order.extend([f'{kp_name}_x_body_rel', f'{kp_name}_y_body_rel'])
        column_order.extend(['hip_y_abs', 'hip_x_abs', 'shoulder_center_y_abs'])
        # Add ball features
        ball_feature_names = get_ball_feature_names()
        column_order.extend(ball_feature_names)
        
        # Reorder columns if they exist
        existing_cols = [col for col in column_order if col in df.columns]
        if existing_cols:
            df = df[existing_cols]
    
    df.to_csv(output_path, index=False)


def load_resume_state(state_path):
    """Load extraction resume state from JSON file."""
    if not os.path.exists(state_path):
        return {'processed_shots': [], 'completed_csv_files': []}
    try:
        with open(state_path, 'r') as f:
            state = json.load(f)
        state.setdefault('processed_shots', [])
        state.setdefault('completed_csv_files', [])
        return state
    except Exception:
        return {'processed_shots': [], 'completed_csv_files': []}


def save_resume_state(state_path, state):
    """Persist extraction resume state atomically."""
    tmp_path = f"{state_path}.tmp"
    with open(tmp_path, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, state_path)


def main():
    parser = argparse.ArgumentParser(description="Extract shot clips and pose CSVs")
    parser.add_argument(
        '--output-dir',
        type=str,
        default=OUTPUT_DIR,
        help=f'Output directory for generated files (default: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--resume-state',
        type=str,
        default=None,
        help='Path to resume state JSON (default: <output-dir>/extraction_state.json)'
    )
    parser.add_argument(
        '--no-video',
        action='store_true',
        help='Do not save MP4 clips; export CSV only'
    )
    parser.add_argument(
        '--video-filter',
        type=str,
        default=None,
        help='Comma-separated list of video key substrings to process'
    )
    parser.add_argument(
        '--max-csv-files',
        type=int,
        default=None,
        help='Process only the first N annotation CSVs after filters (smoke test / partial run)',
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    video_filter_runtime = None
    if args.video_filter:
        video_filter_runtime = [v.strip() for v in args.video_filter.split(',') if v.strip()]

    state_path = args.resume_state or os.path.join(output_dir, 'extraction_state.json')
    state = load_resume_state(state_path)
    processed_shots = set(state.get('processed_shots', []))
    completed_csv_files = set(state.get('completed_csv_files', []))

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize MMPose
    inferencer = init_mmpose()
    
    # List CSVs from all directories
    csv_files_map = {} # filename -> full_path
    
    for d in SHOTS_CSV_DIRS:
        if not os.path.exists(d): 
            continue
        for f in os.listdir(d):
            if f.endswith('.csv'):
                csv_files_map[f] = os.path.join(d, f)
    
    if not csv_files_map:
        print("No CSV files found in directories:", SHOTS_CSV_DIRS)
        return

    csv_files = sorted(list(csv_files_map.keys()))
    
    # Filter to only videos that have ball trajectory data
    active_video_filter = video_filter_runtime if video_filter_runtime is not None else VIDEO_FILTER
    if active_video_filter is not None:
        csv_files = [f for f in csv_files if any(vid_key in f for vid_key in active_video_filter)]
    else:
        trajectory_keys = list(BALL_TRAJECTORY_MAP.keys())
        csv_files = [f for f in csv_files if any(vid_key in f for vid_key in trajectory_keys)]
    
    if not csv_files:
        print(f"No CSV files found matching trajectory map keys: {list(BALL_TRAJECTORY_MAP.keys())}")
        return

    if args.max_csv_files is not None and args.max_csv_files > 0:
        csv_files = csv_files[: args.max_csv_files]
        print(f"Limited to first {len(csv_files)} CSV file(s) (--max-csv-files)")
    
    print(f"Processing {len(csv_files)} CSV files with ball trajectories available")
    
    for csv_file in tqdm(csv_files, desc="Processing CSVs"):
        if csv_file in completed_csv_files:
            tqdm.write(f"Skipping already completed CSV file: {csv_file}")
            continue

        csv_path = csv_files_map[csv_file]
        
        df = parse_shot_csv(csv_path)
        if df.empty:
            continue
            
        video_path = get_video_path(csv_path, VIDEOS_DIRS)
        if not video_path:
            tqdm.write(f"Video not found for {csv_file}")
            continue
            
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Get calibration for this video
        K, D, H = get_calibration(video_name)
        if H is None:
            tqdm.write(f"Warning: No perspective matrix found for {video_name}, filtering might be ineffective.")
        
        # Load ball trajectories for this video (if available)
        trajectories = {}
        matched_key = None
        for vid_key in BALL_TRAJECTORY_MAP:
            if vid_key in video_name or vid_key in csv_path:
                matched_key = vid_key
                break
        
        if matched_key:
            cam_map = BALL_TRAJECTORY_MAP[matched_key]
            # Find matching camera ID from video name or csv path
            camera_id = None
            for cam_id in cam_map:
                if cam_id in video_name or cam_id in csv_path:
                    camera_id = cam_id
                    break
            # If only one camera in the map, use it
            if camera_id is None and len(cam_map) == 1:
                camera_id = list(cam_map.keys())[0]
            
            if camera_id and camera_id in cam_map:
                ball_csv_path = os.path.join(BALL_TRAJECTORY_DIR, cam_map[camera_id])
                if os.path.exists(ball_csv_path):
                    trajectories = load_ball_trajectories(ball_csv_path, camera_id=camera_id)
                    if trajectories:
                        tqdm.write(f"Loaded {len(trajectories)} ball trajectories for {video_name} (camera: {camera_id})")
                else:
                    tqdm.write(f"Warning: trajectory file not found: {ball_csv_path}")
        
        # Group by row to process each shot
        # Convert to list to use tqdm
        rows = list(df.iterrows())
        total_rows = len(rows)
        completed_rows = 0
        for idx, row in tqdm(rows, desc=f"Shots in {csv_file}", leave=False):
            shot_type = row['Shot']
            center_frame = int(row['FrameId'])
            player_label = row['Player']
            shot_key = f"{csv_file}::{idx}::{center_frame}::{shot_type}::{player_label}"

            if shot_key in processed_shots:
                completed_rows += 1
                continue
            
            # Frame range: 15 before, center, 14 after = 30 frames
            # Start = center - 15
            start_frame = max(0, center_frame - 15)
            duration = 30
            
            # Extract data with filtering
            if DEBUG_MODE:
                result = extract_clip_and_pose(
                    video_path, start_frame, duration, inferencer, K, D, H, return_all_poses=True
                )
                frames, poses_per_frame, all_poses_per_frame, filtering_info = result
            else:
                frames, poses_per_frame = extract_clip_and_pose(
                    video_path, start_frame, duration, inferencer, K, D, H, return_all_poses=False
                )
                all_poses_per_frame = None
                filtering_info = None
            
            if not frames or not poses_per_frame:
                tqdm.write(f"    Failed to extract frames for shot {shot_type} at {center_frame}")
                continue
                
            # Identify players in the CENTER frame (approx index 15)
            center_idx_in_clip = min(15, len(frames) - 1)
            center_poses = poses_per_frame[center_idx_in_clip]
            
            if DEBUG_MODE:
                tqdm.write(f"    Center frame: {len(center_poses)} valid poses, looking for '{player_label}'")
            
            active_idx_initial = identify_player(center_poses, player_label, K, D, H)
            
            if DEBUG_MODE and active_idx_initial != -1:
                # Show transformed position of selected player
                selected_pose = center_poses[active_idx_initial]
                bbox = unwrap_bbox(selected_pose['bbox'])
                foot_pos = get_foot_position(bbox)
                if H is not None:
                    transformed = transform_points([foot_pos], K, D, H)
                    if len(transformed) > 0:
                        tx, ty = transformed[0]
                        tqdm.write(f"    Selected player {active_idx_initial} at court position ({tx:.2f}, {ty:.2f})")
            
            if active_idx_initial == -1:
                tqdm.write(f"    Could not find player '{player_label}' in center frame for shot {shot_type} at {center_frame}")
                continue
                
            idle_idx_initial = get_idle_player(center_poses, active_idx_initial)
            
            tracked_active_indices = {center_idx_in_clip: active_idx_initial}
            tracked_idle_indices = {center_idx_in_clip: idle_idx_initial}
            
            # Prepare init state
            init_active_bbox = unwrap_bbox(center_poses[active_idx_initial]['bbox'])
            init_active_foot = get_foot_position(init_active_bbox)
            init_active_court_pos = None
            if H is not None:
                transformed = transform_points([init_active_foot], K, D, H)
                if len(transformed) > 0:
                    init_active_court_pos = transformed[0]
            
            init_idle_bbox = None
            init_idle_court_pos = None
            if idle_idx_initial != -1:
                init_idle_bbox = unwrap_bbox(center_poses[idle_idx_initial]['bbox'])
                init_idle_foot = get_foot_position(init_idle_bbox)
                if H is not None:
                    transformed = transform_points([init_idle_foot], K, D, H)
                    if len(transformed) > 0:
                        init_idle_court_pos = transformed[0]
            
            # --- Forward Pass ---
            curr_active_bbox = init_active_bbox
            curr_active_court_pos = init_active_court_pos
            curr_idle_bbox = init_idle_bbox
            curr_idle_court_pos = init_idle_court_pos
            
            for i in range(center_idx_in_clip + 1, len(frames)):
                poses = poses_per_frame[i]
                active_match_idx = -1
                idle_match_idx = -1
                
                # Track Active: by position first, only re-identify by label if completely lost
                if poses:
                    # Try position-based tracking first
                    active_match_idx, curr_active_bbox, curr_active_court_pos = match_player_by_position(
                        poses, curr_active_bbox, curr_active_court_pos, K, D, H, exclude_idx=-1
                    )
                    
                    # Only re-identify by label if completely lost AND we had no previous idle tracking
                    # (prevents idle from becoming active when both were present)
                    if active_match_idx == -1 and curr_active_bbox is None and curr_active_court_pos is None:
                        # Check if we had idle tracking - if so, don't re-identify (idle can't become active)
                        had_idle_tracking = (curr_idle_bbox is not None or curr_idle_court_pos is not None)
                        if not had_idle_tracking:
                            label_match_idx = identify_player(poses, player_label, K, D, H)
                            if label_match_idx != -1:
                                active_match_idx = label_match_idx
                                curr_active_bbox = unwrap_bbox(poses[label_match_idx]['bbox'])
                                foot_pos = get_foot_position(curr_active_bbox)
                                if H is not None:
                                    transformed = transform_points([foot_pos], K, D, H)
                                    if len(transformed) > 0:
                                        curr_active_court_pos = transformed[0]
                
                tracked_active_indices[i] = active_match_idx
                
                # Track Idle: by position, excluding active
                if poses:
                    idle_match_idx, curr_idle_bbox, curr_idle_court_pos = match_player_by_position(
                        poses, curr_idle_bbox, curr_idle_court_pos, K, D, H, exclude_idx=active_match_idx
                    )
                    
                    # If we have active but no idle, and there are multiple poses, assign the other one as idle
                    if active_match_idx != -1 and idle_match_idx == -1 and len(poses) > 1:
                        for p_idx in range(len(poses)):
                            if p_idx != active_match_idx:
                                idle_match_idx = p_idx
                                curr_idle_bbox = unwrap_bbox(poses[p_idx]['bbox'])
                                foot_pos = get_foot_position(curr_idle_bbox)
                                if H is not None:
                                    transformed = transform_points([foot_pos], K, D, H)
                                    if len(transformed) > 0:
                                        curr_idle_court_pos = transformed[0]
                                break
                
                tracked_idle_indices[i] = idle_match_idx

            # --- Backward Pass ---
            curr_active_bbox = init_active_bbox
            curr_active_court_pos = init_active_court_pos
            curr_idle_bbox = init_idle_bbox
            curr_idle_court_pos = init_idle_court_pos
            
            for i in range(center_idx_in_clip - 1, -1, -1):
                poses = poses_per_frame[i]
                active_match_idx = -1
                idle_match_idx = -1
                
                # Track Active: by position first, only re-identify by label if completely lost
                if poses:
                    active_match_idx, curr_active_bbox, curr_active_court_pos = match_player_by_position(
                        poses, curr_active_bbox, curr_active_court_pos, K, D, H, exclude_idx=-1
                    )
                    
                    # Only re-identify by label if completely lost AND we had no previous idle tracking
                    if active_match_idx == -1 and curr_active_bbox is None and curr_active_court_pos is None:
                        had_idle_tracking = (curr_idle_bbox is not None or curr_idle_court_pos is not None)
                        if not had_idle_tracking:
                            label_match_idx = identify_player(poses, player_label, K, D, H)
                            if label_match_idx != -1:
                                active_match_idx = label_match_idx
                                curr_active_bbox = unwrap_bbox(poses[label_match_idx]['bbox'])
                                foot_pos = get_foot_position(curr_active_bbox)
                                if H is not None:
                                    transformed = transform_points([foot_pos], K, D, H)
                                    if len(transformed) > 0:
                                        curr_active_court_pos = transformed[0]
                
                tracked_active_indices[i] = active_match_idx
                
                # Track Idle: by position, excluding active
                if poses:
                    idle_match_idx, curr_idle_bbox, curr_idle_court_pos = match_player_by_position(
                        poses, curr_idle_bbox, curr_idle_court_pos, K, D, H, exclude_idx=active_match_idx
                    )
                    
                    # If we have active but no idle, and there are multiple poses, assign the other one as idle
                    if active_match_idx != -1 and idle_match_idx == -1 and len(poses) > 1:
                        for p_idx in range(len(poses)):
                            if p_idx != active_match_idx:
                                idle_match_idx = p_idx
                                curr_idle_bbox = unwrap_bbox(poses[p_idx]['bbox'])
                                foot_pos = get_foot_position(curr_idle_bbox)
                                if H is not None:
                                    transformed = transform_points([foot_pos], K, D, H)
                                    if len(transformed) > 0:
                                        curr_idle_court_pos = transformed[0]
                                break
                
                tracked_idle_indices[i] = idle_match_idx

            # NOW match ball trajectory to active player (after tracking is complete)
            # Build active player pose sequence (only active player, not idle/invalid)
            active_player_poses = []
            for i in range(len(frames)):
                act_idx = tracked_active_indices.get(i, -1)
                if act_idx != -1 and act_idx < len(poses_per_frame[i]):
                    active_player_poses.append(poses_per_frame[i][act_idx])
                else:
                    active_player_poses.append(None)
            
            # Match ball trajectory using ONLY active player poses
            ball_positions = {}
            if trajectories:
                best_traj_id = find_closest_trajectory_to_player(
                    trajectories, active_player_poses, start_frame, duration,
                    min_trajectory_length=MIN_TRAJECTORY_LENGTH, frame_offset=FRAME_OFFSET
                )
                
                if best_traj_id is not None:
                    traj_df = trajectories[best_traj_id]
                    for _, row in traj_df.iterrows():
                        traj_frame_num = int(row['frame_number'])
                        # Convert trajectory frame to video frame (subtract offset)
                        # FRAME_OFFSET is negative (e.g., -7), meaning trajectories are ahead
                        # So video_frame = traj_frame - FRAME_OFFSET (subtracting negative adds)
                        video_frame_num = traj_frame_num - FRAME_OFFSET
                        if start_frame <= video_frame_num < start_frame + duration:
                            # Store ONLY (x, y) — same as extract_clips_with_ball.py
                            ball_positions[video_frame_num] = (row['position_x'], row['position_y'])
                    
                    # Always print if we have ball data
                    tqdm.write(f"    ✅ Matched ball trajectory {best_traj_id}: {len(ball_positions)} frames with ball data (clip: {start_frame} to {start_frame+duration})")
                    # Debug: show sample ball positions
                    if len(ball_positions) > 0:
                        sample_frames = sorted(ball_positions.keys())[:3]
                        for sf in sample_frames:
                            data = ball_positions[sf]
                            tqdm.write(f"      Sample frame {sf}: ball at ({int(data[0])}, {int(data[1])})")
                else:
                    # Debug: show frame ranges
                    if trajectories:
                        all_frames = []
                        for traj_df in trajectories.values():
                            all_frames.extend(traj_df['frame_number'].tolist())
                        if all_frames:
                            min_traj = min(all_frames)
                            max_traj = max(all_frames)
                            adjusted_start = start_frame + FRAME_OFFSET
                            adjusted_end = adjusted_start + duration
                            tqdm.write(f"    ❌ No matching ball trajectory found for clip {start_frame} to {start_frame+duration}")
                            tqdm.write(f"       Looking in trajectory range [{adjusted_start}, {adjusted_end}), trajectories span [{min_traj}, {max_traj}]")
                        else:
                            tqdm.write(f"    ❌ No matching ball trajectory found (trajectories empty)")
                    else:
                        tqdm.write(f"    ❌ No trajectories available")
            else:
                tqdm.write(f"    ⚠️  No trajectories available for this video")

            # Generate output with forward-fill for missing frames (max 5 consecutive)
            pose_data_sequence = []  # Active player
            idle_pose_data_sequence = []  # Idle player
            last_valid_pose = None
            last_valid_idle_pose = None
            consecutive_lost_frames = 0
            consecutive_lost_idle_frames = 0
            MAX_FORWARD_FILL = 5
            
            # Create output video (optional)
            if (not args.no_video) and frames:
                h, w = frames[0].shape[:2]
                clip_filename = f"{video_name}_{center_frame}_{shot_type}_{player_label}.mp4"
                clip_path = os.path.join(output_dir, clip_filename)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(clip_path, fourcc, 30.0, (w, h))
            else:
                out = None
            
            for i in range(len(frames)):
                frame = frames[i]
                poses = poses_per_frame[i]
                
                act_idx = tracked_active_indices.get(i, -1)
                idl_idx = tracked_idle_indices.get(i, -1)
                
                # Check if we're forward-filling
                is_forward_filled = False
                forward_filled_pose = None
                if act_idx == -1 and last_valid_pose is not None and consecutive_lost_frames < MAX_FORWARD_FILL:
                    is_forward_filled = True
                    forward_filled_pose = last_valid_pose
                
                # Overlay - pass all poses info if available
                all_poses_info = all_poses_per_frame[i] if (all_poses_per_frame and i < len(all_poses_per_frame)) else None
                # Get filtering info for this frame
                frame_filtering_info = None
                if filtering_info is not None and i < len(filtering_info.get('filtered_per_frame', [])):
                    frame_filtering_info = {
                        'filtered': filtering_info['filtered_per_frame'][i] if i < len(filtering_info['filtered_per_frame']) else set(),
                        'flickering': filtering_info['flickering_per_frame'][i] if i < len(filtering_info['flickering_per_frame']) else set()
                    }
                # Pass full ball_positions dict for trail drawing
                # Always verify ball_positions is populated on first frame (critical for debugging)
                if i == 0:
                    if ball_positions and len(ball_positions) > 0:
                        tqdm.write(f"      ✓ Passing ball_positions to draw_overlay: {len(ball_positions)} entries, first frame {start_frame}")
                    else:
                        tqdm.write(f"      ⚠️  WARNING - ball_positions is empty or None when calling draw_overlay!")
                
                vis_frame = draw_overlay(frame, poses, act_idx, idl_idx, debug=DEBUG_MODE, 
                                        all_poses_info=all_poses_info, is_forward_filled=is_forward_filled,
                                        forward_filled_pose=forward_filled_pose, filtering_info=frame_filtering_info,
                                        ball_positions=ball_positions, frame_idx_in_clip=i, start_frame=start_frame)
                if out is not None:
                    out.write(vis_frame)
                
                # Collect active pose data with forward-fill (max 5 consecutive frames)
                if act_idx != -1 and act_idx < len(poses):
                    last_valid_pose = poses[act_idx]
                    pose_data_sequence.append(poses[act_idx])
                    consecutive_lost_frames = 0  # Reset counter
                else:
                    # Use previous frame's pose if available and within limit
                    if last_valid_pose is not None and consecutive_lost_frames < MAX_FORWARD_FILL:
                        pose_data_sequence.append(last_valid_pose)
                        consecutive_lost_frames += 1
                    else:
                        pose_data_sequence.append(None)  # Lost for too long or never had valid pose
                        consecutive_lost_frames += 1
                
                # Collect idle pose data with forward-fill (max 5 consecutive frames)
                if idl_idx != -1 and idl_idx < len(poses):
                    last_valid_idle_pose = poses[idl_idx]
                    idle_pose_data_sequence.append(poses[idl_idx])
                    consecutive_lost_idle_frames = 0  # Reset counter
                else:
                    # Use previous frame's pose if available and within limit
                    if last_valid_idle_pose is not None and consecutive_lost_idle_frames < MAX_FORWARD_FILL:
                        idle_pose_data_sequence.append(last_valid_idle_pose)
                        consecutive_lost_idle_frames += 1
                    else:
                        idle_pose_data_sequence.append(None)  # Lost for too long or never had valid pose
                        consecutive_lost_idle_frames += 1
            
            if out is not None:
                out.release()
            
            # Save CSV with normalized features
            # Get image dimensions from first frame (or from video if available)
            if frames and len(frames) > 0:
                img_height, img_width = frames[0].shape[:2]
            else:
                # Fallback: try to get from video if we have the path
                # This shouldn't happen if extraction succeeded, but handle gracefully
                img_height, img_width = None, None
                if video_path and os.path.exists(video_path):
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
            
            # Save active player pose CSV (with ball features)
            # Note: ball_positions was already matched earlier for video visualization
            pose_csv_filename = f"{video_name}_{center_frame}_{shot_type}_{player_label}_pose.csv"
            save_pose_csv(pose_data_sequence, os.path.join(output_dir, pose_csv_filename), 
                         image_width=img_width, image_height=img_height,
                         ball_positions=ball_positions if ball_positions else None,
                         start_frame=start_frame)
            
            # Save idle player pose CSV (if we have idle data).
            # We pass the same ball_positions so idle samples also get ball features
            # normalized w.r.t. the idle player's body (consistent with non-idle CSVs).
            if any(p is not None for p in idle_pose_data_sequence):
                idle_pose_csv_filename = f"{video_name}_{center_frame}_idle_{player_label}_pose.csv"
                save_pose_csv(idle_pose_data_sequence, os.path.join(output_dir, idle_pose_csv_filename), 
                             image_width=img_width, image_height=img_height,
                             ball_positions=ball_positions if ball_positions else None,
                             start_frame=start_frame)

            # Mark shot as completed and persist immediately for crash-safe resume.
            processed_shots.add(shot_key)
            state['processed_shots'] = sorted(processed_shots)
            save_resume_state(state_path, state)
            completed_rows += 1

        # CSV file completed end-to-end.
        if completed_rows >= total_rows:
            completed_csv_files.add(csv_file)
            state['completed_csv_files'] = sorted(completed_csv_files)
            save_resume_state(state_path, state)

if __name__ == "__main__":
    main()
