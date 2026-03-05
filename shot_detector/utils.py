import os
import cv2
import numpy as np
import pandas as pd

def load_fisheye_params(path):
    """
    Load fisheye parameters from a file.
    """
    parameters = {}
    if not os.path.exists(path):
        # Fallback or error
        print(f"Warning: Fisheye params file not found: {path}")
        return None, None

    with open(path, 'r') as file:
        for line in file:
            parts = line.strip().split('=')
            if len(parts) == 2:
                key, value = map(str.strip, parts)
                try:
                    parameters[key] = float(value)
                except ValueError:
                    pass

    if not parameters:
        return None, None

    fx = parameters.get('fx', 0)
    fy = parameters.get('fy', 0)
    cx = parameters.get('cx', 0)
    cy = parameters.get('cy', 0)
    k1 = parameters.get('k1', 0)
    k2 = parameters.get('k2', 0)
    p1 = parameters.get('p1', 0)
    p2 = parameters.get('p2', 0)

    mtx = np.array([[fx, 0., cx],
                    [0., fy, cy],
                    [0., 0., 1.]])
    dist = np.array([[k1, k2, p1, p2]])

    return mtx, dist

def load_perspective_matrix(path):
    """
    Load perspective transformation matrix from a file.
    Returns float32 matrix for OpenCV compatibility.
    """
    if not os.path.exists(path):
        print(f"Warning: Perspective matrix file not found: {path}")
        return None
        
    try:
        matrix = np.loadtxt(path)
        if matrix.shape != (3, 3):
            print(f"Error: Perspective matrix shape is {matrix.shape}, expected (3, 3)")
            return None
        # Ensure float32 for OpenCV compatibility
        matrix = matrix.astype(np.float32)
        return matrix
    except Exception as e:
        print(f"Error loading perspective matrix: {e}")
        return None

def transform_points(points, K=None, D=None, H=None):
    """
    Undistort and/or apply perspective transformation.
    points: List of (x,y) tuples or numpy array (Nx2)
    """
    if points is None or len(points) == 0:
        return np.array([])

    if isinstance(points, list):
        points = np.float32(points)

    # Reshape to (N, 1, 2)
    result = points.reshape(-1, 1, 2)

    if K is not None and D is not None:
        result = cv2.fisheye.undistortPoints(result, K, D, None, K)

    if H is not None:
        result = cv2.perspectiveTransform(result, H)
        
    return result.reshape(-1, 2)

def get_foot_position(bbox):
    """
    Returns the foot position (bottom center) of a bounding box [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def unwrap_bbox(bbox):
    """
    Helper to unwrap nested bbox structure ([x1,y1,x2,y2],) -> [x1,y1,x2,y2]
    """
    if len(bbox) == 1 and isinstance(bbox[0], (list, tuple, np.ndarray)):
        return bbox[0]
    return bbox

# --- Stubs for missing functions required by extract_shots.py ---

def parse_shot_csv(csv_path):
    """
    Parses the shot CSV file.
    Expected columns: Shot, FrameId, Player, etc.
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Ensure required columns exist
        required = ['Shot', 'FrameId', 'Player']
        if not all(col in df.columns for col in required):
            print(f"CSV {csv_path} missing required columns.")
            return pd.DataFrame()
        
        # Coerce FrameId to numeric, handling errors
        df['FrameId'] = pd.to_numeric(df['FrameId'], errors='coerce')
        # Drop rows where FrameId is NaN
        df = df.dropna(subset=['FrameId'])
        
        return df
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return pd.DataFrame()

def get_video_path(csv_path, videos_dirs):
    """
    Infers video path from csv path or content.
    Handles 'annotation_' prefix and suffix like '_period1', '_period2', '_1'.
    Matches video by camera ID (BO-0001 or BO-0002) extracted from annotation filename.
    Searches in a list of video directories.
    """
    filename = os.path.basename(csv_path)
    # Remove extension
    base = os.path.splitext(filename)[0]
    
    # Remove prefix "annotation_"
    if base.startswith("annotation_"):
        base = base[len("annotation_"):]
        
    # Extract camera ID (BO-0001 or BO-0002) from the filename
    camera_id = None
    if 'BO-0001' in base:
        camera_id = 'BO-0001'
    elif 'BO-0002' in base:
        camera_id = 'BO-0002'
    
    # Remove period suffix like "_period1", "_period2", "_period3"
    if '_period' in base:
        base = base.split('_period')[0]
    # Also remove numeric suffix like "_1", "_2" if it's at the end
    elif '_' in base:
        parts = base.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            base = parts[0]
            
    # Now try to find the video in all directories
    if isinstance(videos_dirs, str):
        videos_dirs = [videos_dirs]
        
    # First, try exact match with base name
    for d in videos_dirs:
        if not os.path.exists(d): continue
        for ext in ['.mp4', '.mov', '.avi']:
            vid_path = os.path.join(d, base + ext)
            if os.path.exists(vid_path):
                return vid_path
    
    # If camera_id was found, try to match by camera ID
    if camera_id:
        for d in videos_dirs:
            if not os.path.exists(d): continue
            # List all videos in directory and find one matching the camera ID
            for f in os.listdir(d):
                if f.endswith(('.mp4', '.mov', '.avi')):
                    if camera_id in f:
                        vid_path = os.path.join(d, f)
            if os.path.exists(vid_path):
                return vid_path
            
    # Fallback to first dir + base + .mp4 even if not found
    return os.path.join(videos_dirs[0], base + ".mp4")

def identify_player(poses, player_label, K, D, H):
    """
    Identifies which pose index corresponds to the player_label based on spatial position.
    Uses pixel/image coordinates (not court coordinates).
    
    Args:
        poses: List of pose dicts (result from mmpose)
        player_label: 'left', 'right', 'top', 'bottom', 'top_left', etc.
        K, D, H: Calibration matrices (not used for position matching, but kept for API compatibility)
        
    Returns:
        index of the best matching pose, or -1 if none found.
    """
    if not poses:
        return -1
    
    best_idx = -1
    best_score = float('-inf')
    
    # Pre-calculate pixel positions for all candidates
    positions = []
    for i, pose in enumerate(poses):
        bbox = unwrap_bbox(pose['bbox'])
        if len(bbox) < 4:
            positions.append(None)
            continue
            
        foot_pos = get_foot_position(bbox)
        # Use pixel coordinates directly (not transformed to court space)
        positions.append(foot_pos)

    # Normalize label
    label = player_label.lower().strip()
    
    scores_list = []  # For debugging
    
    for i, pos in enumerate(positions):
        if pos is None: continue
        
        x, y = pos
        score = 0
        
        # Simple heuristics based on label
        # Using pixel/image coordinates:
        # - X: 0 is left edge, larger X is right edge
        # - Y: 0 is top edge, larger Y is bottom edge
        
        if 'left' in label:
            # Prefer smaller X (left side of image)
            score -= x # Smaller X = Higher Score
        elif 'right' in label:
            # Prefer larger X (right side of image)
            score += x # Larger X = Higher Score
            
        if 'top' in label:
            # Prefer smaller Y (top of image, Y=0 is at top)
            score -= y # Smaller Y = Higher Score
        elif 'bottom' in label:
            # Prefer larger Y (bottom of image)
            score += y # Larger Y = Higher Score
             
        # Combine if like 'top_left'
        
        scores_list.append((i, score, x, y))
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    # Debug output
    if len(scores_list) > 1:
        print(f"  identify_player('{player_label}'): Scores: {[(i, f'{s:.2f}', f'({x:.1f},{y:.1f})') for i, s, x, y in scores_list]}")
        print(f"  -> Selected index {best_idx} with score {best_score:.2f}")
    
    return best_idx

def get_idle_player(poses, active_idx):
    """
    Returns the index of the player that is NOT the active_idx.
    Selects the first available index that isn't active_idx.
    """
    if not poses or len(poses) < 2:
        return -1
    
    # If active_idx is -1, just return the first one? Or -1?
    if active_idx == -1:
        return 0 # Fallback
    
    for i in range(len(poses)):
        if i != active_idx:
            return i
    return -1
