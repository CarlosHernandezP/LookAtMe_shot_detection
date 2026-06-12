"""
IoU-based matcher: assigns a stable player_id from players_reid.csv to each
pose detection in a frame.
"""
from __future__ import annotations

from typing import Iterable, List, Optional


def bbox_iou(b1: Iterable[float], b2: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def match_poses_to_reid(
    pose_bboxes: List[List[float]],
    reid_recs: List[dict],
    iou_threshold: float = 0.3,
) -> List[Optional[int]]:
    """
    Greedy max-IoU matching. Returns a list of length len(pose_bboxes); element
    i is the matched player_id or None.
    """
    n = len(pose_bboxes)
    if n == 0 or not reid_recs:
        return [None] * n
    pairs = []
    for pi, pb in enumerate(pose_bboxes):
        for ri, rec in enumerate(reid_recs):
            v = bbox_iou(pb, rec["bbox"])
            if v >= iou_threshold:
                pairs.append((v, pi, ri))
    pairs.sort(reverse=True)
    out: List[Optional[int]] = [None] * n
    used_p, used_r = set(), set()
    for v, pi, ri in pairs:
        if pi in used_p or ri in used_r:
            continue
        out[pi] = reid_recs[ri]["player_id"]
        used_p.add(pi)
        used_r.add(ri)
    return out
