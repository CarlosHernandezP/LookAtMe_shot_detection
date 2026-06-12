"""
Loader for players_reid.csv produced by the LookAtMeProtoApp pipeline.

Schema: frame_number, object_type, object_id, bbox, court_x, court_y,
        confidence, player_id

bbox is a JSON-style list "[x1, y1, x2, y2]" in image pixels.
"""
from __future__ import annotations

import ast
import os
from typing import Dict, List

import pandas as pd


def load_players_reid_by_frame(csv_path: str) -> Dict[int, List[dict]]:
    """
    Return {frame_number: [record, ...]} where each record is:
        {player_id, object_id, bbox=[x1,y1,x2,y2], court_x, court_y, confidence}
    Only rows with object_type == 'player' are kept. Empty dict on missing file.
    """
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    df = df[df["object_type"] == "player"]
    if df.empty:
        return {}

    by_frame: Dict[int, List[dict]] = {}
    for row in df.itertuples(index=False):
        try:
            bbox = ast.literal_eval(row.bbox)
        except (ValueError, SyntaxError):
            continue
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        rec = {
            "player_id": int(row.player_id),
            "object_id": int(row.object_id),
            "bbox": [float(b) for b in bbox],
            "court_x": float(row.court_x),
            "court_y": float(row.court_y),
            "confidence": float(row.confidence),
        }
        by_frame.setdefault(int(row.frame_number), []).append(rec)
    return by_frame
