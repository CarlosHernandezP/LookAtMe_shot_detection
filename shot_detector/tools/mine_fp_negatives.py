"""
Self-training hard-negative mining: run a trained TCN on a fresh pool of
down-time (idle) windows, keep the ones the model WRONGLY classifies as a shot
(confident false positives) and copy them out as extra idle training examples.

These are the confusable motions (high-fives, ball pick-ups, ready stances that
look like a serve) — exactly what inflates production shot counts. Adding them
as idle and retraining teaches the model to reject them. No manual labels.

Usage:
    uv run python -m shot_detector.tools.mine_fp_negatives \
        --bundle shot_detector/runs/tcn_final_v2/seq_tcn_model.pt \
        --pool-dir shot_detector/data/idle_pool \
        --out-dir shot_detector/data/fp_hard_negatives \
        --court-features dist_to_near_baseline,court_x --conf 0.5
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd
import torch

from shot_detector.pose_io import N_FEATURES_RAW, load_pose_csv, pad_or_trim_sequence
from shot_detector.ball_features import get_ball_feature_names
from shot_detector.train_sequence_model import TCN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--pool-dir", required=True, help="dir of mined idle CSVs (the candidate pool)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--court-features", default="")
    ap.add_argument("--conf", type=float, default=0.5, help="keep FP if max non-idle prob >= this")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    court_cols = [c.strip() for c in args.court_features.split(",") if c.strip()]

    b = torch.load(args.bundle, map_location=args.device, weights_only=False)
    classes = list(b["classes"])
    idle_idx = classes.index("idle")
    mu = np.asarray(b["norm_mean"], dtype=np.float32)
    sd = np.asarray(b["norm_std"], dtype=np.float32)
    model = TCN(b["n_features"], len(classes), **b["arch_kwargs"])
    model.load_state_dict(b["state_dict"])
    model.to(args.device).eval()
    n_ball = len(get_ball_feature_names())

    files = sorted(glob.glob(os.path.join(args.pool_dir, "*_idle_*_pose.csv")))
    print(f"pool: {len(files)} idle windows; model featdim={b['n_features']}")

    kept = 0
    batch_paths, batch_X = [], []

    def flush():
        nonlocal kept
        if not batch_X:
            return
        X = np.nan_to_num(np.array(batch_X, dtype=np.float32), nan=0.0)
        X = (X - mu) / sd
        with torch.no_grad():
            proba = torch.softmax(model(torch.tensor(X, device=args.device)), dim=1).cpu().numpy()
        for p, path in zip(proba, batch_paths):
            pred = int(p.argmax())
            if pred != idle_idx and p[pred] >= args.conf:  # confident false shot
                shutil.copy(path, os.path.join(args.out_dir, os.path.basename(path)))
                kept += 1
        batch_paths.clear()
        batch_X.clear()

    for path in files:
        feats = load_pose_csv(path)
        if feats is None or feats.shape[1] < N_FEATURES_RAW:
            continue
        feats = pad_or_trim_sequence(feats[:, :N_FEATURES_RAW])
        if court_cols:
            df = pd.read_csv(path, usecols=lambda c: c in court_cols)
            court = pad_or_trim_sequence(df[court_cols].to_numpy(dtype=np.float32))
            feats = np.concatenate([feats, np.nan_to_num(court, nan=0.0)], axis=1)
        batch_paths.append(path)
        batch_X.append(feats)
        if len(batch_X) >= 256:
            flush()
    flush()

    print(f"kept {kept} confident false-positive windows (>= {args.conf}) -> {args.out_dir}")
    print(f"false-positive rate on pool: {kept / max(1, len(files)):.3f}")


if __name__ == "__main__":
    main()
