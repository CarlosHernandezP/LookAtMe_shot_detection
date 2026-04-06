#!/bin/bash
set -e

# 1. Sanitize Environment (Fix Cursor AppImage leak)
unset LD_LIBRARY_PATH
echo "Sanitized LD_LIBRARY_PATH..."

# 2. Clean up old venv
echo "Removing old .venv..."
rm -rf .venv

# 3. Create new venv with Python 3.11 (avoids conflict with Cursor's Python 3.10)
echo "Creating venv with Python 3.11..."
uv venv .venv --python 3.11

# 4. Activate venv
source .venv/bin/activate

# 5. Install Base Dependencies (PyTorch, OpenCV, etc.)
# setuptools>=82 removed pkg_resources; openmim/mim still imports it — pin below 82.
echo "Installing PyTorch and base deps..."
uv pip install "torch==2.1.0" "torchvision==0.16.0" --index-url https://download.pytorch.org/whl/cu121
uv pip install "setuptools>=70,<82" opencv-python tqdm openmim tomli platformdirs packaging

# Pin NumPy before mim/mm stack (avoids torch/mim failing against NumPy 2.x)
echo "Pinning NumPy <2.0 (binary compatibility)..."
uv pip install "numpy<2.0"

# 6. Install MM-Libraries Dependencies manually (skip chumpy for now if possible, install munkres)
echo "Installing helper deps..."
uv pip install munkres xtcocotools json_tricks

# 7. Install MM-Libraries
echo "Installing MM-Libraries..."
# Install engine and cv
mim install mmengine
mim install "mmcv==2.1.0"
mim install "mmdet>=3.1.0,<3.3.0"

# Install mmpose without deps to avoid chumpy build failure, since we installed critical deps manually
uv pip install --no-deps "mmpose>=1.1.0"

# 9. Shot classifier retraining (XGBoost on pose CSVs)
echo "Installing training stack for shot_detector/train_shot_model.py..."
uv pip install "xgboost>=2.0" "scikit-learn>=1.3" matplotlib joblib

echo "========================================"
echo "Environment setup complete!"
echo "Run pose extraction (use --no-sync so uv does not strip mmpose/mm stack):"
echo "  unset LD_LIBRARY_PATH && uv run --no-sync python -m shot_detector.extract_shots"
echo "Or activate:  source .venv/bin/activate && python -m shot_detector.extract_shots"
echo "========================================"






