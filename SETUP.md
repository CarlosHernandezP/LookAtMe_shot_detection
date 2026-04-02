# Setup Guide

This guide explains how to set up the virtual environment for the shot detection classifier.

## Prerequisites

1. **Install `uv`** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Python 3.10** - The setup script will create a venv with Python 3.10

## Quick Setup

Run the setup script to create the virtual environment and install all dependencies:

```bash
./setup.sh
```

This will:
- Create a `.venv` virtual environment with Python 3.10
- Install PyTorch 2.1.0 with CUDA 12.1 support
- Install OpenCV, pandas, numpy, scikit-learn, and other dependencies
- Install MM-Libraries (MMEngine, MMCV, MMDetection, MMPose)
- Download MMPose RTMO-s config and checkpoint if not present

## Activating the Environment

After setup, activate the virtual environment:

```bash
source .venv/bin/activate
```

## Running Scripts

Once the environment is activated, you can run the shot detection scripts:

```bash
# Extract pose data from videos
python -m shot_detector.extract_shots

# Train + export XGBoost shot classifier (see shot_detector/README.md)
uv run python shot_detector/train_shot_model.py --data-dir shot_detector/data_csv_only --output-dir shot_detector/retrain_results_wall_flat --cv-folds 5 --export-counts

# Extract clips with ball trajectories (for visualization)
python -m shot_detector.extract_clips_with_ball
```

### Alternative: Run without activating

You can also run scripts directly without activating the venv:

```bash
# Important: Unset LD_LIBRARY_PATH to avoid Cursor AppImage conflicts
unset LD_LIBRARY_PATH && .venv/bin/python -m shot_detector.extract_shots
```

## Troubleshooting

### LD_LIBRARY_PATH conflicts

If you encounter library conflicts (especially when using Cursor IDE), unset the variable:

```bash
unset LD_LIBRARY_PATH
source .venv/bin/activate
```

### MMPose installation issues

If MMPose installation fails, the script installs it with `--no-deps` to avoid dependency conflicts. The required dependencies (mmcv, mmdet, mmengine) are installed separately via MIM.

### CUDA/GPU issues

The setup installs PyTorch with CUDA 12.1 support. If you need a different CUDA version, modify the PyTorch installation line in `setup.sh`:

```bash
uv pip install "torch==2.1.0" "torchvision==0.16.0" --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
```

## Dependencies

The setup installs:
- **PyTorch 2.1.0** with CUDA 12.1
- **MMPose 1.1.0+** for pose estimation
- **MMCV 2.1.0** (compatible with MMPose 1.x)
- **MMDetection 3.x** (compatible with MMPose 1.x)
- **scikit-learn** for Random Forest classifier
- **OpenCV** for video processing
- **pandas, numpy** for data handling
- **matplotlib, seaborn** for visualization

## Project Structure

After setup, your project should have:
- `.venv/` - Virtual environment
- `configs/` - MMPose configuration files
- `model_weights/` - Model checkpoints and trained models
- `shot_detector/` - Main package with shot detection code
- `parameters/` - Camera calibration parameters

