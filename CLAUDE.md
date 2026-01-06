# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**sskit** is the development kit for the Spiideo SoccerNet SynLoc task - single-frame athlete detection and world coordinate localization in soccer images using synthetic data.

**Baseline paper**: "Spiideo SoccerNet SynLoc - Single Frame World Coordinate Athlete Detection and Localization with Synthetic Data" (VISAPP 2025)

## Common Commands

```bash
# Install package in development mode
pip install -e .

# Install dependencies
pip install -r .devcontainer/requirements.txt

# Run baseline (YOLOv5 detection + ground projection)
python baseline.py

# Convert dataset to COCO format
python make_coco.py

# Run tests
pytest tests/

# Build and test codabench scoring
cd codabench && ./build.sh && ./test.sh
```

## Architecture

### Coordinate System Pipeline
The project uses 4 coordinate systems with transformation functions in `sskit/camera.py`:

```
World (meters on pitch)
    ↓ world_to_undistorted()
Undistorted image space
    ↓ distort()
Normalized image (-0.5 to 0.5)
    ↓ unnormalize()
Pixel coordinates
```

Key functions:
- `world_to_image()` - Full world→pixel projection
- `image_to_ground()` - Full pixel→world (ground plane) projection
- `load_camera()` - Load camera matrix and distortion polynomials

### Evaluation Metric
Uses **mAP-LocSim** (COCO AP adapted with location similarity) in `sskit/coco.py`:
- `LocSimCOCOeval` - Main evaluation class
- Uses exponential location similarity: `exp(-d/tau)` where tau∈{1,5} meters
- Extracts 3D position from `position_on_pitch` annotation key

### Data Format
Each scene contains:
- `rgb.jpg` - 4K image
- `objects.json` - Annotations with 3D keypoints
- `camera_matrix.npy` - 3x4 projection matrix
- `lens.json` - Distortion polynomial coefficients
- `segmentations.npy.gz` - Per-pixel masks

### Module Responsibilities
- `sskit/camera.py` - Camera model, projections, distortion
- `sskit/utils.py` - Image I/O (`imread`/`imwrite`), `Draw` class
- `sskit/metric.py` - Hungarian matching for detection evaluation
- `sskit/coco.py` - COCO evaluation with LocSim metric

## Dataset Locations

Data zip files in `data/`:
- `train.zip` (12.8GB), `val.zip` (2GB), `test.zip` (2.8GB), `challenge.zip` (3.5GB)
- `annotations.zip` - Pre-built COCO JSON files

Extract with: `cd data && for z in *.zip; do unzip -n "$z"; done`

## Testing

```bash
pytest tests/test_camera.py  # Camera transformation tests
```
