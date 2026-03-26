# Fast Volleyball Ball Tracking → Vertical Reels

A complete high-speed pipeline for volleyball ball detection, tracking, and automatic generation of **9:16 vertical reels** with the ball always centered.
Achieves **up to 270 FPS** on a regular CPU thanks to lightweight grayscale models and optimized inference engines.

## Features

1. **Dual Decoding Logic**: Supports both traditional Heatmap-based models and the new **Grid-based** models for superior sub-pixel accuracy.
2. **Multi-Runtime Support**: High-performance inference using **ONNX Runtime** and **OpenVINO** (for Intel hardware).
3. **Automated Pipeline**: Ball detection → Track calculation → Rally extraction → Vertical Reel generation.
4. **9:16 Vertical Reels**: Smooth ball-centered cropping with lead offsets in movement direction.

## Model Comparison

| Model | FPS | Acc@5px (all) | Acc@5px (visible) |
| :--- | :--- | :--- | :--- |
| VballNetV1_seq9_grayscale_148_h288_w512.onnx | 138.68 | 87.25% | 86.43% |
| VballNetV1_seq9_grayscale_204_h288_w512.onnx | 138.39 | 85.95% | 84.88% |
| VballNetV2_seq9_grayscale_320_h288_w512.onnx | 114.22 | 83.01% | 82.56% |
| VballNetGridV1b_seq9_grayscale_...onnx | 117.55 | 75.49% | 74.03% |
| VballNetFastV1_seq9_grayscale_233_...onnx | **271.86** | 73.20% | 68.99% |

## Installation

```bash
# Clone the repository
git clone https://github.com/asigatchov/fast-volleyball-tracking-inference.git
cd fast-volleyball-tracking-inference

# Install dependencies (uv is recommended)
uv sync
```

## Quick Start

```bash
# 1) Detection -> ball.csv (using ONNX)
python3 main.py --mode track \
  --video_path "examples/video.mp4" \
  --model_path "models/model.onnx" \
  --only_csv

# 1b) Detection using OpenVINO (optimized for Intel CPU/GPU)
python3 main.py --mode track-ov \
  --video_path "examples/video.mp4" \
  --model_path "ov/model.xml" \
  --device CPU

# 2) Tracks from CSV -> track_*.json
uv run src/track_calculator.py \
  --csv_path "output/video/ball.csv" \
  --output_dir "output"

# 3) Vertical reels from tracks
uv run src/make_reels.py \
  --video_path "examples/video.mp4" \
  --json_dir "output/video/tracks" \
  --output_dir "output"
```

## Output Structure

```text
output/video_name/
├── ball.csv                  # Raw ball coordinates
├── tracks/
│   └── track_0001.json       # Individual rally data
└── reels/
    └── reel_video_0001.mp4   # 9:16 vertical reels
```

## Advanced Features

### Grid-based Models
Grid models (prefixed with `VballNetGrid`) improve tracking accuracy by predicting sub-pixel offsets on a grid coordinate system, reducing quantization errors common in heatmap models.

### Pose-aware Tracking
Integrate player pose information into the tracking JSON for proximity-based rally filtering:
```bash
python3 main.py --mode pose --video_path video.mp4 --track_file tracks/track_0001.json
```
