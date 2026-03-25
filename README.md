# Fast Volleyball Ball Tracking → Vertical Reels

A complete high-speed pipeline for volleyball ball detection, tracking, and automatic generation of **9:16 vertical reels** with the ball always centered.
Achieves **~200 FPS** on a regular CPU (Intel i5-10400F) thanks to a lightweight grayscale seq-9 ONNX model.

## Features (fully implemented)

1. Ball detection → `ball.csv`  
2. Track calculation → separate `track_*.json` files  
3. Assembly of all rallies into one horizontal video or individual clips  
4. Creation of **vertical 9:16 reels** with smooth ball-centered cropping (main output)


<table>
   <tr><td>
<img alt="backline" weight="512" src="https://raw.githubusercontent.com/asigatchov/fast-volleyball-tracking-inference/refs/heads/master/examples/output.gif">
   </td><td>
      <img weight="512" src="https://raw.githubusercontent.com/asigatchov/fast-volleyball-tracking-inference/refs/heads/master/examples/sideline.gif" alt="sideline">
</td></tr>
</table>


<video src="https://github.com/asigatchov/fast-volleyball-tracking-inference/raw/refs/heads/master/examples/reel_g.mp4" controls width="100%"></video>
[Examples - reel](https://github.com/asigatchov/fast-volleyball-tracking-inference/raw/refs/heads/master/examples/reel_g.mp4)


## Model Comparison (Acc@5px)
| Model | Family | F1 | Precision | Recall | Accuracy |
|-------|--------|----|-----------|--------|----------|
| VballNetGridV1c_seq9_grayscale_20260317 | Grid | 0.882 | 0.885 | 0.879 | 0.812 |
| VballNetGridV1b_seq9_grayscale_20260319 | Grid | 0.878 | 0.880 | 0.876 | 0.805 |
| VballNetV1b_seq9_grayscale_best.onnx | Heatmap | 0.855 | 0.818 | 0.896 | 0.767 |
| VballNetV1c_seq9_grayscale_best.onnx | Heatmap | 0.847 | 0.793 | 0.908 | 0.754 |
| VballNetFastV1_seq9_grayscale_233_h288_w512.onnx | Heatmap | 0.772 | 0.832 | 0.720 | 0.662 |

*Note: Grid-based models provide superior sub-pixel accuracy compared to traditional heatmap models.*

## Installation

```bash
# Clone the repository
git clone https://github.com/asigatchov/fast-volleyball-tracking-inference.git
cd fast-volleyball-tracking-inference

# Install dependencies (uv is recommended)
uv sync
# or with pip: pip install -r requirements.txt
```

## Standardized 4-Step Pipeline

```bash
VIDEO="examples/beach_volleyball.mp4"
MODEL="models/VballNetGridV1c_seq9_grayscale_20260317.onnx"
OUT="output"

# 1. Ball detection (produces ball.csv)
# Use 'track' for ONNX or 'track-ov' for OpenVINO (requires .xml)
uv run main.py --mode track --video_path $VIDEO --model_path $MODEL --output_dir $OUT --only_csv

# 2. Track calculation (produces tracks/*.json)
uv run src/track_calculator.py --csv_path $OUT/beach_volleyball/ball.csv --output_dir $OUT

# 3. Rally assembly (produces combined.mp4)
uv run src/track_processor.py --video_path $VIDEO --output_dir $OUT

# 4. Vertical Reel generation (produces reels/*.mp4)
uv run src/make_reels.py --video_path $VIDEO --json_dir $OUT/beach_volleyball/tracks --output_dir $OUT
```

## Advanced Usage

### High-Performance OpenVINO Tracking
```bash
uv run main.py --mode track-ov \
  --video_path video.mp4 \
  --model_xml ov/model.xml \
  --output_dir output
```

### Detection with Real-time Preview
```bash
uv run main.py --mode track \
  --video_path video.mp4 \
  --visualize
```

### Ultralytics Hub Integration
```bash
uv run main.py --mode hub-track \
  --video_path video.mp4 \
  --hub_model https://hub.ultralytics.com/models/YOUR_MODEL_ID \
  --api_key YOUR_API_KEY
```

## Output Structure
```text
output/beach_volleyball/
├── ball.csv                  # raw ball coordinates
├── tracks/
│   └── track_0001.json       # one JSON per rally
├── combined.mp4              # all rallies concatenated (optional)
└── reels/
    └── reel_beach_volleyball_0001.mp4   # vertical 9:16 reels
```
