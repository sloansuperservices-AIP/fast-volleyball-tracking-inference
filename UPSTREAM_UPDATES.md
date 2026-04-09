# Upstream Update Report (March 2026)

I have synchronized the codebase with the latest releases and contributions from the volleyball tracking ecosystem. Below is a summary of the major updates incorporated.

## 1. Fast Volleyball Tracking Inference (v0.0.1)
*   **High-Performance Inference**: Integrated `src/inference_openvino_seq_gray_v2.py`, providing up to 270 FPS on Intel hardware.
*   **Distributed Processing**: Added `src/celery_worker.py` for asynchronous video processing and reel generation using Redis.
*   **Unified Pipeline**: Standardized the 4-step tracking workflow:
    1. `detect`: Ball coordinates -> CSV.
    2. `track`: CSV -> Rally Tracks (JSON).
    3. `combined`: Rally Tracks -> Horizontal Video.
    4. `reels`: Rally Tracks -> Vertical 9:16 Reels.
*   **Grid Model Support**: Added decoding logic for the new high-precision "Grid" models.

## 2. GridTrackNet (V2 Architecture)
*   **Ultra-Efficiency**: Reaches 116 FPS on standard CPU hardware.
*   **Increased Resolution**: Upgraded from 512x288 to 768x432.
*   **Enhanced Context**: Now utilizes 5 input/output frames (up from 3) for improved tracking of fast-moving objects.
*   **Grid Output**: Transitioned to a 48x27 output grid for sub-pixel accuracy, significantly reducing coordinate localization errors.

## 3. vball-net
*   **Motion Attention**: Implementation of the `MotionPromptLayer` which focuses on frame differences to enhance temporal context in grayscale sequences.
*   **Optimized Training**: New training recipes for sequence-9 grayscale models, aligning with the latest inference speed requirements.

## 4. TrackNetV4-PyTorch
*   **PyTorch Implementation**: A robust PyTorch port of the motion-attention tracking system, providing an alternative to the TensorFlow-based models for research and deployment flexibility.

## Codebase Synchronization Summary
- **Dependencies**: Updated `pyproject.toml` with `openvino`, `celery[redis]`, `ultralytics`, and `ffmpeg-python`.
- **Scripts**: Synchronized `inference_onnx_seq_gray_v2.py`, `make_reels.py`, `track_calculator.py`, and added `celery_worker.py`.
- **Models**: Added `src/models/GridTrackNet.py` architectural definition.
