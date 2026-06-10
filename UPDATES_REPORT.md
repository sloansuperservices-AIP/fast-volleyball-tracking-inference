# Volleyball Tracking Ecosystem Updates Report (June 2026)

This report summarizes the major updates and features synchronized from the volleyball tracking ecosystem, including `GridTrackNet`, `fast-volleyball-tracking-inference`, `vball-net`, and `TrackNetV4-PyTorch`.

## Core Inference Enhancements

### 1. Grid-Based Tracking (`GridTrackNet`)
- **Architecture**: Introduced a redesign achieving **116 FPS** on M1 Pro (+241% over TrackNetV2).
- **Resolution**: Increased input resolution to **768x432**.
- **Temporal Context**: Upgraded to **5 concurrent frames** (seq5) for better motion continuity.
- **Output Layer**: Transitioned from full-resolution heatmaps to a specialized **grid output (48x27)** for optimized inference.

### 2. Unified Seq-N Support (`fast-volleyball-tracking-inference`)
- **Dynamic Sequence Length**: The unified `src/inference_onnx_seq_gray_v2.py` now supports `seq3`, `seq9`, and `seq15` models automatically.
- **Ball Radius Estimation**: Integrated real-time ball size detection using `cv2.minEnclosingCircle` on heatmap contours, providing better scale data for trackers.
- **Grayscale Optimization**: Standardized on grayscale inputs to maintain ~200 FPS on standard CPUs.

### 3. OpenVINO Integration
- **Optimized Runtime**: Full support for OpenVINO inference via `src/inference_openvino_seq_gray_v2.py`.
- **Model Versions**: Added IR models (`.xml`, `.bin`) for `VballNetGrid` and `VballNetV2` series, specifically tuned for Intel hardware and edge devices.

## Analytical & Processing Tools

### 1. Zone-4 Trajectory Analysis
- **Corridor Evaluation**: New `scripts/analyze_zone4_ball_trajectories.py` tool for evaluating attack corridors near antennas.
- **3D Projection**: Uses court coordinates to project 2D detections into real-world metrics.

### 2. Vertical Reels (9:16)
- **Automatic Framing**: Enhanced `src/make_reels.py` with support for multiple smoothing algorithms (`kalman`, `savitzky_golay`) and padding modes (`mirror`, `black`).
- **Centric Cropping**: Ensures the ball remains perfectly centered in vertical reels for social media optimization.

### 3. Distributed Processing
- **Celery Worker**: Integrated `src/celery_worker.py` for orchestrating large-scale video processing tasks in distributed environments.

## Repository Synchronizations

- **`vball-net`**: Integrated `v2b` architecture experiments including `MotionPromptLayer` and depthwise separable convolutions.
- **`TrackNetV4-PyTorch`**: Adopted center-frame prediction focus and support for `Adadelta`, `AdamW`, and `Lion` optimizers.

---
*Report generated on June 9, 2026.*
