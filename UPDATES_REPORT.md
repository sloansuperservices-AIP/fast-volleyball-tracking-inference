# Updates Report - June 14, 2026

## Core Pipeline (fast-volleyball-tracking-inference)
- **Synchronized with upstream/master (eee3c72):**
  - Integrated **seq15** model support for improved temporal context.
  - Added **OpenVINO** inference support (`src/inference_openvino_seq_gray_v2.py`) with optimized model binaries in `ov/`.
  - Implemented **ball radius detection** in ONNX and OpenVINO pipelines.
  - Integrated **Zone 4 ball trajectory analysis** tooling (`scripts/analyze_zone4_ball_trajectories.py`).
  - Added **Celery worker** for task orchestration (`src/celery_worker.py`).
  - Standardized CSV schema: `["Frame", "Visibility", "X", "Y", "Radius"]`.

## GridTrackNet
- **Inference Optimization:**
  - Unified inference workflow with ONNX runtime support.
  - Grid-based output layer (48x27) at 768x432 resolution, achieving high FPS.

## vball-net
- **Model Architecture Experiments:**
  - `VballNetV2b` experiments incorporating **MotionPromptLayer**, **spatial_attention**, and **Dynamic Transformation (DyT)** layers.
  - Accelerations in training repeatability and speed.

## TrackNetV4-PyTorch
- **Training & Metrics:**
  - Focus on center-frame prediction to improve accuracy metrics.
  - Support for `Adadelta`, `AdamW`, and `Lion` optimizers.

## Local Improvements
- **Main Orchestrator:**
  - Unified `main.py` supporting `track`, `pose`, `analyze`, `hub-track`, and `openvino-track` modes.
  - Robust subprocess handling for consistent virtual environment execution.
