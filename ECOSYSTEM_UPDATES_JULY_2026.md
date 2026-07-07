# Volleyball Tracking Ecosystem Updates - July 2026

This document summarizes the latest updates and contributions from the tracked repositories in the volleyball tracking ecosystem as of July 2026.

## 1. fast-volleyball-tracking-inference
**Core pipeline for inference, tracking, and reel generation.**

### Key Updates:
- **Vertical Reels Generation**: Integrated `src/make_reels.py` to automatically generate 9:16 vertical reels centered on the ball trajectory, optimized for social media.
- **Rotation Support**: Added `--rotate` argument (-90, 90, 180) to OpenVINO and ONNX inference scripts to handle vertical video recordings.
- **Model Updates**: Synchronized with `ballNetGridV1b_seq9_grayscale_20260510` models for improved grid-based tracking.
- **Analysis Tooling**: Added `scripts/analyze_zone4_ball_trajectories.py` for evaluating attack quality near the antennas.
- **Improved Orchestration**: `main.py` now unified with `analyze`, `openvino-track`, and `hub-track` modes.

## 2. GridTrackNet
**High-speed tracking architecture using Multiple-Input Multiple-Output (MIMO).**

### Key Updates:
- **Unified Inference Workflow**: Added support for ONNX runtime, improving cross-platform compatibility.
- **Labelling Tool**: Professionalized `LabellingTool.py` (PySide6-based) for high-precision dataset annotation.
- **Data Pipeline**: Integrated `DataGen.py` for generating TFRecord datasets from raw video and CSV annotations.
- **Architecture**: Employs a 5-frame temporal context to output 768x432 confidence and offset grids.

## 3. vball-net
**Experimental hub for model research and training optimizations.**

### Key Updates:
- **Dynamic Transformation (DyT) Layers**: Extensive experiments with `softsign` and `DyT` (Dynamic Tanh) activation layers to replace standard activations for better accuracy/speed trade-offs.
- **VballNetV2b**: Introduction of the V2b architecture featuring `MotionPromptLayer` for generating motion attention maps.
- **Training Repeatability**: Improved training scripts to ensure better convergence and repeatability across different runs.

## 4. TrackNetV4-PyTorch
**Professional training pipeline and advanced TrackNet architectures.**

### Key Updates:
- **TrackNetV4 Architecture**: Enhanced with Motion Attention modules to focus on moving objects (shuttlecock/ball).
- **Modern Optimizers**: Integrated support for `Lion` and `AdamW` optimizers via a factory-based creation method.
- **Advanced Loss Functions**: Implementation of Weighted Binary Cross Entropy (WBCE) loss strictly following the original paper formulation.
- **Evaluation Suite**: Comprehensive `test.py` script generating PDF reports, confusion matrices, and error distribution visualizations.

---
*Date: July 7, 2026*
