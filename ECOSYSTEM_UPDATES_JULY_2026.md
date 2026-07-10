# Ecosystem Updates - July 2026

## fast-volleyball-tracking-inference
- **New Feature**: "rotate vertical" support for processing 9:16 reels or rotated camera angles.
- **Model Update**: ballNetGridV1b_seq9_grayscale_20260510 release.
- **Inference**: Unified OpenVINO and ONNX pipelines now support `--rotate` and `--device` arguments.

## GridTrackNet
- **High-speed Tracking**: Architecture reaching 116 FPS on M1 Pro.
- **Tools**: Integrated PySide6-based `LabellingTool.py`, `DataGen.py` for TFRecord creation, and `Train.py`.
- **Inference**: `Predict.py` with batch processing support.

## TrackNetV4-PyTorch
- **New Architecture**: `TrackNetV4` with Motion Attention mechanisms using 3-frame sequences.
- **Performance**: Improved tracking for fast objects with significant motion blur.

## vball-net
- **Research**: Experiments with `VballNetV2b` and DyT (Dynamic Tanh) activation layers.
- **Repeatability**: Optimizations for training repeatability and faster convergence.
