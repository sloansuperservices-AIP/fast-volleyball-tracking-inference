# Updates Report - Fast Volleyball Tracking Inference

This report summarizes the latest updates and contributions integrated from the tracked public repositories.

## Integrated Features (as of May 2025)

### 1. Advanced Model Support
- **seq15 Support**: Integrated temporal context handling for 15-frame sequence models (`seq15`), improving tracking stability for high-speed movements.
- **Grid-based Inference**: Support for GridTrackNet-style grid output decoding, enabling faster inference (~116 FPS) compared to traditional heatmap-based models.
- **Dynamic Parameter Inference**: Automatic detection of model properties (sequence length, output type) from ONNX model filenames.

### 2. Detection Enhancements
- **Ball Radius Estimation**: Integrated radius detection using `cv2.minEnclosingCircle` on heatmap contours, providing scale and depth information.
- **Headless Environment Detection**: Automatic disabling of OpenCV visualization in Docker/Server environments using `is_headless()` checks.

### 3. Pipeline Optimizations
- **Performance Tracking**: Optimized `TrackCalculator` using `df.groupby('Frame')` and `itertuples()`, reducing track processing time from $O(N^2)$ to $O(N)$.
- **Advanced Smoothing**: Implemented Savitzky-Golay filtering in `make_reels.py` for smoother 9:16 vertical video cropping.
- **Unified CSV Schema**: Standardized detection CSV output to include `Frame`, `Visibility`, `X`, `Y`, and `Radius`.

### 4. Codebase Modernization
- **Centralized Configuration**: Introduced `src/constants.py` and `src/models.py` for better code organization.
- **Dependency Management**: Standardized on `uv` for lightning-fast environment synchronization.

## Repository Contributions

- **GridTrackNet**: Architecture improvements for high-speed tracking (768x432 grid-based).
- **fast-volleyball-tracking-inference**: Core pipeline enhancements, OpenVINO support, and vertical reel automation.
- **vball-net**: New model experiments with `MotionPromptLayer` and depthwise separable convolutions.
- **TrackNetV4-PyTorch**: Optimizer support (`AdamW`, `Lion`) and center-frame prediction focus.
