# Suggested Updates for Volleyball Tracking System

Based on an analysis of the latest releases and contributions from the following repositories:
- https://github.com/asigatchov/GridTrackNet.git
- https://github.com/asigatchov/fast-volleyball-tracking-inference
- https://github.com/asigatchov/vball-net.git
- https://github.com/asigatchov/TrackNetV4-PyTorch.git

The following updates are highly recommended to improve performance, accuracy, and maintainability.

## 1. Sync with `fast-volleyball-tracking-inference` v0.0.1
The local repository is currently missing several key architectural improvements introduced in the recent v0.0.1 release (March 20, 2026).

### **Core Infrastructure Updates**
- **Add Foundational Modules**: Introduce `src/constants.py` and `src/models.py`. These files centralize project-wide parameters (e.g., `DEFAULT_INPUT_WIDTH`, `COURT_LENGTH_M`) and domain models (`BallTrack`, `BallDetection`), significantly improving code readability and reducing hardcoded values.
- **Implement OpenVINO Inference**: Add `src/inference_openvino_seq_gray_v2.py`. This enables optimized execution on Intel hardware, achieving superior performance compared to standard ONNX runtime.

### **Logic Improvements**
- **Grid-Based Decoding**: Update `src/inference_onnx_seq_gray_v2.py` to support grid-based models. These models predict sub-pixel offsets on a grid, offering higher precision than traditional heatmap-only models.
- **Advanced Rally Filtering**: Update `src/track_calculator.py` with court-aware filtering logic. This uses perspective transforms to map ball coordinates to real-world court positions, allowing the system to filter out "trash" tracks that occur outside the field of play or below the net level.
- **Enhanced Reel Generation**: Update `src/make_reels.py` to include:
    - **Watermarking**: Professional branding for generated reels.
    - **Improved Interpolation**: Better handling of missing frames using 'hold' or 'linear' interpolation.
    - **Trajectory Smoothing**: Support for Savitzky-Golay and Kalman filters to reduce "jitter" in ball-centered crops.

## 2. Incorporate Advancements from `GridTrackNet`
- **Architecture Refinement**: The redesigned GridTrackNet architecture uses 5 input/output frames (instead of 3 or 9), achieving a better balance between temporal context and inference speed (up to 116 FPS on mobile CPUs).
- **Sub-pixel Accuracy**: The grid/offset approach from this repository is the state-of-the-art for small object tracking in this ecosystem.

## 3. Leverage Experimental Features from `vball-net`
- **Spatial Attention**: Recent experiments in `VballNetV2b` introduce spatial attention mechanisms that help the model focus on relevant court areas, potentially reducing false positives from background movement.
- **Inpainting Support**: The `InpaintNet` modules suggest future directions for handling occlusions where the ball is hidden behind players or the net.

## 4. Performance Benchmarks (Reference)
The latest models achieve significantly better Accuracy@5px:
| Model Family | Accuracy (Visible) | FPS (CPU) |
|--------------|--------------------|-----------|
| VballNetV1   | ~86.4%             | ~140      |
| VballNetGrid | ~74.0%             | ~117      |
| VballNetFast | ~68.9%             | ~270      |

## Recommended Action Plan
1. **Pull Latest Changes**: Update the local codebase to match the v0.0.1 release of the inference repo.
2. **Download Grid Models**: Obtain `VballNetGridV1b` or `VballNetGridV1c` models to leverage the new decoding logic.
3. **Configure Court Mappings**: Use the improved `track_calculator.py` with a valid `court.json` to enable professional-grade rally extraction.
