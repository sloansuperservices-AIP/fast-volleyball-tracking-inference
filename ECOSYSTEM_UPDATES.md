# Volleyball Ecosystem Update Report - June 2026

This report summarizes recent updates and releases from the core volleyball tracking repositories and suggests integration steps for the local pipeline.

## 1. fast-volleyball-tracking-inference (`fastvball/master`)

### New Features & Enhancements
- **Ball Radius Detection:** Integrated logic in `inference_onnx_seq_gray_v2.py` that estimates ball size (`Radius`) using contour analysis, useful for depth estimation and filtering.
- **Vertical Video Support:** New `rotate_vertical` flag and processing logic to generate 9:16 reels for social media.
- **Zone 4 Trajectory Tooling:** Added `scripts/analyze_zone4_ball_trajectories.py` to evaluate attack quality near antennas based on court coordinates.
- **Court Transformation:** `track_calculator.py` now supports mapping image coordinates to a 3D/2D court model (18m x 9m) if a `court.json` is provided.
- **Refactored Tracker:** Significant improvements to `TrackCalculator` with better logging, track lifecycle management, and detailed JSON output including trajectory stats.

### Suggested Updates
- **Priority High:** Update `src/track_calculator.py` and `src/inference_onnx_seq_gray_v2.py` to match upstream for radius detection and court support.
- **Priority Medium:** Integrate `scripts/analyze_zone4_ball_trajectories.py` into the `analyze` mode in `main.py`.

---

## 2. GridTrackNet (`gridtracknet/main`)

### Key Innovations
- **Grid-based Prediction:** Uses a confidence grid and X/Y offset grids instead of traditional heatmaps, enabling precise localization even at lower resolutions (768x432).
- **MIMO Architecture:** Processes 5 frames simultaneously (5-in, 5-out), significantly increasing throughput (up to 116 FPS on M1 Pro).
- **Optimized ONNX:** Improved export and inference scripts for high-speed deployment.

### Suggested Updates
- **Priority Low:** Evaluate the `GridTrackNet` model for real-time edge applications where high FPS is critical.

---

## 3. vball-net (`vballnet/main`)

### Experimental Work
- **VballNetV2b:** Features a `MotionPromptLayer` that uses central differences to generate motion attention maps, improving tracking in complex backgrounds.
- **Dynamic Tanh (DyT) Layer:** Experimental learnable activation function (Dynamic Tanh/Softsign) tested to replace `BatchNormalization` and `ReLU` for better training stability and repeatability.
- **VballNetFastV1:** A lightweight U-Net variant optimized for consumer GPUs using depthwise separable convolutions.

### Suggested Updates
- **Priority Medium:** Incorporate `VballNetV2b` architecture into the local training pipeline to benefit from motion-guided attention.

---

## 4. TrackNetV4-PyTorch (`tracknetv4/main`)

### Professionalization
- **Modern Optimizers:** Support for `AdamW` and `Lion` optimizers, which often yield better convergence than standard `Adadelta`.
- **Motion Attention Modules:** Clean PyTorch implementation of the `MotionPrompt` and `MotionFusion` layers described in the TrackNetV4 paper.
- **Robust Training Pipeline:** Improved CLI for dataset preprocessing, training, and evaluation with automated PDF report generation.

### Suggested Updates
- **Priority Medium:** Use this repository as the base for any new model training sessions to leverage better optimizers and cleaner codebase.

---

## Conclusion

The ecosystem has moved towards **motion-aware architectures** (TrackNetV4, VballNetV2b) and **analytical tooling** (Zone 4 analysis, Court mapping). The local repository should prioritize synchronizing with `fast-volleyball-tracking-inference` to adopt the latest analytical capabilities.
