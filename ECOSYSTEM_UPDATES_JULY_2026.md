# Volleyball Tracking Ecosystem Update - July 2026

## Overview
Comprehensive assessment of the volleyball tracking ecosystem, including high-speed tracking, experimental attention layers, and professional training pipelines.

## 1. fast-volleyball-tracking-inference (Core Pipeline)
**Updates as of June/July 2026:**
- **Vertical Reels (9:16):** Added support for "rotate vertical" flag to generate reels optimized for social media.
- **Analytical Tooling:** Integrated `scripts/analyze_zone4_ball_trajectories.py` for evaluating volleyball attack quality near antennas based on standard court dimensions (18m x 9m).
- **Grayscale Models:** Updated to `ballNetGridV1b_seq9_grayscale_20260510` for improved robustness.
- **OpenVINO 2025+:** Robust support for the latest OpenVINO runtimes with dynamic output selection.

## 2. GridTrackNet (High-Speed Tracking)
**Key Contributions:**
- **Efficiency:** Achieves 116 FPS on M1 Pro hardware at 768x432 resolution.
- **MIMO Approach:** Uses 5 temporal input/output frames for superior context compared to traditional 3-frame sequences.
- **Tools:** Provided a PySide6-based GUI labelling tool and TFRecord generation pipeline.

## 3. vball-net (Experimental Hub)
**Research Focus:**
- **DyT Layer:** Experiments with Dynamic Transformation (DyT) and Dynamic Tanh activation layers to explore lightweight alternatives for improving tracking accuracy.
- **Motion Attention:** Development of `VballNetV2b` with `MotionPromptLayer` for generating motion attention maps via temporal differences.

## 4. TrackNetV4-PyTorch (Professional Training)
**Modernization:**
- **Optimizers:** Support for modern optimizers including Adadelta, Adam, AdamW, and Lion.
- **Motion Attention Maps:** Professionalized documentation and focus on motion-aware tracking to handle occlusion and motion blur.

## Local Implementation Sync
- Synchronized with `fastvball/master` to adopt vertical rotation and analytical features.
- Refactored `src/coort_coordinats.py` to `src/court_coordinates.py` for correctness.
- Enabled `--rotate` feature parity across both ONNX and OpenVINO pipelines.
