# Volleyball Tracking Ecosystem Updates Report

This report summarizes recent updates and new contributions integrated into the Fast Volleyball Tracking pipeline from key public repositories.

## 1. fast-volleyball-tracking-inference (Core Pipeline)
**Current Status:** Synchronized with latest upstream `master` (v0.0.1+).

### Major Updates:
- **Ball Radius Estimation:** Integrated real-time ball size detection using `cv2.minEnclosingCircle` on heatmap contours, providing physical scale information.
- **OpenVINO Support:** Added `src/inference_openvino_seq_gray_v2.py` for high-performance inference on Intel hardware, achieving superior FPS on compatible CPUs/iGPUs.
- **Sequence Length Support:** Unified inference scripts now support `seq9` and `seq15` models, automatically detecting sequence length from model metadata.
- **Zone-4 Trajectory Analysis:** New analytical tools in `scripts/` to analyze ball trajectories specifically in the high-stakes Zone 4 of the volleyball court.
- **Celery Orchestration:** Initial support for distributed processing using Celery workers for large-scale video analysis.

## 2. GridTrackNet
**Research Focus:** Real-time high-resolution tracking.

### Key Features:
- **Efficiency:** Reaches up to **116 FPS** on M1 Pro by using a grid-based output layer (48x27) instead of full-resolution heatmaps.
- **Resolution:** Increased input resolution to **768x432**, allowing for better detection of small, fast-moving balls.
- **Temporal Context:** Processes **5 concurrent frames** to maintain tracking stability.

## 3. vball-net
**Research Focus:** Advanced model architectures.

### Innovations:
- **Dynamic Transformation (DyT) Layers:** Experiments with DyT and spatial attention to improve temporal feature extraction.
- **MotionPromptLayer:** Incorporates motion attention maps to focus the model on moving objects, reducing false positives from static backgrounds.
- **Lightweight Models:** Optimization of `VballNetFastV1` for consumer-grade hardware using depthwise separable convolutions.

## 4. TrackNetV4-PyTorch
**Research Focus:** PyTorch-based training and optimization.

### Technical Enhancements:
- **Center-Frame Prediction:** Improved metric accuracy by focusing prediction on the center frame of the input sequence.
- **Optimizer Support:** Added support for modern optimizers including **Adadelta**, **AdamW**, and **Lion** for faster and more stable training convergence.
- **Pipeline Modernization:** Full PyTorch implementation of the TrackNet series with end-to-end training and evaluation scripts.

---
*Report generated after consolidating updates from the asigatchov volleyball tracking ecosystem.*
