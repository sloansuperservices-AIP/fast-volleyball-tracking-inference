# Volleyball Tracking Ecosystem Updates Report (June 2026)

This report summarizes recent updates and recommended integrations from the core repositories in the volleyball tracking ecosystem.

## 1. fast-volleyball-tracking-inference (Upstream)

The core inference engine has seen significant functional expansions in the `master` branch and `v0.0.1` release.

### Key Updates:
*   **Ball Radius Estimation:** Integrated into the ONNX and OpenVINO pipelines. It uses `cv2.minEnclosingCircle` on heatmap contours to estimate ball size, providing better depth/scale context.
*   **OpenVINO Support:** New `src/inference_openvino_seq_gray_v2.py` script and optimized `.xml` models enable much higher performance on Intel hardware (CPU/iGPU).
*   **Analytical Tools:** Added `scripts/analyze_zone4_ball_trajectories.py` for specialized analysis of ball paths in Zone 4.
*   **Pipeline Scalability:** Celery worker integration (`src/celery_worker.py`) for handling large-scale video processing tasks.
*   **CSV Schema Standard:** Unified schema across all inference modules: `["Frame", "Visibility", "X", "Y", "Radius"]`.

### Suggested Updates:
*   Merge `upstream/master` to adopt radius detection and analytical scripts.
*   Integrate `openvino-track` and `analyze` modes into the unified `main.py` entry point.

---

## 2. GridTrackNet

A redesigned architecture focused on ultra-efficiency and higher resolution tracking.

### Key Updates:
*   **Resolution Boost:** Increased input resolution from 512x288 to **768x432**, significantly improving detection of small balls at a distance.
*   **Enhanced Temporal Context:** Uses 5 input/output frames (vs 3 in standard TrackNet) for better continuity.
*   **Performance:** Reaches **116 FPS** on M1 Pro.
*   **Grid Output Layer:** Uses a 48x27 grid-based output instead of full-resolution heatmaps, reducing computational overhead.

### Suggested Updates:
*   Evaluate the grid-based architecture for high-resolution tracking requirements.
*   Adopt the 5-frame temporal context for improved robustness against momentary occlusions.

---

## 3. vball-net

The primary research and training repository for VballNet models.

### Key Updates:
*   **Dynamic Transformation (DyT):** Experiments with DyT layers for better spatial feature extraction.
*   **MotionPromptLayer:** Tailored attention mechanism that focuses on temporal changes between consecutive frames.
*   **Depthwise Separable Convolutions:** Optimized for consumer-grade hardware, balancing accuracy and inference speed.

### Suggested Updates:
*   Train new models using the DyT and MotionPromptLayer enhancements to improve recall in noisy environments.

---

## 4. TrackNetV4-PyTorch

A robust PyTorch implementation focusing on motion-aware tracking.

### Key Updates:
*   **Motion Attention Mechanisms:** Specialized layers to focus on regions with significant temporal changes.
*   **Improved Occlusion Handling:** Architectural focus on maintaining tracks through motion blur and partial occlusions.

### Suggested Updates:
*   Reference the PyTorch implementation for cross-validation of model logic and attention map generation.

---

## Conclusion & Implementation Plan

The immediate priority is synchronizing with `fast-volleyball-tracking-inference` to gain ball radius estimation and OpenVINO support. Future work should explore the 768x432 resolution from `GridTrackNet` and the DyT architecture from `vball-net`.
