# Volleyball Tracking Ecosystem Updates - July 2026

This document summarizes the latest updates from the tracked public repositories and provides recommendations for synchronizing the local codebase with the latest advancements in the ecosystem.

## Repository Updates

### 1. fast-volleyball-tracking-inference
*   **New Feature: Vertical Rotation** (Commit `637c217`, Jun 15, 2026)
    *   Introduced support for rotating frames (90, -90, 180 degrees) during inference.
    *   Essential for processing 9:16 reels and vertical video content.
*   **Model Update: Grayscale V1b** (Commit `eee3c72`, Jun 14, 2026)
    *   Updated to `ballNetGridV1b_seq9_grayscale_20260510`.
    *   Improved tracking stability in grayscale sequence models.
*   **Inference Refinement**: Improved dynamic library path handling for NVIDIA CUDA libraries, moving away from hardcoded paths in the virtual environment.

### 2. GridTrackNet
*   **High-Speed MIMO Architecture**: Demonstrates ultra-efficient tracking reaching 116 FPS on M1 Pro hardware.
*   **Temporal Context**: Uses 5-frame temporal windows for both input and output (MIMO approach).
*   **Recent Commits** (Mar 2026): Unified inference workflow and added robust ONNX runtime support.

### 3. vball-net
*   **Experimental Research**: Focused on Dynamic Transformation (DyT) layers and MotionPromptLayer.
*   **Optimized Architectures**: VballNetV1 (U-Net with motion attention) and VballNetFastV1 (lightweight depthwise separable convolutions).
*   **Training Repeatability**: Recent experiments (Aug 2025) aim to improve training consistency and accelerate convergence.

### 4. TrackNetV4-PyTorch
*   **Motion Attention Maps**: Focuses on temporal differences between consecutive frames to track fast-moving objects.
*   **Modern Optimizers**: Full support for Lion and AdamW optimizers in the training pipeline.
*   **Professional Tooling**: Comprehensive evaluation reports (PDF generation) and standardized data preprocessing scripts.

---

## Specific Recommendations for Local Codebase

### Code Updates
1.  **Feature Parity for ONNX Pipeline**:
    *   Integrate the `--rotate` flag into `src/inference_onnx_seq_gray_v2.py`.
    *   Implement `rotate_frame` and `rotated_dimensions` helpers to match the OpenVINO implementation.
2.  **Orchestrator Synchronization**:
    *   Update `main.py` to accept the `--rotate` argument and pass it through to the underlying inference scripts.
3.  **Dynamic Library Loading**:
    *   Replace hardcoded `LD_LIBRARY_PATH` logic in `src/inference_onnx_seq_gray_v2.py` with a dynamic locator using the `site` module to find `nvidia-cublas-cu12` or similar packages.

### Model Upgrades
*   **Sync Latest Models**: Transition to the `20260510` series of grayscale models for both Grid and Heatmap families to leverage improved precision.

### Future Explorations
*   **Architectural Optimization**: Evaluate the feasibility of adopting the 5-frame MIMO approach from `GridTrackNet` to further increase throughput on edge devices.
*   **DyT Layers**: Monitor the `vball-net` experiments for stable releases of Dynamic Tanh activation layers, which could offer lightweight accuracy gains.
