# Volleyball Tracking Ecosystem: June 2026 Update Report

This report summarizes recent updates and enhancements from the public volleyball tracking repositories. The local repository is currently behind several major feature releases and optimizations.

## 1. Core Pipeline: `fast-volleyball-tracking-inference`

The primary inference repository has seen significant updates in functional capability and runtime flexibility.

### New Features & Capabilities
*   **Vertical Reel Enhancements:** Added a "rotate vertical" feature and a new watermarking system (`vb-ai.ru`) in `src/make_reels.py`. It now supports concatenating multiple rally segments into a single reel using FFmpeg.
*   **Ball Size Detection:** Integrated automated ball radius detection into the inference loop (`src/inference_onnx_seq_gray_v2.py`). The CSV schema has been updated to `["Frame", "Visibility", "X", "Y", "Radius"]`.
*   **Zone 4 Analytical Tools:** New `scripts/analyze_zone4_ball_trajectories.py` for evaluating attack corridors near antennas, providing colored quality metrics (Excellent, OK, Uncomfortable).
*   **OpenVINO Optimization:** Extensive improvements to the OpenVINO runtime usage, including optimized sequence crop inference for high-resolution sources.

### Model Updates
*   **Grid Models:** Introduction of `VballNetGridV1b` (seq9 and seq15) and `VballNetGridV1c`.
*   **Specialized Models:** New models for **Padel** tracking (`VballNetGridV1b_seq9_grayscale_padel`).
*   **Performance:** Grayscale seq9 models continue to reach ~200 FPS on mid-range CPUs.

## 2. High-Speed Tracking: `GridTrackNet`

*   **Benchmark Performance:** Reached **116 FPS** on M1 Pro hardware at 768x432 resolution.
*   **Unified Workflow:** The repository now features a unified inference workflow and full ONNX runtime support, simplifying deployment across different hardware backends.

## 3. Experimental Research: `vball-net`

*   **Dynamic Transformation (DyT):** Recent experiments focus on replacing or optimizing the DyT layer to improve training repeatability.
*   **Attention Mechanisms:** Integration of spatial attention into the `VballNetV1` architecture to improve precision in noisy backgrounds.
*   **Mixed-Precision & Mixup:** New visualization tools for Mixup data augmentation and training acceleration.

## 4. Modern Training: `TrackNetV4-PyTorch`

*   **Advanced Optimizers:** Full support for **Lion** and **AdamW** optimizers, which often provide better convergence than standard Adadelta.
*   **Motion Attention:** Refactored model architecture to explicitly extract and utilize motion attention maps, focusing on temporal changes between frames.
*   **Professionalization:** The repository has been reorganized for better clarity, with updated documentation and standardized inference scripts.

---

## Suggested Synchronization Strategy

To bring the local repository up to date while preserving existing extensions (like the `hub-track` mode):

1.  **Structural Update:** Adopt the `scripts/` directory from `fastvball` and rename `src/coort_coordinats.py` to `src/court_coordinates.py` for consistency.
2.  **Inference Sync:** Merge the radius detection and OpenVINO runtime improvements into `src/inference_onnx_seq_gray_v2.py`.
3.  **Main Orchestrator:** Update `main.py` to include the `analyze` mode and map it to the new zone 4 trajectory scripts.
4.  **Model Migration:** Sync the `models/` and `ov/` directories to include the latest Grid V1b/V1c weights.
