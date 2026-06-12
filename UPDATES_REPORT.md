# Ecosystem Updates Report: Volleyball Tracking (June 2026)

This report summarizes recent updates, releases, and innovations from across the core volleyball tracking repositories. It provides actionable suggestions for integrating these advancements into the local pipeline.

---

## 1. fast-volleyball-tracking-inference (Core Pipeline)
**Target:** Synchronize with `upstream/master` (v0.0.1 and beyond).

### Key Updates:
*   **Release v0.0.1**: Official release featuring standardized ONNX/OpenVINO inference, benchmarking, and the first "Grid" models.
*   **Temporal Context (seq15)**: Added support for 15-frame sequence models (`seq15`), providing significantly better temporal continuity for fast-moving balls.
*   **Ball Size/Radius Detection**: The inference pipeline now estimates ball radius using `cv2.minEnclosingCircle` on heatmap contours. This is exported to `ball.csv` and used in tracking.
*   **Zone-4 Trajectory Analysis**: A new analytical tool (`scripts/analyze_zone4_ball_trajectories.py`) allows for specialized trajectory analysis in the Zone-4 area.
*   **Infrastructure & Scaling**:
    *   **Celery Integration**: `src/celery_worker.py` allows for distributed task processing.
    *   **Docker Support**: Full Dockerization for OpenVINO and ONNX runtimes.
*   **Unified CLI**: Standardized 4-step processing (Detection -> Calculation -> Processing -> Reels) with unified parameters.

### Integration Suggestions:
1.  **Adopt Radius Estimation**: Update `src/inference_onnx_seq_gray_v2.py` and `src/track_calculator.py` to handle the `Radius` column.
2.  **Integrate OpenVINO**: Port `src/inference_openvino_seq_gray_v2.py` for CPU-optimized inference (achieving ~200 FPS).
3.  **Deploy Zone-4 Tooling**: Add `scripts/analyze_zone4_ball_trajectories.py` to the local repository for advanced user analytics.

---

## 2. GridTrackNet (High-Speed Tennis/Ball Tracking)
**Target:** High-performance alternative architecture.

### Key Innovations:
*   **Efficiency**: Reaches **116 FPS** on M1 Pro via a redesigned architecture.
*   **Input Resolution**: Increased from 512x288 to **768x432**, improving detection of small objects.
*   **Grid Output**: Uses a 48x27 grid output instead of full-resolution heatmaps, significantly reducing decoding overhead.
*   **Sequence 5**: Optimized for 5 concurrent frames.

### Integration Suggestions:
1.  **Benchmarking**: Test the Grid-based ONNX models within the local pipeline to compare accuracy vs. speed against `VballNet`.

---

## 3. vball-net (Training Experiments)
**Target:** Cutting-edge model architectures.

### Key Innovations:
*   **VballNet v2b**: Latest experiments incorporating `MotionPromptLayer` and `spatial_attention`.
*   **Dynamic Transformation (DyT) Layer**: Implementation of DyT layers to better capture non-linear ball trajectories.
*   **Training Reliability**: Improved repeatability and acceleration in training notebooks (`vball_net_train.ipynb`).

### Integration Suggestions:
1.  **Architecture Upgrade**: Evaluate the `DyT` and `spatial_attention` layers for the next generation of local models to improve performance in occluded scenarios.

---

## 4. TrackNetV4-PyTorch (Training Framework)
**Target:** Optimization and Evaluation strategies.

### Key Innovations:
*   **Advanced Optimizers**: Support for `AdamW` and `Lion` (via `lion-pytorch`), which often converge faster than standard Adam.
*   **Center-Frame Focus**: Refined evaluation strategy focusing on center-frame predictions to improve metric accuracy.
*   **Simplified Requirements**: Transition to `uv` for dependency management and cleaner project structure.

### Integration Suggestions:
1.  **Evaluation Sync**: Adopt the "center-frame prediction" evaluation logic in local test scripts to ensure performance metrics are comparable with community standards.
2.  **Lion Optimizer**: Incorporate `Lion` into training workflows for potentially better model generalization.

---

## Summary of Actionable Items
| Feature | Source | Priority |
| :--- | :--- | :--- |
| Ball Radius Detection | upstream | **High** |
| OpenVINO Inference | upstream | **High** |
| Zone-4 Analysis Tool | upstream | Medium |
| seq15 Support | upstream | Medium |
| DyT/Attention Layers | vball-net | Low (R&D) |
| Lion Optimizer | tracknetv4 | Low (R&D) |

*Report generated on June 12, 2026.*
