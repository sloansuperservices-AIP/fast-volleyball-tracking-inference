# Update Assessment and Recommendations Report (June 2026)

This report summarizes the latest updates, releases, and contributions from the four primary volleyball tracking repositories and provides recommendations for synchronizing the local codebase.

---

## 1. Repository Analysis

### [fast-volleyball-tracking-inference](https://github.com/asigatchov/fast-volleyball-tracking-inference)
- **Latest Release:** `v0.0.1` (March 20, 2026).
  - Introduced unified ONNX/OpenVINO inference pipelines.
  - Added support for **Grid-based models**.
  - Speed and accuracy benchmark report included in README.
- **New Contributions (May 2026):**
  - **`seq15` Support:** Integrated support for models with a 15-frame temporal window for better tracking stability.
  - **Ball Radius Detection:** New logic in `inference_onnx_seq_gray_v2.py` (`estimate_ball_radius`) using motion masks and contour analysis to track visual scale.
  - **Zone-4 Analytical Tools:** New analytical scripts (`analyze_zone4_ball_trajectories.py`) for evaluating specific court areas.
  - **Performance Optimization:** Refactored `track_calculator.py` to use `df.groupby().itertuples()`, improving speed significantly.

### [GridTrackNet](https://github.com/asigatchov/GridTrackNet)
- **High-Speed Architecture:** Reaches **116 FPS** on M1 Pro.
- **Key Feature:** Uses a 768x432 input resolution and **grid outputs** (48x27) instead of full-size heatmaps, drastically reducing computational overhead.
- **Context:** Processes 5 concurrent frames for both input and output.

### [vball-net](https://github.com/asigatchov/vball-net)
- **Research Frontiers:**
  - **Dynamic Transformation (DyT) Layers:** Experimental integration to enhance spatial-temporal feature extraction.
  - **Spatial Attention:** New model variants (e.g., `v2b`) incorporating attention mechanisms to focus on relevant ball-like regions.
  - **In-painting Experiments:** Research into handling occlusions via specialized net architectures.

### [TrackNetV4-PyTorch](https://github.com/asigatchov/TrackNetV4-PyTorch)
- **Motion Attention Maps:** Focuses on temporal changes between consecutive frames to improve tracking of small, fast objects.
- **Architecture:** 9-channel input (3 consecutive RGB frames) and a Motion Prompt Layer.
- **Optimization:** Support for `Adadelta`, `AdamW`, and `Lion` optimizers for more robust training.

---

## 2. Gap Analysis (Local vs. Upstream)

| Feature | Local State | Upstream/Research State |
| :--- | :--- | :--- |
| **Ball Radius** | Class field exists but not actively estimated | Fully implemented in ONNX/OpenVINO scripts |
| **Grid Support** | Missing | Fully supported in v0.0.1+ |
| **`seq15` Models** | Incomplete batch/GRU logic | Fully integrated |
| **Track Performance** | $O(N^2)$ iteration | $O(N)$ optimized processing |
| **Analytical Tools** | Standard reels/tracks | Zone-4 corridors and trajectory analysis |
| **Model Layers** | Standard CNN/GRU | DyT, Spatial Attention, Motion Prompts |

---

## 3. Recommended Roadmap

### Phase 1: Immediate Sync (Critical)
1.  **Integrate Radius Estimation:** Port `estimate_ball_radius` and its dependencies from upstream to `src/inference_onnx_seq_gray_v2.py`.
2.  **Add Grid Support:** Port `postprocess_grid_output` and `infer_model_params` updates to support the `GridTrackNet` model family.
3.  **Optimize Track Calculation:** Update `src/track_calculator.py` with the `itertuples` loop to handle large datasets efficiently.

### Phase 2: Functional Extension (Tools)
1.  **Port Analytical Scripts:** Add `scripts/analyze_zone4_ball_trajectories.py` and `scripts/inference_openvino_seq_gray_crop.py` to the local `src/` or a new `scripts/` directory.
2.  **Standardize `seq15`:** Update the inference batching logic to fully support 15-frame temporal windows.

### Phase 3: Research Integration (Training)
1.  **DyT/Attention Layers:** For the next model training iteration, evaluate the **DyT** and **Spatial Attention** layers from `vball-net` to improve detection in complex scenes.
2.  **Optimizer Migration:** Update training pipelines to use `AdamW` or `Lion` as implemented in `TrackNetV4`.

---
*Report generated on June 1, 2026.*
