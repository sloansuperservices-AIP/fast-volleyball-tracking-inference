# Suggested Updates Report

This report summarizes the recommended updates and new contributions from the public volleyball tracking ecosystem to enhance the local codebase.

## 1. fast-volleyball-tracking-inference
**Current Status:** Lagging behind upstream `master` (commit `81d16f3`).

### Recommended Updates:
- **Ball Radius Estimation:** Integrate logic to calculate and store ball radius in `ball.csv` and `track_*.json`. This is crucial for distance-to-camera estimation and scale-invariant tracking.
- **Unified Inference Scripts:** Implement `src/inference_openvino_seq_gray_v2.py` to match the ONNX unified version, supporting both heatmap and grid model families.
- **Performance Optimization:** Refactor the `_process_detections` loop in `src/track_calculator.py` from $O(N^2)$ to $O(N)$ using `groupby('Frame')` and `itertuples()` as seen in upstream.
- **Analytical Tools:** Add `scripts/analyze_zone4_ball_trajectories.py` for automated volleyball-specific analysis (e.g., attack corridors, antenna proximity).

## 2. GridTrackNet
**New Contributions:**
- **Grid-Based Output:** Transition from heatmap-based to grid-based output (48x27) for significant speedups.
- **High Resolution:** Increase input resolution to 768x432 while maintaining >116 FPS performance.
- **Temporal Context:** Support `seq15` models (5 concurrent frames in/out) for more robust tracking of high-speed movements.

## 3. vball-net
**Architectural Advancements:**
- **VballNetV2b Support:** Implement `MotionPromptLayer` and `Dynamic Transformation (DyT)` layers in training and inference pipelines.
- **Spatial Attention:** Integrate `spatial_attention` mechanisms to improve ball localization in cluttered backgrounds.

## 4. TrackNetV4-PyTorch
**Optimization & Metrics:**
- **Advanced Optimizers:** Adopt `Lion` and `AdamW` optimizers for faster convergence and better generalization during model training.
- **Center-Frame Focus:** Align evaluation metrics with the center-frame prediction approach to improve F1 scores and precision.

## Synchronization Plan
1. **Upstream Sync:** Run `git merge upstream/master --allow-unrelated-histories` to bring in core logic improvements.
2. **Radius Integration:** Update `inference_onnx_seq_gray_v2.py` and `ball_tracker.py` to handle the `Radius` field.
3. **Architecture Upgrade:** Update `src/models.py` (to be created) with `VballNetV2b` layer definitions.
4. **Tooling Expansion:** Import analytical scripts from the `scripts/` directory of the upstream repositories.
