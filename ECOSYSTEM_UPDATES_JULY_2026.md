# Ecosystem Updates Assessment - July 2026

Following a review of the related repositories, here are the suggested updates and new contributions that have been identified and/or integrated:

## 1. fast-volleyball-tracking-inference (Upstream)
*   **Rotation Support:** Added parity for the `--rotate` argument (-90, 90, 180 degrees) across both ONNX and OpenVINO pipelines. This allows for processing vertical videos or inverted camera setups. (Integrated)
*   **Improved Model Inference:** Updated `infer_model_params` to dynamically detect sequence lengths (e.g., `seq15` vs `seq9`) from model filenames for both Heatmap and Grid-based models. (Integrated)
*   **Zone 4 Trajectory Tooling:** New scripts for analyzing ball trajectories near the antennas (Zone 4) to evaluate attack quality. (Available in `scripts/analyze_zone4_ball_trajectories.py`)
*   **20260510 Model series:** Support for the latest `ballNetGridV1b_seq9_grayscale_20260510` models which offer improved stability.

## 2. GridTrackNet
*   **ONNX Runtime Support:** Upstream has unified its inference workflow to support ONNX runtime alongside TensorFlow.
*   **Ultra-efficient Architecture:** GridTrackNet continues to be the performance leader for high-speed tracking (100+ FPS on edge hardware).

## 3. TrackNetV4-PyTorch
*   **Motion Attention Maps:** Professional-grade implementation of motion attention maps. This contribution improves tracking of fast-moving objects by focusing the model on temporal differences between frames.
*   **Modern Optimizers:** Support for Lion and AdamW optimizers in the training pipeline for better convergence.

## 4. vball-net
*   **Experimental Layers:** Research into DyT (Dynamic Tanh) and softsign activation layers to reduce computational overhead while maintaining non-linearity.
*   **Training Repeatability:** Optimizations to ensure more consistent results across different training runs.

## Recommended Next Steps for this Repository:
1.  **Adopt Motion Attention:** Evaluate integrating the Motion Attention Map logic from `TrackNetV4-PyTorch` into the `VballNet` architecture for the next model iteration.
2.  **Experimental Activations:** Test the `DyT` layers from `vball-net` to potentially increase FPS on older CPU/GPU hardware.
