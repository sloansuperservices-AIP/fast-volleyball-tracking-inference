# Volleyball Tracking Updates Report - May 2026

This report summarizes the recent updates from public volleyball tracking repositories and suggests further integrations based on the latest releases and research.

## 🟢 Recently Integrated Updates

The local repository has been synchronized with the latest advancements from `asigatchov/fast-volleyball-tracking-inference` (upstream):

### 1. Unified Inference & Multi-Model Support
- **`seq15` Support**: Integrated support for 15-frame temporal window models, providing better motion consistency over the previous `seq9` standard.
- **Grid-based Models**: Added support for `VballNetGrid` architectures, which achieve significantly higher FPS by using a 48x27 grid output instead of full-resolution heatmaps.
- **Ball Radius Estimation**: Automated estimation of ball radius using visual contours and motion masks, now included in the `ball.csv` output.
- **Dynamic Library Loading**: Added robust `LD_LIBRARY_PATH` resolution to ensure `onnxruntime` correctly finds Nvidia CUDA libraries across different environments.

### 2. Optimized Runtimes
- **OpenVINO Inference**: Added `src/inference_openvino_seq_gray_v2.py`, enabling hardware-accelerated inference on Intel CPUs and iGPUs.
- **Distributed Processing**: Integrated `src/celery_worker.py` to support background task orchestration via Redis and Celery.

### 3. Pipeline Orchestration
- **Main Entry Point**: Updated `main.py` with support for `openvino-track`, `analyze`, and `hub-track` modes.
- **Domain Models**: Centralized logic into `src/constants.py` and `src/models.py` for improved maintainability.

---

## 🟡 Suggested Future Updates

The following features from the research repositories are recommended for future integration:

### 1. Advanced Architecture (from `vball-net`)
- **VballNetV2b**: Incorporates `MotionPromptLayer` for explicit temporal attention and `DyT` (Dynamic Transformation) layers. This model shows promise in reducing false positives during fast player movements.
- **Spatial Attention**: Integration of the new attention layers found in the latest `vball-net` training notebooks.

### 2. Research & Training (from `TrackNetV4-PyTorch`)
- **PyTorch Training Pipeline**: Transitioning from Keras to PyTorch would allow the use of modern optimizers like **Lion** and **AdamW**, potentially reaching higher F1 scores on existing datasets.
- **Center-Frame Prediction**: Adopting the refined evaluation metrics used in `TrackNetV4-PyTorch` to better align with competition standards.

### 3. Analytical Tools (from `fast-volleyball-tracking-inference` scripts)
- **Zone-4 Trajectory Analysis**: Implementing the antenna-aware perspective correction to automatically identify attack corridors and generate specialized reels for coaches.
- **Perspective Transformer**: Porting the `court_transformer.py` logic to allow 3D trajectory reconstruction in court coordinates.

---

## 🛠 Project Status
- **Environment**: Standardized on Python 3.12+ and `uv` for dependency management.
- **Performance**: Up to 300+ FPS on CPU with optimized OpenVINO models.
- **Security**: Hardcoded API keys removed; all sensitive operations moved to environment variables.
