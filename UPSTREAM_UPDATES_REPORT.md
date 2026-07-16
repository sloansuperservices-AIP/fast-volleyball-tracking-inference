# UPSTREAM REPOSITORIES ASSESSMENT & UPDATE RECOMMENDATIONS REPORT

This report provides a comprehensive review of the active public repositories under `https://github.com/asigatchov/` within the volleyball tracking ecosystem. It assesses recent releases, updates, and contributions across these repositories, compares them with our local project structure, and presents concrete integration recommendations for future enhancement of our fast volleyball tracking and vertical reels generation system.

---

## 1. Overview of Checked Repositories

### A. fast-volleyball-tracking-inference (Core Inference / OpenVINO System)
* **Repository:** `https://github.com/asigatchov/fast-volleyball-tracking-inference.git`
* **Latest Master Release / Commit:** Commit `637c217` ("rotate vertical") dated June 15, 2026.
* **Key Active Branches:**
  * `master`: Production-ready unified ball tracking, court transformation, Savitzky-Golay/Kalman trajectory smoothing, and 9:16 vertical reel maker pipelines.
  * `player-tracker`: Implements YOLO ONNX-based multi-player detection, bottom-edge foot position mapping, and DeepSORT-like ID-retaining tracking.
  * `celery`: Distributed processing task-queue implementation for video inference workloads.
  * `use-court-cooordinats`: Explores using physical court dimensions and net/court filtering to improve ball tracking quality and eliminate outliers.
  * `dev-rally`: Experimental branch focusing on rally extraction and segment classification algorithms.

### B. GridTrackNet (High-Speed Multi-Input Multi-Output Tracking)
* **Repository:** `https://github.com/asigatchov/GridTrackNet.git`
* **Latest Main Commit:** Commit `9682321` ("rebuild src") dated March 17, 2026.
* **Key Active Branches:**
  * `main`: Modular GridTrackNet model definition, PySide6-based labelling tool (`LabellingTool.py`), and data preparation/TFRecord generation pipeline (`DataGen.py`).
  * `run-train`: Refines label generation and fixes training setup loops.
* **Architecture Highlight:** A highly optimized MIMO architecture utilizing 5 temporal frames as inputs to output both confidence maps and offset grids at a 768x432 resolution, capable of running over 110 FPS on local hardware.

### C. vball-net (Experimental Architectures / DyT Research Hub)
* **Repository:** `https://github.com/asigatchov/vball-net.git`
* **Latest Main Commit:** Commit `2de50e4` ("LICENSE") dated June 8, 2026.
* **Key Active Branches:**
  * `main`: Contains Keras-based model architectures (`VballNetV2b`, `VballNetV1`), training notebooks (`vball_net_train.ipynb`), and activation function research.
  * `fast-train`: Evaluates in-memory data loading performance to accelerate and improve repeatability in high-performance training environments.
* **Architecture Highlight:** Explores the experimental "Dynamic Tanh" (`DyT`) and `softsign` activation layers to improve model representation capacity and convergence rates during training.

### D. TrackNetV4-PyTorch (Research Implementations & Optimizations)
* **Repository:** `https://github.com/asigatchov/TrackNetV4-PyTorch.git`
* **Latest Main Commit:** Commit `9fc8a18` ("Merge pull request #4: Rewrite README to be more concise, accurate and professional") dated August 1, 2025.
* **Key Active Branches:**
  * `main`: Fully rewritten PyTorch implementation of TrackNetV4 with motion attention map generation.
  * `streem_video_predict`: Implements batched inference, configuration handlers, and real-time streaming video prediction.
  * `vballnet_v3.3`: Integrates optional GRU recurrent layers and sequence exportation/training loops.
* **Optimization Highlight:** Implements advanced training loops using modern optimizers like `Lion` and `AdamW` to stabilize attention weight convergence during temporal sequence learning.

---

## 2. Integrated Features vs. Outstanding Contributions

To ensure complete clarity on the state of our local repository, the table below highlights which upstream features have been integrated and which are currently candidates for suggested future updates:

| Feature/Component | Upstream Source Repo | Status in Local Repo | Description / Recommended Next Step |
| :--- | :--- | :--- | :--- |
| **`--rotate` Support** | `fastvball` (`master`) | **Integrated** | Allows rotating input frames (90, -90, 180 deg) before inference. |
| **`VballNetGrid` Models** | `fastvball` (`master`) | **Integrated** | Integrated `ballNetGridV1b_seq9_grayscale_20260510` ONNX & OpenVINO models as defaults. |
| **Vertical Reels Pipeline** | `fastvball` (`master`) | **Integrated** | FFmpeg-based vertical crop generation centered on ball trajectories. |
| **Model Architectures** | `vballnet` / `gridtracknet` | **Integrated** | Integrated `GridTrackNet.py`, `TrackNetV4.py`, `VballNetV2b.py` into `src/model/`. |
| **Player Tracking & IDs**| `fastvball` (`player-tracker`) | **Outstanding** | Implements a DeepSORT-like tracker and court mapping for player footprints. |
| **Distributed Task Queue**| `fastvball` (`celery`) | **Outstanding** | Integrates Celery + Redis task orchestration for multi-user scaling. |
| **DyT & Softsign Layers**| `vballnet` (`main`) | **Outstanding** | Integrates experimental lightweight activation function research into model files. |
| **Streaming & Batched PyTorch**| `tracknetv4` (`streem_video_predict`) | **Outstanding** | Real-time prediction streaming & high-throughput batched tensor inference. |

---

## 3. Core Recommendations & Suggestion Packages

We recommend organizing future developments into three logical suggestion packages, prioritizing business value, pipeline performance, and system stability.

### Recommendation Package A: Enterprise Production scaling (Highly Recommended)
1. **Asynchronous Distributed Workloads (`fastvball/celery` branch):**
   * **Suggestion:** Adopt the Celery and Redis-based worker layout defined in `src/celery_worker.py`. This will allow transitioning the unified pipeline into a microservice-ready backend capable of handling multiple concurrent reel generation requests.
   * **Benefits:** Prevents long-running video segmentation or tracking from locking orchestrator execution threads, opening the path for cloud API or web application interfaces.

2. **Advanced Court-Aware Ball Filtering (`fastvball/use-court-cooordinats` branch):**
   * **Suggestion:** Fully integrate court boundary polygons and net filters from the `court_transformer` into our main tracking calculator.
   * **Benefits:** Effectively filters out background movement, spectator false-positives, and ball detections that are physically impossible given the court constraints.

### Recommendation Package B: Comprehensive Player Analytics (Medium Priority)
1. **ONNX Player Tracking & Footprint Coordinates (`fastvball/player-tracker` branch):**
   * **Suggestion:** Integrate player detection (`yolov8n-pose.pt` or ONNX representation) with DeepSORT/ proximity-based ID tracking. Calculate foot coordinates at the bottom center of the bounding box and map them onto the court grid.
   * **Benefits:** Elevates the tool from simple ball-centric highlight reels to an advanced tactical analysis board, tracking team positioning, player velocities, and space coverage.

2. **Jump Tracking Analytics (`fastvball/master` branch components):**
   * **Suggestion:** Harness `src/jump_tracker.py` and pose landmarks from MediaPipe/YOLO-Pose to identify vertical takeoff spikes and block events.
   * **Benefits:** Generates vertical jump height metrics to enrich player performance dashboards.

### Recommendation Package C: Experimental Model Training & Optimization (Low/R&D Priority)
1. **Dynamic Transformation Activation Layers (`vballnet/main` branch):**
   * **Suggestion:** Bring `softsign` and experimental Dynamic Tanh (`DyT`) layers into local Keras models to evaluate accuracy-per-parameter trade-offs.
   * **Benefits:** Enhances model non-linearity for fine-grained ball segmentation with negligible latency cost.

2. **Lion and AdamW Optimizer Training Recipes (`tracknetv4/main` branch):**
   * **Suggestion:** Adopt the modernized `_create_optimizer` factory patterns and training recipes featuring `Lion` (via the `lion-pytorch` library) to optimize attention map weights.
   * **Benefits:** Improves training convergence, temporal learning repeatability, and robustness to video noise during neural network optimization.

---

## 4. Summary & Actionable Roadmap

By tracking and categorizing these public contributions, our pipeline remains synchronized with cutting-edge academic and community research.

* **Next Immediate Step:** We suggest developing a unified CLI orchestrator extension in `main.py` that optionally launches player tracking and court transformations via a `--mode players` flag.
* **Secondary Phase:** Introduce Celery/Redis containerization configuration for robust distributed cloud deployments.
