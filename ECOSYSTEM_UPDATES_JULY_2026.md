# Volleyball Tracking Ecosystem Updates - July 2026

This document summarizes the recent updates and suggested contributions from the four tracked public repositories.

## 1. fast-volleyball-tracking-inference (Core Pipeline)
**Status:** Synchronized with `upstream/master`.
- **New Feature: Frame Rotation:** Support for `--rotate` (90, -90, 180 degrees) added to the pipeline to handle vertical or inverted video sources.
- **Model Updates:** Integrated `ballNetGridV1b_seq9_grayscale_20260510` and seq15 models.
- **Localization:** OpenVINO inference script now includes localized (Russian) messages for better user feedback in specific regions.

**Suggestions:**
- Extend rotation support to all preprocessing tools in the `scripts/` directory.
- Standardize the localization framework across all inference scripts (ONNX/OpenVINO).

## 2. GridTrackNet (High-Speed Tracking)
**Recent Contributions:**
- **Unified Inference Workflow:** Recent commits show a push towards a more unified inference script with ONNX runtime support.
- **Improved Data Generation:** `DataGen.py` received error handling improvements for TFRecord generation.
- **Labelling Tool:** A PySide6-based labelling tool is now stable for custom dataset creation.

**Suggestions:**
- Evaluate the GridTrackNet MIMO architecture (5-frame input/output) for integration into the main pipeline as a "high-speed" alternative (targeting 100+ FPS on mobile/edge devices).

## 3. vball-net (Experimental Hub)
**Recent Contributions:**
- **Dynamic Transformation (DyT) Layer:** Continued experiments with DyT and `softsign` activations for lightweight accuracy gains.
- **Spatial Attention:** Introduction of spatial attention modules in `VballNetV2b`.
- **Training Repeatability:** Commits focus on accelerating and improving the repeatability of training runs.

**Suggestions:**
- Port the `VballNetV2b` architecture once the DyT layer experiments stabilize.
- Incorporate the mixup visualization tools into the local training diagnostics.

## 4. TrackNetV4-PyTorch (Professional Training)
**Recent Contributions:**
- **Motion Attention Modules:** Implemented motion attention layers to focus the model on moving objects (the ball) across temporal frames.
- **Modern Optimizers:** Full support for `Lion` and `AdamW` optimizers, which have shown better convergence in recent benchmarks.
- **Weighted BCE Loss:** Refined the weighted binary cross-entropy loss to strictly follow the latest research papers.

**Suggestions:**
- Adopt the `Lion` optimizer for local model fine-tuning.
- Integrate the motion attention module into the core model architecture to improve tracking in cluttered backgrounds.

---
*Report generated on July 3, 2026.*
