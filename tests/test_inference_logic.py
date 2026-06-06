import pytest
import numpy as np
import os
import cv2
from src.inference_onnx_seq_gray_v2 import infer_model_params, postprocess_heatmap_output

def test_infer_model_params():
    params = infer_model_params("models/VballNetV1_seq15_grayscale_best.onnx")
    assert params["seq"] == 15
    assert params["family"] == "heatmap"

    params = infer_model_params("models/VballNetGridV1b_seq9.onnx")
    assert params["family"] == "grid"
    assert params["input_width"] == 768

def test_postprocess_heatmap_output():
    # Heatmap postprocessing uses cv2.findContours which expects more than a single pixel usually to find a contour
    output = np.zeros((1, 9, 288, 512), dtype=np.float32)
    # Create a small 3x3 blob
    output[0, 0, 100:103, 200:203] = 1.0

    results = postprocess_heatmap_output(output, threshold=0.5, out_dim=9)
    assert len(results) == 9
    assert results[0][0] == 1 # Visibility
    # Center of 200,201,202 should be 201
    assert abs(results[0][1] - 201) <= 1
    assert abs(results[0][2] - 101) <= 1
    assert results[1] == (0, 0, 0)
