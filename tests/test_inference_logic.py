import unittest
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch

# Mock cv2 and other heavy dependencies for fast logic testing
sys.modules['cv2'] = MagicMock()
sys.modules['onnxruntime'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['tqdm'] = MagicMock()

import src.inference_onnx_seq_gray_v2 as inference

class TestInferenceLogic(unittest.TestCase):
    def test_infer_model_params_grid(self):
        params = inference.infer_model_params("VballNetGridV1b.onnx")
        self.assertEqual(params["family"], "grid")
        self.assertEqual(params["input_width"], 768)

    def test_infer_model_params_heatmap(self):
        params = inference.infer_model_params("VballNetV1_seq9.onnx")
        self.assertEqual(params["family"], "heatmap")
        self.assertEqual(params["seq"], 9)

    def test_postprocess_grid_output(self):
        # Create a dummy output: seq=1, 3 channels, 27 rows, 48 cols
        dummy_output = np.zeros((1, 3, 27, 48), dtype=np.float32)
        # Set max confidence at row 10, col 20
        dummy_output[0, 0, 10, 20] = 0.9
        # Set offsets to 0.5
        dummy_output[0, 1, 10, 20] = 0.5
        # Set y-offset to 0.2
        dummy_output[0, 2, 10, 20] = 0.2

        results = inference.postprocess_grid_output(
            dummy_output, 0.5, 1, 432, 768, 27, 48
        )

        self.assertEqual(len(results), 1)
        vis, x, y = results[0]
        self.assertEqual(vis, 1)
        # x = (20 + 0.5) * (768 / 48) = 20.5 * 16 = 328
        self.assertEqual(x, 328)
        # y = (10 + 0.2) * (432 / 27) = 10.2 * 16 = 163.2 -> 163
        self.assertEqual(y, 163)

if __name__ == "__main__":
    unittest.main()
