import unittest
import numpy as np
from src.inference_onnx_seq_gray_v2 import infer_model_params, decode_predictions

class TestInferenceLogic(unittest.TestCase):
    def test_infer_model_params_heatmap(self):
        params = infer_model_params("models/VballNetV1_seq9_grayscale_330.onnx")
        self.assertEqual(params["family"], "heatmap")
        self.assertEqual(params["seq"], 9)
        self.assertEqual(params["input_width"], 512)

    def test_infer_model_params_grid(self):
        params = infer_model_params("models/VballNetGridV1b_seq9_grayscale.onnx")
        self.assertEqual(params["family"], "grid")
        self.assertEqual(params["seq"], 9)
        self.assertEqual(params["input_width"], 768)

    def test_decode_predictions_heatmap(self):
        # Mock output heatmap (1, 9, 288, 512)
        output = np.zeros((1, 9, 288, 512), dtype=np.float32)
        # Set a peak in the first frame. Using a small region around the peak
        # because cv2.findContours needs a region, not just a single pixel.
        output[0, 0, 100:103, 200:203] = 1.0

        model_params = {
            "family": "heatmap",
            "seq": 9,
            "input_height": 288,
            "input_width": 512
        }

        results = decode_predictions(output, model_params, threshold=0.5)
        self.assertEqual(len(results), 9)
        # Moments of a 3x3 block starting at (200, 100) center at (201, 101)
        self.assertEqual(results[0], (1, 201, 101))
        self.assertEqual(results[1], (0, 0, 0))

    def test_decode_predictions_grid(self):
        # Mock output grid (1, 27, 48, 27) -> Reshaped to (9, 3, 27, 48)
        # Output shape is [1, C, H, W] or similar.
        # In postprocess_grid_output: output = output[0].reshape(seq, 3, grid_rows, grid_cols)
        # So it expects output[0] to have size seq * 3 * grid_rows * grid_cols
        seq = 9
        grid_rows = 27
        grid_cols = 48
        output = np.zeros((1, seq * 3 * grid_rows * grid_cols), dtype=np.float32)
        output_reshaped = output[0].reshape(seq, 3, grid_rows, grid_cols)

        # Set a detection in frame 0
        # channel 0: confidence, 1: x_offset, 2: y_offset
        output_reshaped[0, 0, 10, 20] = 0.9 # confidence
        output_reshaped[0, 1, 10, 20] = 0.5 # x_offset
        output_reshaped[0, 2, 10, 20] = 0.2 # y_offset

        model_params = {
            "family": "grid",
            "seq": 9,
            "input_height": 432,
            "input_width": 768,
            "grid_rows": 27,
            "grid_cols": 48
        }

        results = decode_predictions(output, model_params, threshold=0.5)
        self.assertEqual(len(results), 9)

        # Expected x = (col + offset) * (width / grid_cols) = (20 + 0.5) * (768 / 48) = 20.5 * 16 = 328
        # Expected y = (row + offset) * (height / grid_rows) = (10 + 0.2) * (432 / 27) = 10.2 * 16 = 163.2 -> 163
        self.assertEqual(results[0], (1, 328, 163))

if __name__ == '__main__':
    unittest.main()
