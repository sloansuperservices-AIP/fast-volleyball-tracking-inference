import unittest
import numpy as np
from src.inference_onnx_seq_gray_v2 import postprocess_heatmap_output, postprocess_grid_output

class TestInferenceLogic(unittest.TestCase):
    def test_postprocess_heatmap_output(self):
        # Create a mock heatmap with a single "ball" at (10, 20)
        output = np.zeros((1, 1, 288, 512), dtype=np.float32)
        # Note: Contours expect 255-range for thresholding if logic multiplies by 255
        output[0, 0, 20, 10] = 1.0

        # Increase radius slightly to ensure contour detection
        output[0, 0, 19:22, 9:12] = 1.0

        results = postprocess_heatmap_output(output, threshold=0.5, out_dim=1)
        self.assertEqual(len(results), 1)
        visibility, x, y = results[0]
        self.assertEqual(visibility, 1)
        # Moments might shift slightly from center depending on contour shape
        self.assertAlmostEqual(x, 10, delta=1)
        self.assertAlmostEqual(y, 20, delta=1)

    def test_postprocess_grid_output(self):
        # Grid dimensions: 48x27
        seq = 1
        grid_rows, grid_cols = 27, 48
        output = np.zeros((1, seq * 3 * grid_rows * grid_cols), dtype=np.float32)
        reshaped = output[0].reshape(seq, 3, grid_rows, grid_cols)

        # Frame 0: Conf=1.0 at grid (5, 10)
        # indices are [frame, channel, row, col]
        reshaped[0, 0, 10, 5] = 1.0
        reshaped[0, 1, 10, 5] = 0.5 # x offset
        reshaped[0, 2, 10, 5] = 0.5 # y offset

        results = postprocess_grid_output(
            output, threshold=0.5, seq=seq,
            input_height=432, input_width=768,
            grid_rows=grid_rows, grid_cols=grid_cols
        )

        self.assertEqual(len(results), 1)
        visibility, x, y = results[0]
        self.assertEqual(visibility, 1)
        # Calculation: x = (5 + 0.5) * (768/48) = 5.5 * 16 = 88
        #              y = (10 + 0.5) * (432/27) = 10.5 * 16 = 168
        self.assertEqual(x, 88)
        self.assertEqual(y, 168)

if __name__ == "__main__":
    unittest.main()
