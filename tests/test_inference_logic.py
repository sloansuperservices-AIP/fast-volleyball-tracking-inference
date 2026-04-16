import unittest
import numpy as np
from src.inference_onnx_seq_gray_v2 import postprocess_grid_output

class TestInferenceLogic(unittest.TestCase):
    def test_postprocess_grid_output(self):
        # Mock grid output: (1, 3*seq, rows, cols)
        seq = 1
        rows, cols = 2, 2
        input_height, input_width = 100, 100
        output = np.zeros((1, 3 * seq * rows * cols), dtype=np.float32)
        # Reshape to simulate postprocess_grid_output's reshape: (seq, 3, grid_rows, grid_cols)
        # Actually postprocess_grid_output does: output[0].reshape(seq, 3, grid_rows, grid_cols)

        # Manually construct the flattened array to match the reshape logic
        # Indexing: frame_idx=0, channel=0 (conf), row=1, col=1
        # channels: 0=conf, 1=x_offset, 2=y_offset
        conf_idx = 0 * 3 * 2 * 2 + 0 * 2 * 2 + 1 * 2 + 1
        x_off_idx = 0 * 3 * 2 * 2 + 1 * 2 * 2 + 1 * 2 + 1
        y_off_idx = 0 * 3 * 2 * 2 + 2 * 2 * 2 + 1 * 2 + 1

        output[0, conf_idx] = 0.9
        output[0, x_off_idx] = 0.5
        output[0, y_off_idx] = 0.5

        threshold = 0.5
        results = postprocess_grid_output(output, threshold, seq, input_height, input_width, rows, cols)

        self.assertEqual(len(results), 1)
        vis, x, y = results[0]
        self.assertEqual(vis, 1)
        # (col + offset) * (width / cols) = (1 + 0.5) * (100 / 2) = 1.5 * 50 = 75
        self.assertEqual(x, 75)
        self.assertEqual(y, 75)

    def test_postprocess_grid_output_low_conf(self):
        seq = 1
        rows, cols = 2, 2
        input_height, input_width = 100, 100
        output = np.zeros((1, 3 * seq * rows * cols), dtype=np.float32)

        conf_idx = 0 * 3 * 2 * 2 + 0 * 2 * 2 + 1 * 2 + 1
        output[0, conf_idx] = 0.1 # Below threshold

        threshold = 0.5
        results = postprocess_grid_output(output, threshold, seq, input_height, input_width, rows, cols)

        self.assertEqual(len(results), 1)
        vis, x, y = results[0]
        self.assertEqual(vis, 0)

if __name__ == '__main__':
    unittest.main()
