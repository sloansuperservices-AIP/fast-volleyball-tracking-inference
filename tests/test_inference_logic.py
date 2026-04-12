import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from inference_onnx_seq_gray_v2 import postprocess_heatmap_output, postprocess_grid_output

class TestInferenceLogic(unittest.TestCase):
    def test_postprocess_heatmap(self):
        # Create a mock heatmap (batch=1, seq=3, h=10, w=10)
        output = np.zeros((1, 3, 10, 10), dtype=np.float32)
        # Add a ball in the second frame at (5, 5)
        output[0, 1, 4:7, 4:7] = 0.8

        results = postprocess_heatmap_output(output, threshold=0.5, input_height=100, input_width=100, out_dim=3)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], (0, 0, 0)) # No ball
        self.assertEqual(results[1][0], 1) # Ball detected
        self.assertEqual(results[1][1], 5) # x coord (center of 4,5,6 is 5)
        self.assertEqual(results[1][2], 5) # y coord
        self.assertEqual(results[2], (0, 0, 0)) # No ball

    def test_postprocess_grid(self):
        # Create a mock grid output (batch=1, seq=1, channels=3, grid_h=2, grid_w=2)
        # We need to simulate the reshaped output [batch, seq, 3, grid_h, grid_w]
        # But postprocess_grid_output expects the raw output from session.run, then reshapes it.
        # Raw output: [1, seq*3, grid_h, grid_w]
        output = np.zeros((1, 3, 2, 2), dtype=np.float32)
        # conf map (index 0)
        output[0, 0, 0, 1] = 0.9 # Ball at grid row 0, col 1
        # x_offset (index 1)
        output[0, 1, 0, 1] = 0.5 # Halfway in grid cell
        # y_offset (index 2)
        output[0, 2, 0, 1] = 0.2

        # grid_size = input_size / grid_cells = 100 / 2 = 50
        # x_pred = (col + x_offset) * grid_size = (1 + 0.5) * 50 = 75
        # y_pred = (row + y_offset) * grid_size = (0 + 0.2) * 50 = 10

        results = postprocess_grid_output(output, threshold=0.5, seq=1, input_height=100, input_width=100, grid_rows=2, grid_cols=2)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 1)
        self.assertEqual(results[0][1], 75)
        self.assertEqual(results[0][2], 10)

if __name__ == '__main__':
    unittest.main()
