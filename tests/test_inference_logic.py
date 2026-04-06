import unittest
import numpy as np
from src.inference_onnx_seq_gray_v2 import postprocess_output

class TestInference(unittest.TestCase):
    def test_postprocess_heatmap(self):
        # Create a mock heatmap (1, 3, 288, 512)
        output = np.zeros((1, 3, 288, 512), dtype=np.float32)
        # Add a "ball" blob at (100, 200) in the first frame
        output[0, 0, 199:202, 99:102] = 1.0

        results = postprocess_output(output, threshold=0.5, out_seq_len=3, model_type="heatmap")
        self.assertEqual(len(results), 3)
        # Moments of a 3x3 square centered at (100, 200) should be (100, 200)
        self.assertEqual(results[0], (1, 100, 200))
        self.assertEqual(results[1], (0, 0, 0))

    def test_postprocess_grid(self):
        # Create a mock grid output (1, 3*3, 27, 48)
        # out_seq_len = 3, each has 3 channels: vis, x_offset, y_offset
        output = np.zeros((1, 9, 27, 48), dtype=np.float32)

        # Grid index (10, 20), offset (0.5, 0.5)
        # Final coord: (20 + 0.5) * (512 / 48), (10 + 0.5) * (288 / 27)
        output[0, 0, 10, 20] = 1.0 # Visibility for frame 0
        output[0, 1, 10, 20] = 0.5 # X-offset
        output[0, 2, 10, 20] = 0.5 # Y-offset

        results = postprocess_output(output, threshold=0.5, out_seq_len=3, model_type="grid")
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0][0], 1)
        # (20 + 0.5) * (512 / 48) = 20.5 * 10.666 = 218.666 -> 218
        # (10 + 0.5) * (288 / 27) = 10.5 * 10.666 = 112
        self.assertEqual(results[0][1], int((20 + 0.5) * (512 / 48)))
        self.assertEqual(results[0][2], int((10 + 0.5) * (288 / 27)))

if __name__ == "__main__":
    unittest.main()
