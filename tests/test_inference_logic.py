import unittest
from unittest.mock import MagicMock, patch
import sys
import numpy as np

# Mocking dependencies before import
mock_cv2 = MagicMock()
mock_pd = MagicMock()

with patch.dict('sys.modules', {
    'cv2': mock_cv2,
    'pandas': mock_pd,
    'onnxruntime': MagicMock(),
    'tqdm': MagicMock(),
    'filterpy': MagicMock(),
    'filterpy.kalman': MagicMock(),
    'scipy': MagicMock(),
    'scipy.signal': MagicMock(),
}):
    from src.inference_onnx_seq_gray_v2 import postprocess_grid_output, postprocess_heatmap_output

class TestInferenceLogic(unittest.TestCase):
    def test_heatmap_decoding(self):
        # Create a mock heatmap (1, 9, 288, 512)
        output = np.zeros((1, 9, 288, 512), dtype=np.float32)
        output[0, 0, 150, 100] = 1.0 # Object in frame 0

        # Mocking contours based on input image max value
        def mock_findContours(img, mode, method):
            if np.max(img) > 0:
                return [np.array([[100, 150]])], None
            return [], None

        mock_cv2.findContours.side_effect = mock_findContours

        def mock_threshold(img, thresh, maxval, type):
            if np.max(img) >= thresh:
                return None, np.ones((10, 10), dtype=np.uint8)
            else:
                return None, np.zeros((10, 10), dtype=np.uint8)

        mock_cv2.threshold.side_effect = mock_threshold
        mock_cv2.moments.return_value = {"m00": 1, "m10": 100, "m01": 150}
        mock_cv2.contourArea.return_value = 10.0

        results = postprocess_heatmap_output(output, threshold=0.5, out_dim=9)

        self.assertEqual(len(results), 9)
        self.assertEqual(results[0][0], 1) # Visible
        self.assertEqual(results[0][1], 100) # X
        self.assertEqual(results[0][2], 150) # Y
        self.assertEqual(results[1][0], 0) # Not visible (zeros)

    def test_grid_decoding(self):
        seq, rows, cols = 9, 27, 48
        output = np.zeros((1, seq * 3 * rows * cols), dtype=np.float32)
        reshaped = output.reshape(1, seq, 3, rows, cols)

        # Frame 0: ball at row 10, col 20 with offset 0.5
        reshaped[0, 0, 0, 10, 20] = 0.9
        reshaped[0, 0, 1, 10, 20] = 0.5
        reshaped[0, 0, 2, 10, 20] = 0.5

        input_w, input_h = 768, 432

        results = postprocess_grid_output(
            output, threshold=0.5, seq=seq,
            input_height=input_h, input_width=input_w,
            grid_rows=rows, grid_cols=cols
        )

        self.assertEqual(len(results), 9)
        self.assertEqual(results[0][0], 1) # Visible

        # x = (col + offset) * (input_w / cols) = (20 + 0.5) * 16 = 328
        self.assertEqual(results[0][1], 328)

        # y = (row + offset) * (input_h / rows) = (10 + 0.5) * 16 = 168
        self.assertEqual(results[0][2], 168)

        self.assertEqual(results[1][0], 0) # Not visible

if __name__ == '__main__':
    unittest.main()
