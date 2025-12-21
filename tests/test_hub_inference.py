import unittest
from unittest.mock import MagicMock, patch, ANY
import os
import shutil
import cv2
import pandas as pd
import numpy as np
from src.hub_inference import run_hub_inference

class TestHubInference(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = "test_output_hub"
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
        os.makedirs(self.test_output_dir)

        # Create a dummy video
        self.video_path = os.path.join(self.test_output_dir, "test.mp4")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (100, 100))
        for _ in range(5):
            out.write(frame)
        out.release()

    def tearDown(self):
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    @patch('src.hub_inference.YOLO')
    @patch('src.hub_inference.cv2.imshow')
    @patch('src.hub_inference.cv2.waitKey')
    def test_run_hub_inference(self, mock_waitKey, mock_imshow, MockYOLO):
        # Mock the model and its return values
        mock_model = MockYOLO.return_value

        # Create mock results
        mock_result = MagicMock()
        mock_result.orig_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock detection box
        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = [10, 10, 50, 50]
        mock_box.conf = [MagicMock()]
        mock_box.conf[0].cpu.return_value.numpy.return_value = 0.9
        mock_box.cls = [MagicMock()]
        mock_box.cls[0].cpu.return_value.numpy.return_value = 0

        mock_result.boxes = [mock_box]

        # Mock model.track to return a list of 5 results (one for each frame)
        mock_model.track.return_value = [mock_result] * 5

        run_hub_inference(
            video_path=self.video_path,
            model_url="https://hub.ultralytics.com/models/test",
            api_key="test_key",
            output_dir=self.test_output_dir,
            visualize=False
        )

        # Verify model loaded
        MockYOLO.assert_called_with("https://hub.ultralytics.com/models/test")

        # Verify model.track called
        mock_model.track.assert_called_with(
            source=self.video_path,
            stream=True,
            persist=True,
            conf=0.25,
            iou=0.45
        )

        # Verify output files
        csv_path = os.path.join(self.test_output_dir, "test_hub_predict.csv")
        video_path = os.path.join(self.test_output_dir, "test_hub_predict.mp4")

        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(video_path))

        # Verify CSV content
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 5)
        self.assertEqual(df.iloc[0]["Visibility"], 1)
        self.assertEqual(df.iloc[0]["Class"], 0)

if __name__ == '__main__':
    unittest.main()
