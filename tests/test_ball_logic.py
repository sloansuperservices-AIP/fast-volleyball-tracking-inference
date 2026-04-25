import unittest
import numpy as np
import cv2
import os
import sys
from collections import deque
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ball_tracker import BallTracker, Track
from inference_onnx_seq_gray_v2 import postprocess_output

class TestBallLogic(unittest.TestCase):
    def test_postprocess_output_with_radius(self):
        # Create a dummy heatmap with a circle
        heatmap = np.zeros((1, 1, 288, 512), dtype=np.float32)
        cv2.circle(heatmap[0, 0], (100, 150), 10, 1.0, -1)

        results = postprocess_output(heatmap, threshold=0.5, out_dim=1)
        self.assertEqual(len(results), 1)
        visibility, cx, cy, radius = results[0]
        self.assertEqual(visibility, 1)
        self.assertAlmostEqual(cx, 100, delta=2)
        self.assertAlmostEqual(cy, 150, delta=2)
        self.assertGreater(radius, 5) # minEnclosingCircle of a filled circle of radius 10 should be ~10

    def test_track_calculate_stats(self):
        track = Track(fps=30.0)
        # Add some positions
        # ( (x,y), frame )
        track.positions = deque([
            ([100.0, 100.0], 0),
            ([110.0, 110.0], 1),
            ([125.0, 125.0], 2)
        ])
        track.calculate_stats()

        self.assertEqual(track.max_height, 100.0)
        self.assertGreater(track.total_distance, 0)
        self.assertGreater(track.avg_speed, 0)
        self.assertGreater(track.max_speed, 0)

    def test_ball_tracker_radius(self):
        tracker = BallTracker()
        # Add 3 positions to satisfy the main_ball requirement (len >= 3)
        for f in range(3):
            detections = [
                {"x1": 90 + f*5, "y1": 140 + f*5, "x2": 110 + f*5, "y2": 160 + f*5, "confidence": 0.9} # Radius 10
            ]
            main_ball, tracks, deleted = tracker.update(detections, f)

        self.assertIsNotNone(main_ball)
        track = tracks[main_ball]
        self.assertEqual(len(track.ball_sizes), 3)
        self.assertAlmostEqual(track.ball_sizes[0], 10.0)

if __name__ == '__main__':
    unittest.main()
