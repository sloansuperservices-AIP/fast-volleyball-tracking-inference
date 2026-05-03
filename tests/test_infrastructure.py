import unittest
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from constants import DEFAULT_FPS, DEFAULT_INPUT_WIDTH
from models import BallTrack, BallDetection

class TestCoreInfrastructure(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(DEFAULT_FPS, 30.0)
        self.assertEqual(DEFAULT_INPUT_WIDTH, 512)

    def test_ball_track_model(self):
        track = BallTrack(maxlen=5)
        self.assertEqual(track.maxlen, 5)
        track.update((10, 20))
        self.assertEqual(len(track.points()), 1)
        self.assertEqual(track.tail(), (10, 20))
        track.reset()
        self.assertEqual(len(track.points()), 0)

    def test_ball_detection_model(self):
        det = BallDetection(frame_index=1, visible=True, x=100.5, y=200.5)
        self.assertEqual(det.frame_index, 1)
        self.assertTrue(det.visible)
        self.assertEqual(det.x, 100.5)

if __name__ == "__main__":
    unittest.main()
