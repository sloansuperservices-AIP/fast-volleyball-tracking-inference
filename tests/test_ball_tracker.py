import unittest
import numpy as np
from src.ball_tracker import BallTracker, Track

class TestBallTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = BallTracker(
            buffer_size=100,
            max_disappeared=5,
            max_distance=50
        )

    def test_add_track(self):
        detection = {"x1": 100, "y1": 100, "x2": 120, "y2": 120}
        self.tracker._add_track(detection, 1)
        self.assertEqual(len(self.tracker.tracks), 1)
        self.assertEqual(self.tracker.next_id, 1)
        track = self.tracker.tracks[0]
        self.assertEqual(track.start_frame, 1)
        self.assertEqual(track.last_frame, 1)

    def test_update_track(self):
        # Add initial track
        det1 = {"x1": 100, "y1": 100, "x2": 120, "y2": 120}
        self.tracker._add_track(det1, 1)

        # Update it
        det2 = {"x1": 105, "y1": 105, "x2": 125, "y2": 125}
        self.tracker._update_track(0, det2, 2)

        track = self.tracker.tracks[0]
        self.assertEqual(len(track.positions), 2)
        self.assertEqual(track.last_frame, 2)
        # Center of det2 is (115, 115)
        self.assertEqual(track.positions[-1][0], [115.0, 115.0])

    def test_track_to_dict_and_from_dict(self):
        track = Track(track_id=42, start_frame=10, last_frame=20)
        track.positions.append(([100.0, 200.0], 10))

        d = track.to_dict()
        self.assertEqual(d["track_id"], 42)
        self.assertEqual(d["start_frame"], 10)

        new_track = Track.from_dict(d)
        self.assertEqual(new_track.track_id, 42)
        self.assertEqual(list(new_track.positions), list(track.positions))

if __name__ == '__main__':
    unittest.main()
