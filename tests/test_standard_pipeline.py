import unittest
import subprocess
import os
import sys
from pathlib import Path

class TestStandardPipeline(unittest.TestCase):
    def test_main_help(self):
        result = subprocess.run([sys.executable, "main.py", "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("Unified Volleyball Tracking Pipeline", result.stdout)

    def test_inference_help(self):
        result = subprocess.run([sys.executable, "src/inference_onnx_seq_gray_v2.py", "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--video_path", result.stdout)

    def test_calculator_help(self):
        result = subprocess.run([sys.executable, "src/track_calculator.py", "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--csv_path", result.stdout)

if __name__ == "__main__":
    unittest.main()
