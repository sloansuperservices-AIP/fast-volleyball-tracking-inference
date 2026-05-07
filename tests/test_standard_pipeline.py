import unittest
import os
import sys
import subprocess
import pandas as pd
import numpy as np

class TestStandardPipeline(unittest.TestCase):
    def test_help_messages(self):
        """Verify that all scripts can be run with --help."""
        scripts = [
            ["main.py", "--help"],
            ["src/inference_onnx_seq_gray_v2.py", "--help"],
            ["src/inference_onnx_seq9_gray_v2.py", "--help"],
            ["src/track_calculator.py", "--help"],
            ["src/track_processor.py", "--help"],
            ["src/make_reels.py", "--help"],
        ]
        for cmd_args in scripts:
            with self.subTest(script=cmd_args[0]):
                cmd = [sys.executable] + cmd_args
                result = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, f"Script {cmd_args[0]} failed to show help. Error: {result.stderr}")

if __name__ == "__main__":
    unittest.main()
