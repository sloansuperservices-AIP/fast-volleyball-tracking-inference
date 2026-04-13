#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
Supports 4-step pipeline: detect -> track -> combined -> reels.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger(__name__)


def run_command(cmd: list[str]) -> None:
    """Helper to run subprocess commands."""
    LOG.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        LOG.error("Command failed with exit code %s", exc.returncode)
        sys.exit(exc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["detect", "track", "combined", "reels", "all", "pose", "hub-track", "analyze"],
        default="all",
        help="Pipeline step to execute",
    )
    parser.add_argument("--video_path", type=str, required=True, help="Input video path")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
        help="ONNX model path",
    )
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--court_json", type=str, help="Court geometry JSON (optional)")
    parser.add_argument("--visualize", action="store_true", help="Show processing preview")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    # Tracking params
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    parser.add_argument("--min_duration", type=float, default=1.0, help="Min track duration (sec)")

    # Hub specific
    parser.add_argument(
        "--hub_model",
        type=str,
        default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
        help="Ultralytics Hub model URL",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("ULTRALYTICS_HUB_API_KEY"),
        help="Hub API key",
    )

    args = parser.parse_args()

    video_name = Path(args.video_path).stem
    video_out_dir = Path(args.output_dir) / video_name
    ball_csv = video_out_dir / "ball.csv"
    tracks_dir = video_out_dir / "tracks"

    python_exe = sys.executable

    # 1. Detection
    if args.mode in ["detect", "all"]:
        LOG.info("Step 1: Ball Detection")
        cmd = [
            python_exe,
            "src/inference_onnx_seq_gray_v2.py",
            "--video_path",
            args.video_path,
            "--model_path",
            args.model_path,
            "--output_dir",
            args.output_dir,
            "--only_csv",
        ]
        if args.verbose:
            cmd.append("--verbose")
        run_command(cmd)

    # 2. Track Calculation
    if args.mode in ["track", "all"]:
        LOG.info("Step 2: Track Calculation")
        cmd = [
            python_exe,
            "src/track_calculator.py",
            "--csv_path",
            str(ball_csv),
            "--output_dir",
            args.output_dir,
            "--fps",
            str(args.fps),
            "--min_duration_sec",
            str(args.min_duration),
        ]
        if args.court_json:
            cmd.extend(["--court_json_path", args.court_json])
        if args.verbose:
            cmd.append("--verbose")
        run_command(cmd)

    # 3. Horizontal Assembly
    if args.mode in ["combined", "all"]:
        LOG.info("Step 3: Horizontal Assembly")
        cmd = [
            python_exe,
            "src/track_processor.py",
            "--video_path",
            args.video_path,
            "--json_dir",
            str(tracks_dir),
            "--output_dir",
            args.output_dir,
        ]
        run_command(cmd)

    # 4. Vertical Reels
    if args.mode in ["reels", "all"]:
        LOG.info("Step 4: Vertical Reel Generation")
        cmd = [
            python_exe,
            "src/make_reels.py",
            "--video_path",
            args.video_path,
            "--json_dir",
            str(tracks_dir),
            "--output_dir",
            args.output_dir,
        ]
        run_command(cmd)

    # Pose mode
    if args.mode == "pose":
        LOG.info("Pose Detection Mode")
        cmd = [
            python_exe,
            "src/pose_detector.py",
            "--video_path",
            args.video_path,
            "--output_dir",
            args.output_dir,
        ]
        if args.visualize:
            cmd.append("--visualize")
        run_command(cmd)

    # Hub tracking
    if args.mode == "hub-track":
        LOG.info("Hub Tracking Mode")
        cmd = [
            python_exe,
            "src/hub_inference.py",
            "--video_path",
            args.video_path,
            "--hub_model",
            args.hub_model,
            "--api_key",
            args.api_key if args.api_key else "",
            "--output_dir",
            args.output_dir,
        ]
        if args.visualize:
            cmd.append("--visualize")
        run_command(cmd)

    # Analyze placeholder
    if args.mode == "analyze":
        LOG.info("Analysis mode is not yet implemented in unified pipeline")

    return 0


if __name__ == "__main__":
    sys.exit(main())
