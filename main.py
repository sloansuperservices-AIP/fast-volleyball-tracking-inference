#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
Orchestrates the 4-step pipeline: detect, track, combined, reels.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def run_command(cmd: list[str]) -> None:
    LOG.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        LOG.error("Command failed with exit code %s", result.returncode)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["detect", "track", "combined", "reels", "all", "pose", "hub-track"],
        default="all",
        help="Processing mode (default: all)",
    )
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
        help="Path to ball detection ONNX model",
    )
    parser.add_argument("--output_dir", type=str, default="output", help="Root output directory")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    # Mode-specific arguments
    parser.add_argument("--track_file", type=str, help="Path to track JSON (for pose mode)")
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
        help="Ultralytics Hub API key",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    video_path = Path(args.video_path)
    if not video_path.exists():
        LOG.error("Video file not found: %s", args.video_path)
        return 1

    video_basename = video_path.stem
    ball_csv = Path(args.output_dir) / video_basename / "ball.csv"
    tracks_dir = Path(args.output_dir) / video_basename / "tracks"

    python_exe = sys.executable

    if args.mode == "hub-track":
        if not args.api_key:
            LOG.error("Error: --api_key is required for hub-track mode")
            return 1
        try:
            from src.hub_inference import run_hub_inference
            run_hub_inference(
                video_path=args.video_path,
                model_url=args.hub_model,
                api_key=args.api_key,
                output_dir=args.output_dir,
                visualize=args.visualize,
            )
        except Exception as e:
            LOG.error("Hub inference failed: %s", e)
            return 1
        return 0

    if args.mode in ["detect", "all"]:
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
        if args.visualize:
            cmd.append("--visualize")
        run_command(cmd)

    if args.mode in ["track", "all"]:
        cmd = [
            python_exe,
            "src/track_calculator.py",
            "--csv_path",
            str(ball_csv),
            "--output_dir",
            args.output_dir,
            "--fps",
            str(args.fps),
        ]
        run_command(cmd)

    if args.mode in ["combined", "all"]:
        cmd = [
            python_exe,
            "src/track_processor.py",
            "--video_path",
            args.video_path,
            "--output_dir",
            args.output_dir,
        ]
        run_command(cmd)

    if args.mode in ["reels", "all"]:
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

    if args.mode == "pose":
        if not args.track_file:
            LOG.error("Error: --track_file is required for pose mode")
            return 1
        try:
            from src.pose_detector import add_pose_to_track_json
            add_pose_to_track_json(
                track_file=args.track_file,
                video_path=args.video_path,
                output_dir=args.output_dir,
                visualize=args.visualize,
            )
        except Exception as e:
            LOG.error("Pose detection failed: %s", e)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
