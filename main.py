#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
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
    LOG.debug("Running command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["detect", "track", "combined", "reels", "all", "pose", "hub-track", "analyze"],
        default="all",
        help="Processing mode",
    )
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/VballNetV1b_seq9_grayscale_best.onnx",
        help="Path to ball detection model file (.onnx or .xml)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Directory to save output files"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Enable visualization on display"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    # Track calculator arguments
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    parser.add_argument("--court_json", type=str, help="Path to court coordinates JSON")

    # Hub specific arguments
    parser.add_argument(
        "--hub_model",
        type=str,
        default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
        help="Ultralytics Hub model URL or ID",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("ULTRALYTICS_HUB_API_KEY"),
        help="Ultralytics Hub API key",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.video_path:
        LOG.error("--video_path is required")
        return 1

    video_path = Path(args.video_path)
    video_name = video_path.stem
    output_dir = Path(args.output_dir)
    video_output_dir = output_dir / video_name
    ball_csv = video_output_dir / "ball.csv"

    python_exe = sys.executable

    try:
        # Step 1: Detection
        if args.mode in ["detect", "all"]:
            LOG.info("Step 1/4: Ball detection...")
            detect_script = (
                "src/inference_openvino_seq_gray_v2.py"
                if args.model_path.endswith(".xml")
                else "src/inference_onnx_seq_gray_v2.py"
            )
            cmd = [
                python_exe,
                detect_script,
                "--video_path",
                str(video_path),
                "--model_path" if detect_script.endswith("onnx_seq_gray_v2.py") else "--model_xml",
                args.model_path,
                "--output_dir",
                args.output_dir,
                "--only_csv",
            ]
            if args.visualize:
                cmd.append("--visualize")
            if args.verbose:
                cmd.append("--verbose")
            run_command(cmd)

        # Step 2: Track Calculation
        if args.mode in ["track", "all"]:
            LOG.info("Step 2/4: Track calculation...")
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
            if args.court_json:
                cmd.extend(["--court_json_path", args.court_json])
            if args.verbose:
                cmd.append("--verbose")
            run_command(cmd)

        # Step 3: Rally Assembly (Combined video)
        if args.mode in ["combined", "all"]:
            LOG.info("Step 3/4: Assembling combined video...")
            cmd = [
                python_exe,
                "src/track_processor.py",
                "--video_path",
                str(video_path),
                "--output_dir",
                args.output_dir,
            ]
            if args.verbose:
                cmd.append("--verbose")
            run_command(cmd)

        # Step 4: Vertical Reels
        if args.mode in ["reels", "all"]:
            LOG.info("Step 4/4: Generating vertical reels...")
            cmd = [
                python_exe,
                "src/make_reels.py",
                "--video_path",
                str(video_path),
                "--json_dir",
                str(video_output_dir / "tracks"),
                "--output_dir",
                args.output_dir,
            ]
            if args.verbose:
                cmd.append("--verbose")
            run_command(cmd)

        # Additional modes
        if args.mode == "pose":
            LOG.info("Running pose detection...")
            cmd = [
                python_exe,
                "src/pose_detector.py",
                "--video_path",
                str(video_path),
                "--output_dir",
                args.output_dir,
            ]
            if args.visualize:
                cmd.append("--visualize")
            run_command(cmd)

        if args.mode == "hub-track":
            LOG.info("Running Ultralytics Hub inference...")
            cmd = [
                python_exe,
                "src/hub_inference.py",
                "--video_path",
                str(video_path),
                "--model_url",
                args.hub_model,
                "--output_dir",
                args.output_dir,
            ]
            if args.api_key:
                cmd.extend(["--api_key", args.api_key])
            if args.visualize:
                cmd.append("--visualize")
            run_command(cmd)

        if args.mode == "analyze":
            LOG.info("Analysis mode is a placeholder.")

    except subprocess.CalledProcessError as e:
        LOG.error("Pipeline failed at step %s", args.mode)
        return e.returncode

    LOG.info("Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
