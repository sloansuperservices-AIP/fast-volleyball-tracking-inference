#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
Coordinates the 4-step pipeline: detect -> track -> combined -> reels.
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


def run_command(cmd: list[str], description: str) -> None:
    LOG.info("Running %s...", description)
    LOG.debug("Command: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        LOG.error("%s failed with exit code %s", description, exc.returncode)
        sys.exit(exc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["detect", "track", "combined", "reels", "all", "pose", "hub-track", "analyze"],
        default="all",
        help="Processing mode (default: all)",
    )
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Directory to save output files"
    )
    parser.add_argument(
        "--court_json", type=str, help="Path to court coordinates JSON (optional)"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Enable visualization for applicable steps"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    # Mode-specific overrides
    parser.add_argument("--track_file", type=str, help="Path to track JSON (for pose mode)")
    parser.add_argument("--hub_model", type=str, help="Ultralytics Hub model ID/URL")
    parser.add_argument("--api_key", type=str, help="Ultralytics Hub API key")

    args = parser.parse_args()
    setup_logging(args.verbose)

    video_path = Path(args.video_path).resolve()
    video_basename = video_path.stem
    output_dir = Path(args.output_dir).resolve()
    video_out_dir = output_dir / video_basename
    ball_csv = video_out_dir / "ball.csv"
    tracks_dir = video_out_dir / "tracks"

    python_exe = sys.executable

    # 1. Detection
    if args.mode in ["detect", "all"]:
        cmd = [
            python_exe,
            "src/inference_onnx_seq_gray_v2.py",
            "--video_path",
            str(video_path),
            "--model_path",
            args.model_path,
            "--output_dir",
            str(output_dir),
            "--only_csv",
        ]
        if args.visualize:
            cmd.remove("--only_csv")
            cmd.append("--visualize")
        if args.verbose:
            cmd.append("--verbose")
        run_command(cmd, "ball detection")

    # 2. Track calculation
    if args.mode in ["track", "all"]:
        cmd = [
            python_exe,
            "src/track_calculator.py",
            "--csv_path",
            str(ball_csv),
            "--output_dir",
            str(output_dir),
        ]
        if args.court_json:
            cmd.extend(["--court_json_path", args.court_json])
        if args.verbose:
            cmd.append("--verbose")
        run_command(cmd, "track calculation")

    # 3. Horizontal assembly (Combined)
    if args.mode in ["combined", "all"]:
        cmd = [
            python_exe,
            "src/track_processor.py",
            "--video_path",
            str(video_path),
            "--output_dir",
            str(output_dir),
        ]
        if args.verbose:
            cmd.append("--verbose")
        run_command(cmd, "horizontal assembly")

    # 4. Vertical reels
    if args.mode in ["reels", "all"]:
        cmd = [
            python_exe,
            "src/make_reels.py",
            "--video_path",
            str(video_path),
            "--json_dir",
            str(tracks_dir),
            "--output_dir",
            str(output_dir),
        ]
        if args.visualize:
            cmd.append("--visualize")
        if args.verbose:
            cmd.append("--verbose")
        run_command(cmd, "vertical reel generation")

    # Specialty modes
    if args.mode == "pose":
        if not args.track_file:
            LOG.error("--track_file is required for pose mode")
            sys.exit(1)
        cmd = [
            python_exe,
            "src/pose_detector.py",
            "--track_file",
            args.track_file,
            "--video_path",
            str(video_path),
            "--output_dir",
            str(output_dir),
        ]
        if args.visualize:
            cmd.append("--visualize")
        # src/pose_detector.py doesn't have a verbose flag in its argparse yet
        run_command(cmd, "pose detection")

    if args.mode == "hub-track":
        cmd = [
            python_exe,
            "src/hub_inference.py",
            "--video_path",
            str(video_path),
            "--output_dir",
            str(output_dir),
        ]
        if args.hub_model:
            cmd.extend(["--model_url", args.hub_model])
        if args.api_key:
            cmd.extend(["--api_key", args.api_key])
        if args.visualize:
            cmd.append("--visualize")
        run_command(cmd, "Ultralytics Hub inference")

    if args.mode == "analyze":
        LOG.info("Analysis mode is currently a placeholder")

    LOG.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
