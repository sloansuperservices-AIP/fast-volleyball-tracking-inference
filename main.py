#!/usr/bin/env python3
"""
Unified entry point for the fast volleyball tracking pipeline.
Supports 4 main steps: detect, track, combined, reels.
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


def run_command(cmd: list[str], verbose: bool = False) -> None:
    """Run a shell command and stream output."""
    LOG.debug("Running command: %s", " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    for line in process.stdout:
        if verbose:
            print(line, end="")
        else:
            # Show only high-level progress or errors in non-verbose mode
            if "Processing" in line or "Saved" in line or "Done" in line:
                print(line, end="")

    process.wait()
    if process.returncode != 0:
        LOG.error("Command failed with return code %s", process.returncode)
        sys.exit(process.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Fast Volleyball Ball Tracking -> Vertical Reels"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["detect", "track", "combined", "reels", "all", "pose", "hub-track", "analyze"],
        default="all",
        help="Processing mode (default: all)",
    )
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Root output directory"
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    parser.add_argument(
        "--visualize", action="store_true", help="Enable real-time visualization"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--only_csv", action="store_true", help="Save only CSV in detect mode")

    # Mode-specific arguments
    parser.add_argument("--track_file", type=str, help="Path to track JSON (for pose mode)")
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY"),
                        help="Ultralytics Hub API key")

    args = parser.parse_args()
    setup_logging(args.verbose)

    video_path = Path(args.video_path)
    video_basename = video_path.stem
    video_out_dir = Path(args.output_dir) / video_basename
    ball_csv = video_out_dir / "ball.csv"
    tracks_dir = video_out_dir / "tracks"

    python_exe = sys.executable

    # 1. Detection
    if args.mode in ["detect", "all"]:
        LOG.info("--- Step 1: Ball Detection ---")
        cmd = [
            python_exe,
            "src/inference_onnx_seq_gray_v2.py",
            "--video_path",
            str(video_path),
            "--model_path",
            args.model_path,
            "--output_dir",
            args.output_dir,
        ]
        if args.visualize:
            cmd.append("--visualize")
        if args.only_csv:
            cmd.append("--only_csv")
        if args.verbose:
            cmd.append("--verbose")
        run_command(cmd, args.verbose)

    # 2. Track Calculation
    if args.mode in ["track", "all"]:
        LOG.info("--- Step 2: Track Calculation ---")
        if not ball_csv.exists():
            LOG.error("ball.csv not found at %s. Run with --mode detect first.", ball_csv)
            return 1

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
        run_command(cmd, args.verbose)

    # 3. Horizontal Assembly
    if args.mode in ["combined", "all"]:
        LOG.info("--- Step 3: Horizontal Assembly ---")
        cmd = [
            python_exe,
            "src/track_processor.py",
            "--video_path",
            str(video_path),
            "--output_dir",
            args.output_dir,
        ]
        run_command(cmd, args.verbose)

    # 4. Vertical Reels
    if args.mode in ["reels", "all"]:
        LOG.info("--- Step 4: Vertical Reel Generation ---")
        if not tracks_dir.exists():
            LOG.error("Tracks directory not found at %s.", tracks_dir)
            return 1

        cmd = [
            python_exe,
            "src/make_reels.py",
            "--video_path",
            str(video_path),
            "--json_dir",
            str(tracks_dir),
            "--output_dir",
            args.output_dir,
        ]
        if args.visualize:
            cmd.append("--visualize")
        run_command(cmd, args.verbose)

    # Pose Detection (Independent step)
    if args.mode == "pose":
        LOG.info("--- Pose Detection ---")
        if not args.track_file:
            LOG.error("--track_file is required for pose mode")
            return 1
        cmd = [
            python_exe,
            "src/pose_detector.py",
            "--video_path",
            str(video_path),
            "--track_file",
            args.track_file,
            "--output_dir",
            args.output_dir,
        ]
        if args.visualize:
            cmd.append("--visualize")
        run_command(cmd, args.verbose)

    # Hub Tracking (Alternative Step 1)
    if args.mode == "hub-track":
        LOG.info("--- Hub Tracking ---")
        cmd = [
            python_exe,
            "src/hub_inference.py",
            "--video_path",
            str(video_path),
            "--hub_model",
            args.hub_model,
            "--output_dir",
            args.output_dir,
        ]
        if args.api_key:
            cmd.extend(["--api_key", args.api_key])
        if args.visualize:
            cmd.append("--visualize")
        run_command(cmd, args.verbose)

    if args.mode == "analyze":
        LOG.info("Analysis mode selected. This mode is not yet implemented.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
