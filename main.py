#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
Supports 4-step pipeline: detect, track, combined, reels.
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
    LOG.debug("Running: %s", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        LOG.error("Command failed: %s", e)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["detect", "track", "combined", "reels", "all", "pose", "hub-track", "analyze"],
        default="all",
        help="Processing mode (all = detect + track + combined + reels)",
    )
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.5, help="Heatmap threshold"
    )
    parser.add_argument("--court_json", type=str, help="Optional court JSON for filtering")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")

    # Pose/Hub specific
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

    video_name = Path(args.video_path).stem
    video_output_root = os.path.join(args.output_dir, video_name)
    ball_csv = os.path.join(video_output_root, "ball.csv")
    tracks_dir = os.path.join(video_output_root, "tracks")

    modes = [args.mode]
    if args.mode == "all":
        modes = ["detect", "track", "combined", "reels"]

    for mode in modes:
        LOG.info("Starting mode: %s", mode)

        if mode == "detect":
            cmd = [
                sys.executable,
                "src/inference_onnx_seq_gray_v2.py",
                "--video_path",
                args.video_path,
                "--model_path",
                args.model_path,
                "--output_dir",
                args.output_dir,
                "--confidence_threshold",
                str(args.confidence_threshold),
            ]
            if args.visualize:
                cmd.append("--visualize")
            if args.verbose:
                cmd.append("--verbose")
            # Always use only_csv for intermediate detection
            cmd.append("--only_csv")
            run_command(cmd)

        elif mode == "track":
            cmd = [
                sys.executable,
                "src/track_calculator.py",
                "--csv_path",
                ball_csv,
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

        elif mode == "combined":
            cmd = [
                sys.executable,
                "src/track_processor.py",
                "--video_path",
                args.video_path,
                "--output_dir",
                args.output_dir,
            ]
            if args.verbose:
                cmd.append("--verbose")
            run_command(cmd)

        elif mode == "reels":
            cmd = [
                sys.executable,
                "src/make_reels.py",
                "--video_path",
                args.video_path,
                "--json_dir",
                tracks_dir,
                "--output_dir",
                args.output_dir,
            ]
            if args.visualize:
                cmd.append("--visualize")
            run_command(cmd)

        elif mode == "pose":
            if not args.track_file:
                LOG.error("--track_file is required for pose mode")
                sys.exit(1)
            cmd = [
                sys.executable,
                "src/pose_detector.py",
                "--video_path",
                args.video_path,
                "--track_file",
                args.track_file,
                "--output_dir",
                args.output_dir,
            ]
            if args.visualize:
                cmd.append("--visualize")
            run_command(cmd)

        elif mode == "hub-track":
            cmd = [
                sys.executable,
                "src/hub_inference.py",
                "--video_path",
                args.video_path,
                "--hub_model",
                args.hub_model,
                "--output_dir",
                args.output_dir,
            ]
            if args.api_key:
                cmd.extend(["--api_key", args.api_key])
            if args.visualize:
                cmd.append("--visualize")
            run_command(cmd)

        elif mode == "analyze":
            LOG.info("Analysis mode is currently integrated into 'track' via statistics.")

    LOG.info("Pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
