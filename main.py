#!/usr/bin/env python3
"""
Unified entry point for the fast volleyball tracking inference system.
Orchestrates the 4-step pipeline: detect -> track -> combined -> reels.
"""

import argparse
import logging
import os
import subprocess
import sys
from typing import List

LOG = logging.getLogger(__name__)

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

def run_command(cmd: List[str]) -> bool:
    LOG.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        LOG.error("Command failed with exit code %s", exc.returncode)
        return False
    except FileNotFoundError:
        LOG.error("Command not found: %s", cmd[0])
        return False

def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference Pipeline")
    parser.add_argument("--mode", type=str,
                        choices=["detect", "track", "combined", "reels", "all", "pose", "analyze", "hub-track"],
                        default="all", help="Processing mode")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--model_path", type=str, default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--output_dir", type=str, default="output", help="Root directory for outputs")
    parser.add_argument("--visualize", action="store_true", help="Show live preview (where supported)")
    parser.add_argument("--court_json", type=str, help="Path to court keypoints JSON")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    
    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY"),
                        help="Ultralytics Hub API key")

    # Pose mode arguments
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.mode == "hub-track":
        if not args.api_key:
            LOG.error("ULTRALYTICS_HUB_API_KEY is required for hub-track mode")
            return 1
        if not args.video_path:
            LOG.error("--video_path is required")
            return 1

        from src.hub_inference import run_hub_inference
        run_hub_inference(
            video_path=args.video_path,
            model_url=args.hub_model,
            api_key=args.api_key,
            output_dir=args.output_dir,
            visualize=args.visualize
        )
        return 0

    if not args.video_path and args.mode != "analyze":
        parser.print_help()
        return 0

    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    video_out_dir = os.path.join(args.output_dir, video_basename)
    csv_path = os.path.join(video_out_dir, "ball.csv")
    json_dir = os.path.join(video_out_dir, "tracks")

    # Step 1: Detect
    if args.mode in ["detect", "all"]:
        cmd = [
            sys.executable, "src/inference_onnx_seq_gray_v2.py",
            "--video_path", args.video_path,
            "--model_path", args.model_path,
            "--output_dir", args.output_dir,
            "--only_csv"
        ]
        if args.visualize:
            cmd.remove("--only_csv")
            cmd.append("--visualize")
        if not run_command(cmd):
            return 1

    # Step 2: Track
    if args.mode in ["track", "all"]:
        cmd = [
            sys.executable, "src/track_calculator.py",
            "--csv_path", csv_path,
            "--output_dir", args.output_dir
        ]
        if args.court_json:
            cmd.extend(["--court_json_path", args.court_json])
        if not run_command(cmd):
            return 1

    # Step 3: Combined
    if args.mode in ["combined", "all"]:
        cmd = [
            sys.executable, "src/track_processor.py",
            "--video_path", args.video_path,
            "--output_dir", args.output_dir
        ]
        if not run_command(cmd):
            LOG.warning("Combined video step failed or no tracks found.")

    # Step 4: Reels
    if args.mode in ["reels", "all"]:
        cmd = [
            sys.executable, "src/make_reels.py",
            "--video_path", args.video_path,
            "--json_dir", json_dir,
            "--output_dir", args.output_dir
        ]
        if args.visualize:
            cmd.append("--visualize")
        if not run_command(cmd):
            LOG.warning("Reels generation failed.")

    if args.mode == "pose":
        if not args.track_file or not args.video_path:
            LOG.error("--track_file and --video_path are required for pose mode")
            return 1
        cmd = [
            sys.executable, "src/pose_detector.py",
            "--video_path", args.video_path,
            "--track_file", args.track_file,
            "--output_dir", args.output_dir
        ]
        if args.visualize:
            cmd.append("--visualize")
        if not run_command(cmd):
            return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
