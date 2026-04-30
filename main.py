#!/usr/bin/env python3
"""
Standardized 4-step processing pipeline for volleyball tracking.
Orchestrates detection, track calculation, video processing, and reel generation.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Use sys.executable to ensure we use the same Python interpreter for subprocesses
PYTHON = sys.executable

def run_command(command, verbose=False):
    """Helper to run a shell command and handle errors."""
    if verbose:
        print(f"Running: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}: {' '.join(command)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Unified Volleyball Tracking Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["detect", "track", "combined", "reels", "all", "pose", "hub-track"],
        default="all",
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
        help="Path to ball detection ONNX model",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Root directory for outputs"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Frames per second for processing"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Enable visualization during processing"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )
    
    # Hub specific
    parser.add_argument("--hub_model", type=str, help="Ultralytics Hub model ID")
    parser.add_argument("--api_key", type=str, default=os.getenv("ULTRALYTICS_HUB_API_KEY"), help="Hub API key")

    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    video_basename = video_path.stem
    video_output_dir = Path(args.output_dir) / video_basename

    if args.mode == "detect" or args.mode == "all":
        print("\n--- STEP 1: Ball Detection ---")
        cmd = [
            PYTHON, "src/inference_onnx_seq_gray_v2.py",
            "--video_path", str(video_path),
            "--model_path", args.model_path,
            "--output_dir", args.output_dir,
            "--only_csv"
        ]
        if args.visualize: cmd.append("--visualize")
        if args.verbose: cmd.append("--verbose")
        if not run_command(cmd, args.verbose): return 1

    if args.mode == "track" or args.mode == "all":
        print("\n--- STEP 2: Track Calculation ---")
        csv_path = video_output_dir / "ball.csv"
        cmd = [
            PYTHON, "src/track_calculator.py",
            "--csv_path", str(csv_path),
            "--output_dir", args.output_dir,
            "--fps", str(args.fps)
        ]
        if not run_command(cmd, args.verbose): return 1

    if args.mode == "combined" or args.mode == "all":
        print("\n--- STEP 3: Combined Video Processing ---")
        cmd = [
            PYTHON, "src/track_processor.py",
            "--video_path", str(video_path),
            "--output_dir", args.output_dir
        ]
        if args.verbose: cmd.append("--verbose")
        if not run_command(cmd, args.verbose): return 1

    if args.mode == "reels" or args.mode == "all":
        print("\n--- STEP 4: 9:16 Reel Generation ---")
        json_dir = video_output_dir / "tracks"
        cmd = [
            PYTHON, "src/make_reels.py",
            "--video_path", str(video_path),
            "--json_dir", str(json_dir),
            "--output_dir", args.output_dir
        ]
        if args.visualize: cmd.append("--visualize")
        if not run_command(cmd, args.verbose): return 1
        
    if args.mode == "pose":
        print("\n--- Pose Analysis ---")
        cmd = [
            PYTHON, "src/pose_detector.py",
            "--video_path", str(video_path),
            "--output_dir", args.output_dir
        ]
        if args.visualize: cmd.append("--visualize")
        if not run_command(cmd, args.verbose): return 1
        
    if args.mode == "hub-track":
        print("\n--- Ultralytics Hub Tracking ---")
        if not args.hub_model:
            print("Error: --hub_model is required for hub-track mode")
            return 1
        cmd = [
            PYTHON, "src/hub_inference.py",
            "--video_path", str(video_path),
            "--model_url", args.hub_model,
            "--output_dir", args.output_dir
        ]
        if args.api_key:
            cmd.extend(["--api_key", args.api_key])
        if args.visualize: cmd.append("--visualize")
        if not run_command(cmd, args.verbose): return 1

    print("\nProcessing complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
