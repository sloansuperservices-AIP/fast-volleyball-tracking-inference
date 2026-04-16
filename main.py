#!/usr/bin/env python3
"""
Unified entry point for the volleyball tracking pipeline.
Standardized 4-step processing:
1. Detect (ball.csv)
2. Track (track_*.json)
3. Combined (combined.mp4 or individual horizontal clips)
4. Reels (vertical 9:16 reels)
"""

import argparse
import os
import sys
import subprocess
from src.constants import DEFAULT_FPS, DEFAULT_MAX_DISTANCE

def run_command(command):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Volleyball Tracking Pipeline")
    parser.add_argument("--mode", type=str,
                        choices=["detect", "track", "combined", "reels", "all", "pose", "hub-track"],
                        default="all", help="Processing mode")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model_path", type=str, default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Video FPS")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Pose specific
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")

    # Hub specific
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY"),
                        help="Ultralytics Hub API key")

    args = parser.parse_args()
    
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    video_out_dir = os.path.join(args.output_dir, video_name)
    csv_path = os.path.join(video_out_dir, "ball.csv")
    json_dir = os.path.join(video_out_dir, "tracks")

    if args.mode in ["detect", "all"]:
        print("\n--- Step 1: Ball Detection ---")
        cmd = [
            sys.executable, "src/inference_onnx_seq_gray_v2.py",
            "--video_path", args.video_path,
            "--model_path", args.model_path,
            "--output_dir", args.output_dir,
            "--only_csv"
        ]
        if args.visualize: cmd.append("--visualize")
        if args.verbose: cmd.append("--verbose")
        run_command(cmd)

    if args.mode in ["track", "all"]:
        print("\n--- Step 2: Track Calculation ---")
        cmd = [
            sys.executable, "src/track_calculator.py",
            "--csv_path", csv_path,
            "--output_dir", args.output_dir,
            "--fps", str(args.fps)
        ]
        run_command(cmd)

    if args.mode in ["combined", "all"]:
        print("\n--- Step 3: Horizontal Assembly ---")
        cmd = [
            sys.executable, "src/track_processor.py",
            "--video_path", args.video_path,
            "--output_dir", args.output_dir,
            "--fps", str(args.fps)
        ]
        run_command(cmd)

    if args.mode in ["reels", "all"]:
        print("\n--- Step 4: Vertical Reel Generation ---")
        cmd = [
            sys.executable, "src/make_reels.py",
            "--video_path", args.video_path,
            "--json_dir", json_dir,
            "--output_dir", args.output_dir
        ]
        if args.visualize: cmd.append("--visualize")
        run_command(cmd)

    if args.mode == "pose":
        if not args.track_file:
            print("Error: --track_file is required for pose mode")
            return 1
        cmd = [
            sys.executable, "src/pose_detector.py",
            "--video_path", args.video_path,
            "--track_file", args.track_file,
            "--output_dir", args.output_dir
        ]
        if args.visualize: cmd.append("--visualize")
        run_command(cmd)

    if args.mode == "hub-track":
        if not args.api_key:
            print("Error: --api_key or ULTRALYTICS_HUB_API_KEY is required for hub-track mode")
            return 1
        cmd = [
            sys.executable, "src/hub_inference.py",
            "--video_path", args.video_path,
            "--model_url", args.hub_model,
            "--api_key", args.api_key,
            "--output_dir", args.output_dir
        ]
        if args.visualize: cmd.append("--visualize")
        run_command(cmd)

    return 0

if __name__ == "__main__":
    sys.exit(main())
