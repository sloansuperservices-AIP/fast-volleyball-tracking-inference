#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking pipeline.
Orchestrates detection, tracking, assembly, and reel generation.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_command(command):
    """Executes a command and validates its exit code."""
    print(f"Executing: {' '.join(command)}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"Error: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Unified Volleyball Tracking Pipeline")
    parser.add_argument("--mode", type=str, choices=["detect", "track", "combined", "reels", "all", "hub-track", "pose"],
                        default="all", help="Pipeline step to run")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model_path", type=str, default="models/VballNetV1_seq9_grayscale_204_h288_w512.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--output_dir", type=str, default="output", help="Root directory for outputs")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--only_csv", action="store_true", help="Detection step: save only CSV")

    # Pose mode specific
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    
    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY"),
                        help="Ultralytics Hub API key")

    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {args.video_path}")
        return 1

    video_stem = video_path.stem
    csv_path = Path(args.output_dir) / video_stem / "ball.csv"
    tracks_dir = Path(args.output_dir) / video_stem / "tracks"

    # Define steps
    def step_detect():
        cmd = [sys.executable, "src/inference_onnx_seq_gray_v2.py",
               "--video_path", str(video_path),
               "--model_path", args.model_path,
               "--output_dir", args.output_dir]
        if args.visualize: cmd.append("--visualize")
        if args.only_csv: cmd.append("--only_csv")
        run_command(cmd)

    def step_track():
        if not csv_path.exists():
            print(f"Error: CSV not found at {csv_path}. Run 'detect' mode first.")
            sys.exit(1)
        cmd = [sys.executable, "src/track_calculator.py",
               "--csv_path", str(csv_path),
               "--output_dir", args.output_dir,
               "--fps", str(args.fps)]
        run_command(cmd)

    def step_combined():
        cmd = [sys.executable, "src/track_processor.py",
               "--video_path", str(video_path),
               "--output_dir", args.output_dir]
        run_command(cmd)

    def step_reels():
        # Reels typically run per track JSON, here we run for the whole tracks dir if possible
        # Or we call a script that handles the directory
        if not tracks_dir.exists():
            print(f"Error: Tracks directory not found at {tracks_dir}. Run 'track' mode first.")
            sys.exit(1)
            
        json_files = list(tracks_dir.glob("*.json"))
        if not json_files:
            print(f"No track JSON files found in {tracks_dir}")
            return

        for json_file in json_files:
            cmd = [sys.executable, "src/make_reels.py",
                   "--video_path", str(video_path),
                   "--track_json", str(json_file),
                   "--output_dir", args.output_dir]
            if args.visualize: cmd.append("--visualize")
            run_command(cmd)

    # Execution logic
    if args.mode == "detect":
        step_detect()
    elif args.mode == "track":
        step_track()
    elif args.mode == "combined":
        step_combined()
    elif args.mode == "reels":
        step_reels()
    elif args.mode == "all":
        step_detect()
        step_track()
        step_combined()
        step_reels()
    elif args.mode == "hub-track":
        if not args.api_key:
            print("Error: API key is required for hub-track mode. Set ULTRALYTICS_HUB_API_KEY env var or use --api_key")
            return 1
        from src.hub_inference import run_hub_inference
        run_hub_inference(
            video_path=str(video_path),
            model_url=args.hub_model,
            api_key=args.api_key,
            output_dir=args.output_dir,
            visualize=args.visualize
        )
    elif args.mode == "pose":
        if not args.track_file:
            print("Error: --track_file is required for pose mode")
            return 1
        from src.pose_detector import add_pose_to_track_json
        add_pose_to_track_json(
            track_file=args.track_file,
            video_path=str(video_path),
            output_dir=args.output_dir,
            visualize=args.visualize
        )

    return 0

if __name__ == "__main__":
    sys.exit(main())
