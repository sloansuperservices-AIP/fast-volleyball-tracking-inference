#!/usr/bin/env python3
"""
Standardized 4-step pipeline orchestration for fast-volleyball-tracking-inference.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, verbose=False):
    """Executes a shell command and validates its exit code."""
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        if not verbose:
            print(result.stderr)
        sys.exit(result.returncode)
    return result

def main():
    parser = argparse.ArgumentParser(description="Unified Volleyball Tracking Pipeline")
    parser.add_argument("--mode", type=str, choices=["detect", "track", "combined", "reels", "all", "pose", "hub-track"],
                        default="all", help="Stage of the pipeline to run")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model_path", type=str, default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Root directory for results")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--visualize", action="store_true", help="Enable UI visualization")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    
    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY", ""),
                        help="Ultralytics Hub API key")

    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    video_name = video_path.stem
    video_out_dir = Path(args.output_dir) / video_name
    ball_csv = video_out_dir / "ball.csv"
    tracks_dir = video_out_dir / "tracks"

    python_exe = sys.executable

    # 1. Detection
    if args.mode in ["detect", "all"]:
        print(f"--- [1/4] Detection: {video_name} ---")
        cmd = [python_exe, "src/inference_onnx_seq_gray_v2.py",
               "--video_path", str(video_path),
               "--model_path", args.model_path,
               "--output_dir", args.output_dir,
               "--only_csv"]
        if args.visualize: cmd.append("--visualize")
        if args.verbose: cmd.append("--verbose")
        run_command(cmd, args.verbose)

    # 2. Track Calculation
    if args.mode in ["track", "all"]:
        print(f"--- [2/4] Track Calculation: {video_name} ---")
        cmd = [python_exe, "src/track_calculator.py",
               "--csv_path", str(ball_csv),
               "--output_dir", args.output_dir,
               "--fps", str(args.fps)]
        if args.verbose: cmd.append("--verbose")
        run_command(cmd, args.verbose)

    # 3. Combined Video
    if args.mode in ["combined", "all"]:
        print(f"--- [3/4] Rally Assembly: {video_name} ---")
        cmd = [python_exe, "src/track_processor.py",
               "--video_path", str(video_path),
               "--output_dir", args.output_dir]
        if args.verbose: cmd.append("--verbose")
        run_command(cmd, args.verbose)

    # 4. Vertical Reels
    if args.mode in ["reels", "all"]:
        print(f"--- [4/4] Vertical Reels: {video_name} ---")
        cmd = [python_exe, "src/make_reels.py",
               "--video_path", str(video_path),
               "--json_dir", str(tracks_dir),
               "--output_dir", args.output_dir]
        if args.visualize: cmd.append("--visualize")
        run_command(cmd, args.verbose)
        
    # Pose Mode
    if args.mode == "pose":
        print(f"--- Pose Detection: {video_name} ---")
        # Find first track if not specified? For now, we assume user might want to run on a specific file,
        # but following unified pattern, we could loop through tracks.
        if tracks_dir.exists():
            for track_file in tracks_dir.glob("track_*.json"):
                print(f"Processing pose for {track_file.name}")
                cmd = [python_exe, "src/pose_detector.py",
                       "--video_path", str(video_path),
                       "--track_file", str(track_file),
                       "--output_dir", args.output_dir]
                if args.visualize: cmd.append("--visualize")
                run_command(cmd, args.verbose)

    # Hub Tracking Mode
    if args.mode == "hub-track":
        print(f"--- Hub Tracking: {video_name} ---")
        cmd = [python_exe, "src/hub_inference.py",
               "--video_path", str(video_path),
               "--hub_model", args.hub_model,
               "--api_key", args.api_key,
               "--output_dir", args.output_dir]
        if args.visualize: cmd.append("--visualize")
        run_command(cmd, args.verbose)

    print(f"Pipeline completed for {video_name} in mode: {args.mode}")

if __name__ == "__main__":
    main()
