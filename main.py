#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
"""

import argparse
import os
import sys
import subprocess


def run_command(cmd_list):
    """Run a command using sys.executable and subprocess."""
    print(f"Executing: {' '.join(cmd_list)}")
    try:
        subprocess.check_call(cmd_list)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["detect", "track", "combined", "reels", "all", "pose", "hub-track", "analyze"],
        default="all",
        help="Processing mode (default: all)",
    )
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model_path", type=str, default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true",
                        help="Enable visualization on display")
    parser.add_argument("--only_csv", action="store_true",
                        help="Save only CSV during detection")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY"),
                        help="Ultralytics Hub API key")

    # Pose specific arguments
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")

    args = parser.parse_args()
    
    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    video_output_dir = os.path.join(args.output_dir, video_basename)
    csv_path = os.path.join(video_output_dir, "ball.csv")
    json_dir = os.path.join(video_output_dir, "tracks")

    modes_to_run = []
    if args.mode == "all":
        modes_to_run = ["detect", "track", "combined", "reels"]
    else:
        modes_to_run = [args.mode]

    for mode in modes_to_run:
        print(f"\n--- Starting mode: {mode} ---")
        
        if mode == "detect":
            cmd = [sys.executable, "src/inference_onnx_seq_gray_v2.py",
                   "--video_path", args.video_path,
                   "--model_path", args.model_path,
                   "--output_dir", args.output_dir]
            if args.visualize:
                cmd.append("--visualize")
            if args.only_csv:
                cmd.append("--only_csv")
            if args.verbose:
                cmd.append("--verbose")
            run_command(cmd)

        elif mode == "track":
            cmd = [sys.executable, "src/track_calculator.py",
                   "--csv_path", csv_path,
                   "--output_dir", args.output_dir]
            if args.verbose:
                cmd.append("--verbose")
            run_command(cmd)

        elif mode == "combined":
            cmd = [sys.executable, "src/track_processor.py",
                   "--video_path", args.video_path,
                   "--output_dir", args.output_dir]
            if args.verbose:
                cmd.append("--verbose")
            run_command(cmd)

        elif mode == "reels":
            cmd = [sys.executable, "src/make_reels.py",
                   "--video_path", args.video_path,
                   "--json_dir", json_dir,
                   "--output_dir", args.output_dir]
            if args.visualize:
                cmd.append("--visualize")
            run_command(cmd)

        elif mode == "pose":
            if not args.track_file:
                print("Error: --track_file is required for pose mode")
                return 1
            cmd = [sys.executable, "src/pose_detector.py",
                   "--video_path", args.video_path,
                   "--track_file", args.track_file,
                   "--output_dir", args.output_dir]
            if args.visualize:
                cmd.append("--visualize")
            run_command(cmd)

        elif mode == "hub-track":
            cmd = [sys.executable, "src/hub_inference.py",
                   "--video_path", args.video_path,
                   "--model_url", args.hub_model,
                   "--output_dir", args.output_dir]
            if args.api_key:
                cmd.extend(["--api_key", args.api_key])
            if args.visualize:
                cmd.append("--visualize")
            run_command(cmd)

        elif mode == "analyze":
            print("Analysis mode is not yet implemented")

    print("\n--- Pipeline execution finished ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
