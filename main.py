#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
"""

import argparse
import os
import sys
import subprocess

def run_cmd(cmd):
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument("--mode", type=str, choices=["detect", "track", "combined", "reels", "all", "pose", "hub-track"],
                        default="all", help="Processing mode")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model_path", type=str, default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable visualization on display using cv2")
    
    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY"),
                        help="Ultralytics Hub API key")

    # Track calculator specific
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second")

    # Pose specific
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")

    args = parser.parse_args()
    
    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    video_output_dir = os.path.join(args.output_dir, video_basename)
    ball_csv = os.path.join(video_output_dir, "ball.csv")
    tracks_dir = os.path.join(video_output_dir, "tracks")

    python_exe = sys.executable

    if args.mode == "hub-track":
        if not args.api_key:
            print("Error: --api_key or ULTRALYTICS_HUB_API_KEY environment variable is required for hub-track mode")
            return 1
        from src.hub_inference import run_hub_inference
        run_hub_inference(
            video_path=args.video_path,
            model_url=args.hub_model,
            api_key=args.api_key,
            output_dir=args.output_dir,
            visualize=args.visualize
        )

    elif args.mode == "detect" or args.mode == "all":
        cmd = [
            python_exe, "src/inference_onnx_seq_gray_v2.py",
            "--video_path", args.video_path,
            "--model_path", args.model_path,
            "--output_dir", args.output_dir,
            "--only_csv"
        ]
        if args.visualize:
            cmd.append("--visualize")
        run_cmd(cmd)

    if args.mode == "track" or args.mode == "all":
        cmd = [
            python_exe, "src/track_calculator.py",
            "--csv_path", ball_csv,
            "--output_dir", args.output_dir,
            "--fps", str(args.fps)
        ]
        run_cmd(cmd)

    if args.mode == "combined" or args.mode == "all":
        cmd = [
            python_exe, "src/track_processor.py",
            "--video_path", args.video_path,
            "--output_dir", args.output_dir
        ]
        run_cmd(cmd)

    if args.mode == "reels" or args.mode == "all":
        cmd = [
            python_exe, "src/make_reels.py",
            "--video_path", args.video_path,
            "--json_dir", tracks_dir,
            "--output_dir", args.output_dir
        ]
        if args.visualize:
            cmd.append("--visualize")
        run_cmd(cmd)

    if args.mode == "pose":
        if not args.track_file:
            print("Error: --track_file is required for pose mode")
            return 1
        from src.pose_detector import add_pose_to_track_json
        add_pose_to_track_json(
            track_file=args.track_file,
            video_path=args.video_path,
            output_dir=args.output_dir,
            visualize=args.visualize
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
