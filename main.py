#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
Orchestrates the pipeline from detection to track calculation and visualization.
"""

import argparse
import os
import sys
import subprocess
import logging

LOG = logging.getLogger(__name__)

def run_command(command):
    """Helper to run a shell command and exit on failure."""
    print(f"🚀 Running: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument("--mode", type=str, choices=["track", "pose", "analyze", "hub-track"],
                        default="track", help="Processing mode")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    parser.add_argument("--model_path", type=str,
                        default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable visualization on display using cv2")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY", ""),
                        help="Ultralytics Hub API key")

    args = parser.parse_args()
    
    if args.mode == "hub-track":
        if not args.video_path:
            print("Error: --video_path is required for hub-track mode")
            return 1

        try:
            from src.hub_inference import run_hub_inference
            print("Hub tracking mode selected")
            run_hub_inference(
                video_path=args.video_path,
                model_url=args.hub_model,
                api_key=args.api_key,
                output_dir=args.output_dir,
                visualize=args.visualize
            )
        except Exception as e:
            print(f"Error during hub inference: {e}")
            return 1

    elif args.mode == "track":
        if not args.video_path:
            print("Error: --video_path is required for tracking mode")
            return 1
            
        print(f"--- Step 1: Detection using {args.model_path} ---")
        detect_cmd = [
            "python3", "src/inference_onnx_seq_gray_v2.py",
            "--video_path", args.video_path,
            "--model_path", args.model_path,
            "--output_dir", args.output_dir,
            "--only_csv"
        ]
        if args.visualize:
            detect_cmd.remove("--only_csv")
            detect_cmd.append("--visualize")
        if args.verbose:
            detect_cmd.append("--verbose")
        run_command(detect_cmd)

        video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
        csv_path = os.path.join(args.output_dir, video_basename, "ball.csv")

        print(f"--- Step 2: Track calculation from {csv_path} ---")
        track_cmd = [
            "python3", "src/track_calculator.py",
            "--csv_path", csv_path,
            "--output_dir", args.output_dir
        ]
        if args.verbose:
            track_cmd.append("--verbose")
        run_command(track_cmd)

        print(f"✅ Pipeline complete. Tracks saved in {os.path.join(args.output_dir, video_basename, 'tracks')}")

    elif args.mode == "pose":
        if not args.track_file or not args.video_path:
            print("Error: --track_file and --video_path are required for pose mode")
            return 1
            
        try:
            from src.pose_detector import add_pose_to_track_json
            print("Pose detection mode selected")
            add_pose_to_track_json(
                track_file=args.track_file,
                video_path=args.video_path,
                output_dir=args.output_dir,
                visualize=args.visualize
            )
        except Exception as e:
            print(f"Error during_pose detection: {e}")
            return 1
            
    elif args.mode == "analyze":
        print("Analysis mode selected")
        if not args.video_path:
            print("Error: --video_path is required for analyze mode")
            return 1
        
        # Example of calling a script from the new scripts/ directory
        analyze_cmd = [
            "python3", "scripts/analyze_zone4_ball_trajectories.py",
            "--video_path", args.video_path
        ]
        run_command(analyze_cmd)
        
    return 0


if __name__ == "__main__":
    main()
