#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
Supports a 4-step pipeline: detection, tracking, assembly, and reel generation.
"""

import argparse
import os
import sys
import subprocess

def run_command(command, verbose=False):
    if verbose:
        print(f"Running: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument("--mode", type=str, choices=["detect", "track", "combined", "reels", "all", "pose", "analyze", "hub-track"],
                        default="track", help="Processing mode")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--model_path", type=str, help="Path to ONNX model file")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--only_csv", action="store_true", help="Save only CSV, skip video output")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Pose specific
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    
    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY", "5ea02b4238fc9528408b8c36dcdb3834e11a9cbf58"),
                        help="Ultralytics Hub API key")

    args = parser.parse_args()
    
    # Define a helper to run specific modes
    def run_mode(mode_name):
        if mode_name == "hub-track":
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

        elif mode_name == "detect":
            if not args.video_path or not args.model_path:
                print("Error: --video_path and --model_path are required for detect mode")
                return 1
            
            cmd = [sys.executable, "src/inference_onnx_seq_gray_v2.py",
                   "--video_path", args.video_path,
                   "--model_path", args.model_path,
                   "--output_dir", args.output_dir]
            if args.visualize: cmd.append("--visualize")
            if args.only_csv: cmd.append("--only_csv")
            if args.verbose: cmd.append("--verbose")
            run_command(cmd, args.verbose)

        elif mode_name == "track":
            if not args.video_path:
                print("Error: --video_path is required to resolve standard CSV path for tracking")
                return 1
            
            video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
            csv_path = os.path.join(args.output_dir, video_basename, "ball.csv")
            
            cmd = [sys.executable, "src/track_calculator.py",
                   "--csv_path", csv_path,
                   "--output_dir", args.output_dir]
            if args.verbose: cmd.append("--verbose")
            run_command(cmd, args.verbose)

        elif mode_name == "combined":
            if not args.video_path:
                print("Error: --video_path is required for combined mode")
                return 1
            cmd = [sys.executable, "src/track_processor.py",
                   "--video_path", args.video_path,
                   "--output_dir", args.output_dir]
            if args.verbose: cmd.append("--verbose")
            run_command(cmd, args.verbose)

        elif mode_name == "reels":
            if not args.video_path:
                print("Error: --video_path is required for reels mode")
                return 1
            video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
            json_dir = os.path.join(args.output_dir, video_basename, "tracks")
            cmd = [sys.executable, "src/make_reels.py",
                   "--video_path", args.video_path,
                   "--json_dir", json_dir,
                   "--output_dir", args.output_dir]
            if args.verbose: cmd.append("--verbose")
            run_command(cmd, args.verbose)

        elif mode_name == "pose":
            if not args.track_file or not args.video_path:
                print("Error: --track_file and --video_path are required for pose mode")
                return 1

            try:
                from src.pose_detector import add_pose_to_track_json
                add_pose_to_track_json(
                    track_file=args.track_file,
                    video_path=args.video_path,
                    output_dir=args.output_dir,
                    visualize=args.visualize
                )
            except Exception as e:
                print(f"Error during pose detection: {e}")
                return 1

        elif mode_name == "analyze":
            print("Analysis mode selected - not yet implemented")
        
        return 0

    if args.mode == "all":
        # Chained execution of the 4-step pipeline
        for stage in ["detect", "track", "combined", "reels"]:
            print(f"\n>>> Starting Stage: {stage}")
            ret = run_mode(stage)
            if ret != 0:
                print(f"Pipeline failed at stage {stage}")
                return ret
        return 0
    else:
        return run_mode(args.mode)

if __name__ == "__main__":
    sys.exit(main())
