#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
"""

import argparse
import os
import sys
import subprocess

def run_command(command):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument("--mode", type=str, choices=["detect", "track", "combined", "reels", "all", "pose", "analyze", "hub-track"],
                        default="all", help="Processing mode")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    parser.add_argument("--model_path", type=str, default="models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable visualization on display using cv2")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    
    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY", ""),
                        help="Ultralytics Hub API key")

    args = parser.parse_args()
    
    if args.mode == "hub-track":
        # Hub tracking mode
        if not args.video_path:
            print("Error: --video_path is required for hub-track mode")
            return 1
        if not args.api_key:
            print("Error: --api_key is required for hub-track mode")
            return 1

        try:
            from src.hub_inference import run_hub_inference
            print("Hub tracking mode selected")
            print(f"Video: {args.video_path}")
            print(f"Model: {args.hub_model}")

            run_hub_inference(
                video_path=args.video_path,
                model_url=args.hub_model,
                api_key=args.api_key,
                output_dir=args.output_dir,
                visualize=args.visualize
            )
        except ImportError as e:
            print(f"Error importing hub inference module: {e}")
            return 1
        except Exception as e:
            print(f"Error during hub inference: {e}")
            return 1

    video_basename = os.path.splitext(os.path.basename(args.video_path))[0] if args.video_path else ""
    csv_path = os.path.join(args.output_dir, video_basename, "ball.csv")
    json_dir = os.path.join(args.output_dir, video_basename, "tracks")

    if args.mode in ["detect", "all"]:
        if not args.video_path:
            print("Error: --video_path is required")
            return 1

        cmd = [sys.executable, "src/inference_onnx_seq9_gray_v2.py",
               "--video_path", args.video_path,
               "--model_path", args.model_path,
               "--output_dir", args.output_dir,
               "--only_csv"]
        if not run_command(cmd): return 1

    if args.mode in ["track", "all"]:
        cmd = [sys.executable, "src/track_calculator.py",
               "--csv_path", csv_path,
               "--output_dir", args.output_dir,
               "--fps", str(args.fps)]
        if not run_command(cmd): return 1

    if args.mode in ["combined", "all"]:
        cmd = [sys.executable, "src/track_processor.py",
               "--video_path", args.video_path,
               "--output_dir", args.output_dir,
               "--fps", str(args.fps)]
        if not run_command(cmd): return 1

    if args.mode in ["reels", "all"]:
        cmd = [sys.executable, "src/make_reels.py",
               "--video_path", args.video_path,
               "--json_dir", json_dir,
               "--output_dir", args.output_dir]
        if not run_command(cmd): return 1

    if args.mode == "pose":
        # Pose detection mode
        if not args.track_file or not args.video_path:
            print("Error: --track_file and --video_path are required for pose mode")
            return 1
            
        # Import and run pose detection
        try:
            from src.pose_detector import add_pose_to_track_json
            print("Pose detection mode selected")
            print(f"Track file: {args.track_file}")
            print(f"Video: {args.video_path}")
            print(f"Visualize: {args.visualize}")
            
            add_pose_to_track_json(
                track_file=args.track_file,
                video_path=args.video_path,
                output_dir=args.output_dir,
                visualize=args.visualize
            )
        except ImportError as e:
            print(f"Error importing pose detection module: {e}")
            return 1
        except Exception as e:
            print(f"Error during pose detection: {e}")
            return 1
            
    elif args.mode == "analyze":
        # Analysis mode
        print("Analysis mode selected")
        print("This mode is not yet implemented")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())