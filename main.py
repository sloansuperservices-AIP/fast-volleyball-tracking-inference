#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
"""

import argparse
import os
import sys
import subprocess

def run_command(cmd, verbose=False):
    if verbose:
        print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument("--mode", type=str,
                        choices=["detect", "track", "combined", "reels", "all", "pose", "analyze", "hub-track"],
                        default="track", help="Processing mode")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    parser.add_argument("--model_path", type=str, default="models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx",
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
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY", "5ea02b4238fc9528408b8c36dcdb3834e11a9cbf58"),
                        help="Ultralytics Hub API key")

    args = parser.parse_args()
    
    video_basename = os.path.splitext(os.path.basename(args.video_path))[0] if args.video_path else ""
    video_out_dir = os.path.join(args.output_dir, video_basename) if video_basename else args.output_dir

    if args.mode == "detect" or args.mode == "all":
        if not args.video_path:
            print("Error: --video_path is required for detection")
            return 1

        cmd = [sys.executable, "src/inference_onnx_seq_gray_v2.py",
               "--video_path", args.video_path,
               "--model_path", args.model_path,
               "--output_dir", args.output_dir,
               "--only_csv"]
        if args.visualize:
            cmd.append("--visualize")

        print("--- Step 1: Ball Detection ---")
        run_command(cmd, args.verbose)

    if args.mode == "track" or args.mode == "all":
        if not args.video_path:
            print("Error: --video_path is required for tracking")
            return 1

        csv_path = os.path.join(video_out_dir, "ball.csv")
        cmd = [sys.executable, "src/track_calculator.py",
               "--csv_path", csv_path,
               "--output_dir", args.output_dir]

        print("--- Step 2: Track Calculation ---")
        run_command(cmd, args.verbose)

    if args.mode == "combined" or args.mode == "all":
        if not args.video_path:
            print("Error: --video_path is required for combined video generation")
            return 1

        cmd = [sys.executable, "src/track_processor.py",
               "--video_path", args.video_path,
               "--output_dir", args.output_dir]

        print("--- Step 3: Combined Video Generation ---")
        run_command(cmd, args.verbose)

    if args.mode == "reels" or args.mode == "all":
        if not args.video_path:
            print("Error: --video_path is required for reels generation")
            return 1

        json_dir = os.path.join(video_out_dir, "tracks")
        cmd = [sys.executable, "src/make_reels.py",
               "--video_path", args.video_path,
               "--json_dir", json_dir,
               "--output_dir", args.output_dir]
        if args.visualize:
            cmd.append("--visualize")

        print("--- Step 4: Vertical Reels Generation ---")
        run_command(cmd, args.verbose)

    if args.mode == "hub-track":
        # Hub tracking mode
        if not args.video_path:
            print("Error: --video_path is required for hub-track mode")
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

    elif args.mode == "pose":
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
        
    else:
        print("Hello from fast-volleyball-tracking-inference!")
        print("Use --mode to specify the processing mode")
        print("Available modes: track, pose, analyze")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())