#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
"""

import argparse
import os
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument("--mode", type=str, choices=["detect", "track", "combined", "reels", "pose", "analyze", "hub-track"],
                        default="detect", help="Processing mode")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    parser.add_argument("--model_path", type=str, default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable visualization on display using cv2")
    parser.add_argument("--only_csv", action="store_true",
                        help="Save only CSV, skip video output in detect mode")
    
    # Track calculator arguments
    parser.add_argument("--court_json_path", type=str, help="Path to court coordinates JSON file")

    # Reels arguments
    parser.add_argument("--smoothing", choices=["none", "moving_avg", "savitzky_golay", "kalman"],
                        default="moving_avg", help="Smoothing method for reels")

    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY", "5ea02b4238fc9528408b8c36dcdb3834e11a9cbf58"),
                        help="Ultralytics Hub API key")

    args = parser.parse_args()
    
    python_exe = sys.executable

    if args.mode == "detect":
        if not args.video_path:
            print("Error: --video_path is required for detect mode")
            return 1

        cmd = [
            python_exe, "src/inference_onnx_seq_gray_v2.py",
            "--video_path", args.video_path,
            "--model_path", args.model_path,
            "--output_dir", args.output_dir
        ]
        if args.visualize: cmd.append("--visualize")
        if args.only_csv: cmd.append("--only_csv")

        return subprocess.call(cmd)

    elif args.mode == "track":
        if not args.video_path:
            print("Error: --video_path is required to resolve output folder")
            return 1

        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        csv_path = os.path.join(args.output_dir, video_name, "ball.csv")

        cmd = [
            python_exe, "src/track_calculator.py",
            "--csv_path", csv_path,
            "--output_dir", args.output_dir
        ]
        if args.court_json_path:
            cmd.extend(["--court_json_path", args.court_json_path])

        return subprocess.call(cmd)

    elif args.mode == "combined":
        if not args.video_path:
            print("Error: --video_path is required")
            return 1

        cmd = [
            python_exe, "src/track_processor.py",
            "--video_path", args.video_path,
            "--output_dir", args.output_dir
        ]
        return subprocess.call(cmd)

    elif args.mode == "reels":
        if not args.video_path:
            print("Error: --video_path is required")
            return 1

        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        json_dir = os.path.join(args.output_dir, video_name, "tracks")

        cmd = [
            python_exe, "src/make_reels.py",
            "--video_path", args.video_path,
            "--json_dir", json_dir,
            "--output_dir", args.output_dir,
            "--smoothing", args.smoothing
        ]
        if args.visualize: cmd.append("--visualize")

        return subprocess.call(cmd)

    elif args.mode == "hub-track":
        if not args.video_path:
            print("Error: --video_path is required for hub-track mode")
            return 1

        try:
            from src.hub_inference import run_hub_inference
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

    elif args.mode == "pose":
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
            
    elif args.mode == "analyze":
        print("Analysis mode selected. This mode is not yet implemented.")
        
    else:
        parser.print_help()
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
