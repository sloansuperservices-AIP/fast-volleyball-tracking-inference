#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def run_command(command):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument("--mode", type=str, choices=["track", "pose", "analyze", "hub-track", "openvino-track"],
                        default="track", help="Processing mode")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    parser.add_argument("--model_path", type=str, default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
                        help="Path to model file (ONNX or OpenVINO XML)")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable visualization on display using cv2")
    parser.add_argument("--only_csv", action="store_true", help="Save only CSV, skip video output")
    
    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY", "5ea02b4238fc9528408b8c36dcdb3834e11a9cbf58"),
                        help="Ultralytics Hub API key")

    args = parser.parse_args()
    
    # Ensure sys.path includes src for internal script imports
    src_path = str(Path(__file__).parent / "src")
    if src_path not in sys.path:
        sys.path.append(src_path)

    if args.mode == "hub-track":
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

    elif args.mode == "track":
        if not args.video_path:
            print("Error: --video_path is required for tracking mode")
            return 1

        cmd = [sys.executable, "src/inference_onnx_seq_gray_v2.py",
               "--video_path", args.video_path,
               "--model_path", args.model_path,
               "--output_dir", args.output_dir]
        if args.visualize: cmd.append("--visualize")
        if args.only_csv: cmd.append("--only_csv")

        if not run_command(cmd): return 1
            
    elif args.mode == "openvino-track":
        if not args.video_path:
            print("Error: --video_path is required for openvino-track mode")
            return 1

        cmd = [sys.executable, "src/inference_openvino_seq_gray_v2.py",
               "--video_path", args.video_path,
               "--model_xml", args.model_path,
               "--output_dir", args.output_dir]
        if args.visualize: cmd.append("--visualize")
        if args.only_csv: cmd.append("--only_csv")

        if not run_command(cmd): return 1

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
        if not args.video_path:
            print("Error: --video_path is required for analyze mode to locate ball.csv")
            return 1

        video_name = Path(args.video_path).stem
        csv_path = Path(args.output_dir) / video_name / "ball.csv"

        if not csv_path.exists():
            print(f"Error: ball.csv not found at {csv_path}. Run 'track' mode first.")
            return 1

        cmd = [sys.executable, "src/track_calculator.py",
               "--csv_path", str(csv_path),
               "--output_dir", args.output_dir]

        if not run_command(cmd): return 1
        
    else:
        parser.print_help()
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
