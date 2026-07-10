#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
"""

import argparse
import os
import sys
import subprocess

def run_command(args):
    """Run a command as a subprocess and return its exit code."""
    try:
        process = subprocess.run(args, check=True)
        return process.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return e.returncode

def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument("--mode", type=str, choices=["track", "pose", "analyze", "hub-track", "openvino-track"],
                        default="track", help="Processing mode")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    parser.add_argument("--model_path", type=str, default="models/VballNetGridV1b_seq9_grayscale_20260319_193937.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--model_xml", type=str, default="ov/VballNetGridV1b_seq9_grayscale_20260510_183219.xml",
                        help="Path to OpenVINO XML file")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable visualization on display using cv2")
    parser.add_argument("--only_csv", action="store_true", help="Only output CSV, no video")
    parser.add_argument("--rotate", type=int, default=0, choices=[-90, 0, 90, 180, -180],
                        help="Rotate frames before inference/output")
    parser.add_argument("--device", type=str, default="CPU", help="Device for OpenVINO (CPU, GPU, AUTO)")
    
    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY", ""),
                        help="Ultralytics Hub API key")

    # Analyze specific arguments
    parser.add_argument("--csv_path", type=str, help="Path to ball.csv for analysis")
    parser.add_argument("--court_json_path", type=str, help="Path to court.json for analysis")

    args = parser.parse_args()
    
    python_exe = sys.executable

    if args.mode == "hub-track":
        if not args.video_path:
            print("Error: --video_path is required for hub-track mode")
            return 1
        if not args.api_key:
            print("Error: ULTRALYTICS_HUB_API_KEY environment variable or --api_key is required")
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

        cmd = [python_exe, "src/inference_onnx_seq_gray_v2.py",
               "--video_path", args.video_path,
               "--model_path", args.model_path,
               "--output_dir", args.output_dir]
        if args.visualize:
            cmd.append("--visualize")
        if args.only_csv:
            cmd.append("--only_csv")
        if args.rotate != 0:
            cmd.extend(["--rotate", str(args.rotate)])

        print(f"Running ONNX tracking: {' '.join(cmd)}")
        return run_command(cmd)
            
    elif args.mode == "openvino-track":
        if not args.video_path:
            print("Error: --video_path is required for openvino-track mode")
            return 1

        cmd = [python_exe, "src/inference_openvino_seq_gray_v2.py",
               "--video_path", args.video_path,
               "--model_xml", args.model_xml,
               "--output_dir", args.output_dir,
               "--device", args.device]
        if args.visualize:
            cmd.append("--visualize")
        if args.only_csv:
            cmd.append("--only_csv")
        if args.rotate != 0:
            cmd.extend(["--rotate", str(args.rotate)])
            
        print(f"Running OpenVINO tracking: {' '.join(cmd)}")
        return run_command(cmd)

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
            print(f"Error during pose detection: {e}")
            return 1
            
    elif args.mode == "analyze":
        if not args.csv_path:
            print("Error: --csv_path is required for analyze mode")
            return 1
        
        cmd = [python_exe, "scripts/analyze_zone4_ball_trajectories.py",
               "--csv-path", args.csv_path]
        if args.court_json_path:
            cmd.extend(["--court-json-path", args.court_json_path])

        print(f"Running analysis: {' '.join(cmd)}")
        return run_command(cmd)
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
