#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
"""

import argparse
import os
import sys
import subprocess

def run_command(command):
    """Run a shell command and return its exit code."""
    try:
        result = subprocess.run(command, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return e.returncode

def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument("--mode", type=str, choices=["track", "pose", "analyze", "hub-track", "openvino-track"],
                        default="track", help="Processing mode")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    parser.add_argument("--model_path", type=str, default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--model_xml", type=str, default="ov/VballNetV2_seq9_grayscale_ov.xml",
                        help="Path to OpenVINO XML file")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable visualization on display using cv2")
    
    # Analysis specific arguments
    parser.add_argument("--csv_path", type=str, help="Path to ball.csv for analysis")
    parser.add_argument("--court_json_path", type=str, help="Path to court.json for analysis")

    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY"),
                        help="Ultralytics Hub API key")

    args = parser.parse_args()
    
    if args.mode == "hub-track":
        if not args.video_path:
            print("Error: --video_path is required for hub-track mode")
            return 1

        from src.hub_inference import run_hub_inference
        run_hub_inference(
            video_path=args.video_path,
            model_url=args.hub_model,
            api_key=args.api_key,
            output_dir=args.output_dir,
            visualize=args.visualize
        )

    elif args.mode == "openvino-track":
        if not args.video_path:
            print("Error: --video_path is required for openvino-track mode")
            return 1

        cmd = [
            "uv", "run", "src/inference_openvino_seq_gray_v2.py",
            "--video_path", args.video_path,
            "--model_xml", args.model_xml,
            "--output_dir", args.output_dir
        ]
        if args.visualize:
            cmd.append("--visualize")
        else:
            cmd.append("--only_csv")

        return run_command(cmd)

    elif args.mode == "track":
        if not args.video_path:
            print("Error: --video_path is required for tracking mode")
            return 1
            
        cmd = [
            "uv", "run", "src/inference_onnx_seq_gray_v2.py",
            "--video_path", args.video_path,
            "--model_path", args.model_path,
            "--output_dir", args.output_dir
        ]
        if args.visualize:
            cmd.append("--visualize")
        else:
            cmd.append("--only_csv")

        return run_command(cmd)
            
    elif args.mode == "pose":
        if not args.track_file or not args.video_path:
            print("Error: --track_file and --video_path are required for pose mode")
            return 1
            
        from src.pose_detector import add_pose_to_track_json
        add_pose_to_track_json(
            track_file=args.track_file,
            video_path=args.video_path,
            output_dir=args.output_dir,
            visualize=args.visualize
        )
            
    elif args.mode == "analyze":
        if not args.csv_path or not args.court_json_path:
            print("Error: --csv_path and --court_json_path are required for analyze mode")
            return 1
            
        cmd = [
            "uv", "run", "scripts/analyze_zone4_ball_trajectories.py",
            "--csv-path", args.csv_path,
            "--court-json-path", args.court_json_path,
            "--output-dir", args.output_dir
        ]
        return run_command(cmd)
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
