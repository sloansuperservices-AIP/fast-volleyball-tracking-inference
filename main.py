#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
"""

import argparse
import os
import sys
import subprocess

def run_script(script_path, args_list):
    """Run a script using uv run python."""
    cmd = ["uv", "run", "python", script_path] + args_list
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)

def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument("--mode", type=str, choices=["track", "pose", "analyze", "hub-track", "openvino-track"],
                        default="track", help="Processing mode")

    # Common arguments
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable visualization")

    # Track mode arguments
    parser.add_argument("--model_path", type=str, default="models/vballNetV1.onnx",
                        help="Path to ONNX model file")

    # Pose mode arguments
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")

    # Analyze mode arguments
    parser.add_argument("--csv_path", type=str, help="Path to ball.csv for analysis")
    parser.add_argument("--court_json_path", type=str, help="Path to court coordinates JSON file")
    
    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY", "5ea02b4238fc9528408b8c36dcdb3834e11a9cbf58"),
                        help="Ultralytics Hub API key")

    # OpenVINO specific arguments
    parser.add_argument("--model_xml", type=str, help="Path to OpenVINO .xml model")
    parser.add_argument("--device", type=str, default="CPU", help="OpenVINO device (CPU, GPU, AUTO)")

    args = parser.parse_args()
    
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

        script_args = [
            "--video_path", args.video_path,
            "--model_path", args.model_path,
            "--output_dir", args.output_dir
        ]
        if args.visualize:
            script_args.append("--visualize")

        return run_script("src/inference_onnx_seq_gray_v2.py", script_args)
            
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
        if not args.csv_path or not args.court_json_path:
            print("Error: --csv_path and --court_json_path are required for analyze mode")
            return 1
        
        script_args = [
            "--csv-path", args.csv_path,
            "--court-json-path", args.court_json_path,
            "--output-dir", args.output_dir
        ]
        if args.visualize:
            script_args.append("--visualize")
        if args.video_path:
            script_args.extend(["--video-path", args.video_path])

        return run_script("scripts/analyze_zone4_ball_trajectories.py", script_args)

    elif args.mode == "openvino-track":
        if not args.video_path or not args.model_xml:
            print("Error: --video_path and --model_xml are required for openvino-track mode")
            return 1

        script_args = [
            "--video_path", args.video_path,
            "--model_xml", args.model_xml,
            "--device", args.device,
            "--output_dir", args.output_dir
        ]
        if args.visualize:
            script_args.append("--visualize")

        return run_script("src/inference_openvino_seq_gray_v2.py", script_args)
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
