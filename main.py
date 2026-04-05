#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
Orchestrates the 4-step pipeline: detect -> track -> combined -> reels.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

def run_script(script_path, args_list):
    cmd = [sys.executable, script_path] + args_list
    print(f"--- Running: {' '.join(cmd)} ---")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference Pipeline")
    parser.add_argument("--mode", choices=["detect", "track", "combined", "reels", "all", "pose", "hub-track", "analyze"],
                        default="track", help="Pipeline mode to execute")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--model_path", type=str, help="Path to model file (ONNX or OpenVINO XML)")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--court_json", type=str, help="Path to court coordinates JSON file")
    parser.add_argument("--device", type=str, default="CPU", help="Inference device (CPU, GPU, AUTO)")
    
    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY", ""),
                        help="Ultralytics Hub API key")

    # Pose specific arguments
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")

    args = parser.parse_args()
    
    video_path = args.video_path
    output_dir = args.output_dir
    video_name = Path(video_path).stem if video_path else None

    def step_detect():
        model = args.model_path or "models/VballNetV1_seq9_grayscale_330_h288_w512.onnx"
        if model.endswith(".xml"):
            script = "src/inference_openvino_seq_gray_v2.py"
            cmd_args = ["--video_path", video_path, "--model_xml", model, "--output_dir", output_dir, "--device", args.device]
        else:
            script = "src/inference_onnx_seq_gray_v2.py"
            cmd_args = ["--video_path", video_path, "--model_path", model, "--output_dir", output_dir]

        if args.visualize:
            cmd_args.append("--visualize")
        else:
            cmd_args.append("--only_csv")

        run_script(script, cmd_args)

    def step_track():
        # Expect ball.csv in output_dir/video_name/
        csv_path = os.path.join(output_dir, video_name, "ball.csv")
        # Handle different naming from OpenVINO script if necessary
        if not os.path.exists(csv_path):
            ov_csv = os.path.join(output_dir, f"{video_name}_predict_ball.csv")
            if os.path.exists(ov_csv):
                csv_path = ov_csv

        cmd_args = ["--csv_path", csv_path, "--output_dir", output_dir]
        if args.court_json:
            cmd_args.extend(["--court_json_path", args.court_json])
        run_script("src/track_calculator.py", cmd_args)

    def step_combined():
        cmd_args = ["--video_path", video_path, "--output_dir", output_dir]
        run_script("src/track_processor.py", cmd_args)

    def step_reels():
        json_dir = os.path.join(output_dir, video_name, "tracks")
        cmd_args = ["--video_path", video_path, "--output_dir", output_dir, "--json_dir", json_dir]
        run_script("src/make_reels.py", cmd_args)

    if args.mode == "detect":
        step_detect()
    elif args.mode == "track":
        step_track()
    elif args.mode == "combined":
        step_combined()
    elif args.mode == "reels":
        step_reels()
    elif args.mode == "all":
        step_detect()
        step_track()
        step_combined()
        step_reels()
    elif args.mode == "pose":
        if not args.track_file:
            print("Error: --track_file is required for pose mode")
            return 1
        run_script("src/pose_detector.py", ["--video_path", video_path, "--track_file", args.track_file, "--output_dir", output_dir])
    elif args.mode == "hub-track":
        if not args.api_key:
            print("Error: --api_key or ULTRALYTICS_HUB_API_KEY environment variable is required for hub-track mode")
            return 1
        run_script("src/hub_inference.py", ["--video_path", video_path, "--hub_model", args.hub_model, "--api_key", args.api_key, "--output_dir", output_dir])
    elif args.mode == "analyze":
        print("Analysis mode selected")
        print("This mode is not yet implemented")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
