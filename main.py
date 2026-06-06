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
    parser.add_argument("--mode", type=str, choices=["track", "pose", "analyze", "hub-track", "openvino-track"],
                        default="track", help="Processing mode")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    parser.add_argument("--model_path", type=str, default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--model_xml", type=str, default="ov/VballNetV2_seq9_grayscale_ov.xml",
                        help="Path to OpenVINO XML model file")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable visualization on display using cv2")
    parser.add_argument("--court_json", type=str, help="Path to court keypoints JSON (for analyze mode)")
    
    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY", "5ea02b4238fc9528408b8c36dcdb3834e11a9cbf58"),
                        help="Ultralytics Hub API key")

    args = parser.parse_args()
    
    if args.mode == "hub-track":
        if not args.video_path:
            print("Error: --video_path is required for hub-track mode")
            return 1

        cmd = ["uv", "run", "src/hub_inference.py", "--video_path", args.video_path, "--model_url", args.hub_model, "--output_dir", args.output_dir]
        if args.visualize: cmd.append("--visualize")
        run_command(cmd)

    elif args.mode == "track":
        if not args.video_path:
            print("Error: --video_path is required for tracking mode")
            return 1
            
        cmd = ["uv", "run", "src/inference_onnx_seq_gray_v2.py", "--video_path", args.video_path, "--model_path", args.model_path, "--output_dir", args.output_dir]
        if args.visualize: cmd.append("--visualize")
        run_command(cmd)

    elif args.mode == "openvino-track":
        if not args.video_path:
            print("Error: --video_path is required for openvino-track mode")
            return 1
            
        cmd = ["uv", "run", "src/inference_openvino_seq_gray_v2.py", "--video_path", args.video_path, "--model_xml", args.model_xml, "--output_dir", args.output_dir]
        if args.visualize: cmd.append("--visualize")
        run_command(cmd)

    elif args.mode == "pose":
        if not args.track_file or not args.video_path:
            print("Error: --track_file and --video_path are required for pose mode")
            return 1
            
        cmd = ["uv", "run", "src/pose_detector.py", "--video_path", args.video_path, "--track_file", args.track_file, "--output_dir", args.output_dir]
        if args.visualize: cmd.append("--visualize")
        run_command(cmd)
            
    elif args.mode == "analyze":
        if not args.video_path:
             print("Error: --video_path is required for analyze mode")
             return 1
        if not args.court_json:
             print("Error: --court_json is required for analyze mode (Zone 4 Trajectories)")
             return 1

        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        csv_path = os.path.join(args.output_dir, video_name, "ball.csv")

        if not os.path.exists(csv_path):
            print(f"Error: Detection CSV not found at {csv_path}. Run 'track' mode first.")
            return 1

        cmd = ["uv", "run", "scripts/analyze_zone4_ball_trajectories.py", "--csv-path", csv_path, "--court-json-path", args.court_json, "--video-path", args.video_path, "--output-dir", os.path.join(args.output_dir, video_name, "analysis")]
        if args.visualize: cmd.append("--visualize")
        run_command(cmd)
        
    else:
        print("Hello from fast-volleyball-tracking-inference!")
        print("Use --mode to specify the processing mode")
        print("Available modes: track, openvino-track, pose, analyze, hub-track")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
