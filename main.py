#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
"""

import argparse
import os
import sys
import subprocess

def run_command(command):
    """Helper to run a shell command and handle errors."""
    try:
        # Use uv run if possible to ensure dependencies are available
        cmd = ["uv", "run"] + command
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False
    except FileNotFoundError:
        # Fallback to direct execution if uv is not present
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument("--mode", type=str, choices=["track", "pose", "analyze", "hub-track", "openvino-track"],
                        default="track", help="Processing mode")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    parser.add_argument("--model_path", type=str, default="models/vballNetV1.onnx", 
                        help="Path to ONNX model file")
    parser.add_argument("--model_xml", type=str, help="Path to OpenVINO .xml model file")
    parser.add_argument("--court_json_path", type=str, help="Path to court annotation JSON (for analyze mode)")
    parser.add_argument("--csv_path", type=str, help="Path to ball detection CSV (for analyze mode)")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable visualization on display using cv2")
    
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
        except ImportError as e:
            print(f"Error importing hub inference module: {e}")
            return 1
        except Exception as e:
            print(f"Error during hub inference: {e}")
            return 1

    elif args.mode == "track":
        if not args.video_path:
            print("Error: --video_path is required for tracking mode")
            return 1
            
        print("Ball tracking mode selected (ONNX)")
        cmd = [sys.executable, "src/inference_onnx_seq_gray_v2.py", "--video_path", args.video_path, "--model_path", args.model_path]
        if args.visualize: cmd.append("--visualize")
        if args.output_dir: cmd.extend(["--output_dir", args.output_dir])
        run_command(cmd)

    elif args.mode == "openvino-track":
        if not args.video_path:
            print("Error: --video_path is required for OpenVINO tracking mode")
            return 1

        model_xml = args.model_xml or args.model_path
        if not model_xml:
             print("Error: --model_xml or --model_path is required for OpenVINO tracking mode")
             return 1

        print(f"OpenVINO tracking mode selected using model: {model_xml}")
        cmd = [sys.executable, "src/inference_openvino_seq_gray_v2.py", "--video_path", args.video_path, "--model_xml", model_xml]
        if args.output_dir: cmd.extend(["--output_dir", args.output_dir])
        if args.visualize: cmd.append("--visualize")
        run_command(cmd)
            
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
        except ImportError as e:
            print(f"Error importing pose detection module: {e}")
            return 1
            
    elif args.mode == "analyze":
        if not args.csv_path or not args.court_json_path:
            print("Error: --csv_path and --court_json_path are required for analyze mode")
            return 1

        print("Analysis mode selected")
        script_path = os.path.join("scripts", "analyze_zone4_ball_trajectories.py")
        if os.path.exists(script_path):
            print(f"Running analysis script: {script_path}")
            cmd = [sys.executable, script_path, "--csv-path", args.csv_path, "--court-json-path", args.court_json_path]
            if args.output_dir: cmd.extend(["--output-dir", args.output_dir])
            if args.video_path: cmd.extend(["--video-path", args.video_path])
            if args.visualize: cmd.append("--visualize")
            run_command(cmd)
        else:
            print("Error: Analysis script not found in scripts/ directory")
        
    else:
        print("Available modes: track, pose, analyze, hub-track, openvino-track")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
