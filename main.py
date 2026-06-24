#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
"""

import argparse
import os
import sys
import subprocess

def run_command(command):
    """Run a shell command and return its return code."""
    print(f"Executing: {' '.join(command)}")
    try:
        process = subprocess.Popen(command)
        process.wait()
        return process.returncode
    except Exception as e:
        print(f"Error executing command: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument("--mode", type=str, choices=["track", "pose", "analyze", "hub-track", "openvino-track"],
                        default="track", help="Processing mode")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    parser.add_argument("--model_path", type=str, default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
                        help="Path to ONNX model file")
    parser.add_argument("--model_xml", type=str, default="ov/VballNetGridV1b_seq9_grayscale_20260510_183219.xml",
                        help="Path to OpenVINO model XML file")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable visualization on display using cv2")
    
    # Analyze specific arguments
    parser.add_argument("--csv_path", type=str, help="Path to ball detection CSV (for analyze mode)")
    parser.add_argument("--court_json_path", type=str, help="Path to court JSON (for analyze mode)")

    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY", "5ea02b4238fc9528408b8c36dcdb3834e11a9cbf58"),
                        help="Ultralytics Hub API key")

    args = parser.parse_args()
    
    if args.mode == "hub-track":
        # Hub tracking mode
        if not args.video_path:
            print("Error: --video_path is required for hub-track mode")
            return 1

        if not args.api_key:
            print("Error: ULTRALYTICS_HUB_API_KEY environment variable is not set")
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

    elif args.mode == "track":
        # Ball tracking mode (ONNX)
        if not args.video_path:
            print("Error: --video_path is required for tracking mode")
            return 1
            
        cmd = [sys.executable, "src/inference_onnx_seq_gray_v2.py",
               "--video_path", args.video_path,
               "--model_path", args.model_path,
               "--output_dir", args.output_dir]
        if args.visualize:
            cmd.append("--visualize")

        return run_command(cmd)

    elif args.mode == "openvino-track":
        # Ball tracking mode (OpenVINO)
        if not args.video_path:
            print("Error: --video_path is required for openvino-track mode")
            return 1
            
        cmd = [sys.executable, "src/inference_openvino_seq_gray_v2.py",
               "--video_path", args.video_path,
               "--model_xml", args.model_xml,
               "--output_dir", args.output_dir]
        if args.visualize:
            cmd.append("--visualize")

        return run_command(cmd)

    elif args.mode == "pose":
        # Pose detection mode
        if not args.track_file or not args.video_path:
            print("Error: --track_file and --video_path are required for pose mode")
            return 1
            
        try:
            from src.pose_detector import add_pose_to_track_json
            print("Pose detection mode selected")
            print(f"Track file: {args.track_file}")
            print(f"Video: {args.video_path}")
            
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
        if not args.csv_path or not args.court_json_path:
            print("Error: --csv_path and --court_json_path are required for analyze mode")
            return 1

        cmd = [sys.executable, "scripts/analyze_zone4_ball_trajectories.py",
               "--csv-path", args.csv_path,
               "--court-json-path", args.court_json_path,
               "--output-dir", args.output_dir]

        return run_command(cmd)
        
    else:
        print("Fast Volleyball Tracking Inference Orchestrator")
        print("Use --mode to specify the processing mode")
        print("Available modes: track, openvino-track, pose, analyze, hub-track")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
