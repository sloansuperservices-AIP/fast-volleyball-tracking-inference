#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
"""

import argparse
import os
import subprocess
import sys


def run_command(command):
    """Run a command as a subprocess and handle its exit code."""
    try:
        subprocess.check_call(command)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return e.returncode
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference Orchestrator")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["track", "pose", "analyze", "hub-track", "openvino-track"],
        default="track",
        help="Processing mode",
    )
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--model_path", type=str, help="Path to ONNX model file (for track mode)")
    parser.add_argument("--model_xml", type=str, help="Path to OpenVINO .xml model file (for openvino-track mode)")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--only_csv", action="store_true", help="Save only CSV, skip video output")

    # Analysis specific arguments
    parser.add_argument("--csv_path", type=str, help="Path to detections CSV (for analyze mode)")
    parser.add_argument("--court_json_path", type=str, help="Path to court JSON (for analyze mode)")

    # Pose specific arguments
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")

    # Hub specific arguments
    parser.add_argument(
        "--hub_model",
        type=str,
        default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
        help="Ultralytics Hub model URL or ID",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("ULTRALYTICS_HUB_API_KEY"),
        help="Ultralytics Hub API key",
    )

    args = parser.parse_args()

    if args.mode == "track":
        if not args.video_path or not args.model_path:
            print("Error: --video_path and --model_path are required for track mode")
            return 1
        cmd = [
            sys.executable,
            "src/inference_onnx_seq_gray_v2.py",
            "--video_path", args.video_path,
            "--model_path", args.model_path,
            "--output_dir", args.output_dir,
        ]
        if args.visualize:
            cmd.append("--visualize")
        if args.only_csv:
            cmd.append("--only_csv")
        return run_command(cmd)

    elif args.mode == "openvino-track":
        if not args.video_path or not args.model_xml:
            print("Error: --video_path and --model_xml are required for openvino-track mode")
            return 1
        cmd = [
            sys.executable,
            "src/inference_openvino_seq_gray_v2.py",
            "--video_path", args.video_path,
            "--model_xml", args.model_xml,
            "--output_dir", args.output_dir,
        ]
        if args.visualize:
            cmd.append("--visualize")
        if args.only_csv:
            cmd.append("--only_csv")
        return run_command(cmd)

    elif args.mode == "analyze":
        if not args.csv_path or not args.court_json_path:
            print("Error: --csv_path and --court_json_path are required for analyze mode")
            return 1
        cmd = [
            sys.executable,
            "scripts/analyze_zone4_ball_trajectories.py",
            "--csv-path", args.csv_path,
            "--court-json-path", args.court_json_path,
            "--output-dir", args.output_dir,
        ]
        if args.video_path:
            cmd.extend(["--video-path", args.video_path])
        if args.visualize:
            cmd.append("--visualize")
        return run_command(cmd)

    elif args.mode == "hub-track":
        if not args.video_path:
            print("Error: --video_path is required for hub-track mode")
            return 1
        if not args.api_key:
            print("Error: ULTRALYTICS_HUB_API_KEY environment variable or --api_key is required for hub-track mode")
            return 1

        # Call hub_inference tool
        cmd = [
            sys.executable,
            "src/hub_inference.py",
            "--video_path", args.video_path,
            "--model_url", args.hub_model,
            "--api_key", args.api_key,
            "--output_dir", args.output_dir,
        ]
        if args.visualize:
            cmd.append("--visualize")
        return run_command(cmd)

    elif args.mode == "pose":
        if not args.track_file or not args.video_path:
            print("Error: --track_file and --video_path are required for pose mode")
            return 1
        
        cmd = [
            sys.executable,
            "src/process_track_pose.py",
            "--video_path", args.video_path,
            "--track_file", args.track_file,
            "--output_dir", args.output_dir,
        ]
        if args.visualize:
            cmd.append("--visualize")
        return run_command(cmd)

    return 0


if __name__ == "__main__":
    sys.exit(main())
