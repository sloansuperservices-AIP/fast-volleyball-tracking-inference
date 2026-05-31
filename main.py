#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
"""

import argparse
import os
import subprocess
import sys


def run_command(command):
    """Runs a shell command and ensures it completes successfully."""
    print(f"Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["track", "pose", "analyze", "hub-track", "openvino-track"],
        default="track",
        help="Processing mode",
    )
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument(
        "--track_file", type=str, help="Path to track JSON file (for pose mode)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--model_xml",
        type=str,
        default="ov/VballNetV2_seq9_grayscale_ov.xml",
        help="Path to OpenVINO .xml model",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Directory to save output files"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Enable visualization on display using cv2"
    )
    parser.add_argument(
        "--only_csv", action="store_true", help="Skip writing output video"
    )

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

    if args.mode == "hub-track":
        if not args.video_path:
            print("Error: --video_path is required for hub-track mode")
            return 1

        if not args.api_key:
            print("Error: --api_key or ULTRALYTICS_HUB_API_KEY environment variable is required for hub-track mode")
            return 1

        from src.hub_inference import run_hub_inference

        run_hub_inference(
            video_path=args.video_path,
            model_url=args.hub_model,
            api_key=args.api_key,
            output_dir=args.output_dir,
            visualize=args.visualize,
        )

    elif args.mode == "track":
        if not args.video_path:
            print("Error: --video_path is required for tracking mode")
            return 1
        command = [
            "uv",
            "run",
            "src/inference_onnx_seq_gray_v2.py",
            "--video_path",
            args.video_path,
            "--model_path",
            args.model_path,
            "--output_dir",
            args.output_dir,
        ]
        if args.visualize:
            command.append("--visualize")
        if args.only_csv:
            command.append("--only_csv")
        run_command(command)

    elif args.mode == "openvino-track":
        if not args.video_path:
            print("Error: --video_path is required for openvino-track mode")
            return 1
        command = [
            "uv",
            "run",
            "src/inference_openvino_seq_gray_v2.py",
            "--video_path",
            args.video_path,
            "--model_xml",
            args.model_xml,
            "--output_dir",
            args.output_dir,
        ]
        if args.visualize:
            command.append("--visualize")
        if args.only_csv:
            command.append("--only_csv")
        run_command(command)

    elif args.mode == "pose":
        if not args.track_file or not args.video_path:
            print("Error: --track_file and --video_path are required for pose mode")
            return 1
        from src.pose_detector import add_pose_to_track_json

        add_pose_to_track_json(
            track_file=args.track_file,
            video_path=args.video_path,
            output_dir=args.output_dir,
            visualize=args.visualize,
        )

    elif args.mode == "analyze":
        if not args.video_path:
            print("Error: --video_path is required for analyze mode")
            return 1
        # In analyze mode, we first need to find the ball.csv
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        csv_path = os.path.join(args.output_dir, video_name, "ball.csv")
        if not os.path.exists(csv_path):
            # Try fallback name
            csv_path = os.path.join(args.output_dir, f"{video_name}_predict_ball.csv")

        if not os.path.exists(csv_path):
            print(f"Error: Could not find ball.csv for {video_name} in {args.output_dir}")
            return 1

        command = [
            "uv",
            "run",
            "src/track_calculator.py",
            "--csv_path",
            csv_path,
            "--output_dir",
            args.output_dir,
        ]
        run_command(command)

    return 0


if __name__ == "__main__":
    sys.exit(main())
