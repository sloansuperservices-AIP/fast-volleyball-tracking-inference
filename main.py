#!/usr/bin/env python3
"""
Standardized pipeline orchestrator for fast volleyball ball tracking.
Supports: detect -> track -> combined -> reels
"""

import argparse
import os
import sys
import subprocess


def run_command(command):
    """Run a shell command and handle errors."""
    print(f"Executing: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(command)}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Fast Volleyball Tracking Pipeline Orchestrator"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["detect", "track", "combined", "reels", "all", "pose", "analyze", "hub-track"],
        default="all",
        help="Processing mode",
    )
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Directory to save output files"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Frames per second for track calculation"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Enable visualization on display"
    )
    parser.add_argument(
        "--track_file", type=str, help="Path to track JSON file (for pose mode)"
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
            sys.exit(1)
        if not args.api_key:
            print("Error: API key is required for hub-track mode. Set ULTRALYTICS_HUB_API_KEY or use --api_key")
            sys.exit(1)

        try:
            from src.hub_inference import run_hub_inference
            run_hub_inference(
                video_path=args.video_path,
                model_url=args.hub_model,
                api_key=args.api_key,
                output_dir=args.output_dir,
                visualize=args.visualize,
            )
        except ImportError as e:
            print(f"Error importing hub inference module: {e}")
            sys.exit(1)
        return

    if args.mode in ["detect", "all"]:
        if not args.video_path:
            print("Error: --video_path is required for detect mode")
            sys.exit(1)

        # Use the generic v2 script which handles both base and GRU models
        detect_cmd = [
            sys.executable,
            "src/inference_onnx_seq_gray_v2.py",
            "--video_path", args.video_path,
            "--model_path", args.model_path,
            "--output_dir", args.output_dir,
            "--only_csv"
        ]
        if args.visualize:
            detect_cmd.remove("--only_csv")
            detect_cmd.append("--visualize")
        run_command(detect_cmd)

    if args.mode in ["track", "all"]:
        if not args.video_path:
             print("Error: --video_path is required for track mode")
             sys.exit(1)

        video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
        csv_path = os.path.join(args.output_dir, video_basename, "ball.csv")

        track_cmd = [
            sys.executable,
            "src/track_calculator.py",
            "--csv_path", csv_path,
            "--output_dir", args.output_dir,
            "--fps", str(args.fps)
        ]
        run_command(track_cmd)

    if args.mode in ["combined", "all"]:
        if not args.video_path:
             print("Error: --video_path is required for combined mode")
             sys.exit(1)

        combined_cmd = [
            sys.executable,
            "src/track_processor.py",
            "--video_path", args.video_path,
            "--output_dir", args.output_dir
        ]
        run_command(combined_cmd)

    if args.mode in ["reels", "all"]:
        if not args.video_path:
             print("Error: --video_path is required for reels mode")
             sys.exit(1)

        video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
        json_dir = os.path.join(args.output_dir, video_basename, "tracks")

        reels_cmd = [
            sys.executable,
            "src/make_reels.py",
            "--video_path", args.video_path,
            "--json_dir", json_dir,
            "--output_dir", args.output_dir
        ]
        if args.visualize:
            reels_cmd.append("--visualize")
        run_command(reels_cmd)

    if args.mode == "pose":
        if not args.track_file or not args.video_path:
            print("Error: --track_file and --video_path are required for pose mode")
            sys.exit(1)
            
        pose_cmd = [
            sys.executable,
            "src/pose_detector.py",
            "--track_file", args.track_file,
            "--video_path", args.video_path
        ]
        # pose_detector.py output_dir is hardcoded in add_pose_to_track_json but can be modified if needed.
        # It doesn't seem to use --output_dir in its main().
        if args.visualize:
            pose_cmd.append("--visualize")

        run_command(pose_cmd)

    if args.mode == "analyze":
        print("Analysis mode selected")
        print("This mode is not yet implemented")


if __name__ == "__main__":
    main()
