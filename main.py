#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
Unified automated 4-step pipeline and advanced sports analytics.
"""

import argparse
import os
import sys
import subprocess


def run_command(command: list[str]) -> bool:
    """Helper to run a subprocess command and return success status."""
    print(f"Executing: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["detect", "track", "combined", "reels", "all", "pose", "analyze", "hub-track"],
        default="all",
        help="Processing mode",
    )
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization on display using cv2")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for track calculation")
    parser.add_argument("--court_json", type=str, help="Path to optional court annotation JSON")

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

    if not args.video_path and args.mode != "analyze":
        print(f"Error: --video_path is required for mode '{args.mode}'")
        return 1

    video_basename = os.path.splitext(os.path.basename(args.video_path))[0] if args.video_path else ""
    csv_path = os.path.join(args.output_dir, video_basename, "ball.csv")
    json_dir = os.path.join(args.output_dir, video_basename, "tracks")

    if args.mode == "hub-track":
        if not args.api_key:
            print("Error: Ultralytics Hub API key is required. Use --api_key or set ULTRALYTICS_HUB_API_KEY.")
            return 1
        try:
            from src.hub_inference import run_hub_inference

            run_hub_inference(
                video_path=args.video_path,
                model_url=args.hub_model,
                api_key=args.api_key,
                output_dir=args.output_dir,
                visualize=args.visualize,
            )
        except Exception as e:
            print(f"Error during hub inference: {e}")
            return 1

    elif args.mode in ["detect", "all"]:
        cmd = [
            sys.executable,
            "src/inference_onnx_seq_gray_v2.py",
            "--video_path",
            args.video_path,
            "--model_path",
            args.model_path,
            "--output_dir",
            args.output_dir,
            "--only_csv",
        ]
        if args.visualize:
            cmd.append("--visualize")
        if not run_command(cmd):
            return 1

    if args.mode in ["track", "all"]:
        cmd = [
            sys.executable,
            "src/track_calculator.py",
            "--csv_path",
            csv_path,
            "--output_dir",
            args.output_dir,
            "--fps",
            str(args.fps),
        ]
        if args.court_json:
            cmd.extend(["--court_json_path", args.court_json])
        if not run_command(cmd):
            return 1

    if args.mode in ["combined", "all"]:
        cmd = [
            sys.executable,
            "src/track_processor.py",
            "--video_path",
            args.video_path,
            "--output_dir",
            args.output_dir,
        ]
        if not run_command(cmd):
            return 1

    if args.mode in ["reels", "all"]:
        cmd = [
            sys.executable,
            "src/make_reels.py",
            "--video_path",
            args.video_path,
            "--json_dir",
            json_dir,
            "--output_dir",
            args.output_dir,
        ]
        if not run_command(cmd):
            return 1

    elif args.mode == "pose":
        if not args.track_file:
            print("Error: --track_file is required for pose mode")
            return 1
        cmd = [
            sys.executable,
            "src/pose_detector.py",
            "--video_path",
            args.video_path,
            "--track_file",
            args.track_file,
            "--output_dir",
            args.output_dir,
        ]
        if args.visualize:
            cmd.append("--visualize")
        if not run_command(cmd):
            return 1

    elif args.mode == "analyze":
        print("Analysis mode selected. This usually involves zone-4 trajectory processing.")
        if not args.court_json:
            print("Error: --court_json is required for analyze mode")
            return 1
        cmd = [
            sys.executable,
            "scripts/analyze_zone4_ball_trajectories.py",
            "--csv-path",
            csv_path,
            "--court-json-path",
            args.court_json,
            "--output-dir",
            os.path.join(args.output_dir, video_basename, "analysis"),
            "--reels",
        ]
        if not run_command(cmd):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
