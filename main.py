#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking pipeline.
Orchestrates the 4-step pipeline: detect, track, combined, and reels.
Also supports pose detection and Ultralytics Hub inference.
"""

import argparse
import os
import subprocess
import sys


def run_command(command, verbose=False):
    """Run a shell command and handle output."""
    if verbose:
        print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=not verbose, text=True)
    if result.returncode != 0:
        if not verbose:
            print(result.stdout)
            print(result.stderr)
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["detect", "track", "combined", "reels", "all", "pose", "hub-track"],
        default="all",
        help="Pipeline step or mode to run",
    )
    parser.add_argument("--video_path", type=str, required=True, help="Input video file")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
        help="Path to ball detection ONNX model",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Root output directory"
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    parser.add_argument(
        "--court_json", type=str, help="Optional path to court annotations"
    )
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    parser.add_argument("--visualize", action="store_true", help="Enable live preview")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    
    # Hub and Pose specific arguments
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
    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    video_out_dir = os.path.join(args.output_dir, video_basename)
    ball_csv = os.path.join(video_out_dir, "ball.csv")
    tracks_dir = os.path.join(video_out_dir, "tracks")

    steps = []
    if args.mode == "all":
        steps = ["detect", "track", "combined", "reels"]
    else:
        steps = [args.mode]

    python_exec = sys.executable

    try:
        for step in steps:
            print(f"\n--- Step: {step.upper()} ---")

            if step == "detect":
                cmd = [
                    python_exec,
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
                run_command(cmd, args.verbose)

            elif step == "track":
                if not os.path.exists(ball_csv):
                    print(f"Error: {ball_csv} not found. Run 'detect' first.")
                    return 1
                cmd = [
                    python_exec,
                    "src/track_calculator.py",
                    "--csv_path",
                    ball_csv,
                    "--output_dir",
                    args.output_dir,
                    "--fps",
                    str(args.fps),
                ]
                if args.court_json:
                    cmd.extend(["--court_json_path", args.court_json])
                if args.verbose:
                    cmd.append("--verbose")
                run_command(cmd, args.verbose)

            elif step == "combined":
                cmd = [
                    python_exec,
                    "src/track_processor.py",
                    "--video_path",
                    args.video_path,
                    "--output_dir",
                    args.output_dir,
                ]
                if args.verbose:
                    cmd.append("--verbose")
                run_command(cmd, args.verbose)

            elif step == "reels":
                if not os.path.exists(tracks_dir):
                    print(f"Error: {tracks_dir} not found. Run 'track' first.")
                    return 1
                cmd = [
                    python_exec,
                    "src/make_reels.py",
                    "--video_path",
                    args.video_path,
                    "--json_dir",
                    tracks_dir,
                    "--output_dir",
                    args.output_dir,
                ]
                if args.visualize:
                    cmd.append("--visualize")
                if args.verbose:
                    cmd.append("--verbose")
                run_command(cmd, args.verbose)

            elif step == "pose":
                if not args.track_file:
                    # Try to find default track if not provided
                    if os.path.exists(tracks_dir):
                        json_files = [f for f in os.listdir(tracks_dir) if f.endswith(".json")]
                        if json_files:
                            args.track_file = os.path.join(tracks_dir, sorted(json_files)[0])

                if not args.track_file or not os.path.exists(args.track_file):
                    print("Error: --track_file is required for pose mode and could not be auto-detected.")
                    return 1

                cmd = [
                    python_exec,
                    "src/pose_detector.py",
                    "--video_path",
                    args.video_path,
                    "--track_file",
                    args.track_file,
                ]
                if args.visualize:
                    cmd.append("--visualize")
                run_command(cmd, args.verbose)

            elif step == "hub-track":
                cmd = [
                    python_exec,
                    "src/hub_inference.py",
                    "--video_path",
                    args.video_path,
                    "--model_url",
                    args.hub_model,
                    "--output_dir",
                    args.output_dir,
                ]
                if args.api_key:
                    cmd.extend(["--api_key", args.api_key])
                if args.visualize:
                    cmd.append("--visualize")
                run_command(cmd, args.verbose)

        print("\nOperation completed successfully.")

    except Exception as e:
        print(f"\nOperation failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
