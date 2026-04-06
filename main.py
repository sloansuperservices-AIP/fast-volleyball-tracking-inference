#!/usr/bin/env python3
"""
Unified entry point for the fast volleyball tracking pipeline.
"""

import argparse
import os
import sys
import subprocess


def run_command(command):
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["detect", "track", "combined", "reels", "all", "pose", "hub-track"],
        default="detect",
        help="Processing mode (all runs detect -> track -> combined -> reels)",
    )
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
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
        "--visualize", action="store_true", help="Enable visualization"
    )
    parser.add_argument(
        "--only_csv", action="store_true", help="Detection: save only CSV"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Track calculation: video FPS"
    )

    args = parser.parse_args()

    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    video_out_dir = os.path.join(args.output_dir, video_basename)
    csv_path = os.path.join(video_out_dir, "ball.csv")
    tracks_dir = os.path.join(video_out_dir, "tracks")

    modes_to_run = []
    if args.mode == "all":
        modes_to_run = ["detect", "track", "combined", "reels"]
    else:
        modes_to_run = [args.mode]

    for mode in modes_to_run:
        print(f"\n--- Running mode: {mode} ---")
        if mode == "detect":
            cmd = [
                sys.executable,
                "src/inference_onnx_seq_gray_v2.py",
                "--video_path",
                args.video_path,
                "--model_path",
                args.model_path,
                "--output_dir",
                args.output_dir,
            ]
            if args.visualize:
                cmd.append("--visualize")
            if args.only_csv:
                cmd.append("--only_csv")
            if not run_command(cmd):
                return 1

        elif mode == "track":
            if not os.path.exists(csv_path):
                print(f"Error: {csv_path} not found. Run 'detect' mode first.")
                return 1
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
            if not run_command(cmd):
                return 1

        elif mode == "combined":
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

        elif mode == "reels":
            if not os.path.exists(tracks_dir):
                print(f"Error: {tracks_dir} not found. Run 'track' mode first.")
                return 1
            cmd = [
                sys.executable,
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
            if not run_command(cmd):
                return 1

        elif mode == "pose":
            # Assuming track_file might be needed, or it processes the whole video
            cmd = [
                sys.executable,
                "src/pose_detector.py",
                "--video_path",
                args.video_path,
                "--output_dir",
                args.output_dir,
            ]
            if args.visualize:
                cmd.append("--visualize")
            if not run_command(cmd):
                return 1

        elif mode == "hub-track":
            cmd = [
                sys.executable,
                "src/hub_inference.py",
                "--video_path",
                args.video_path,
                "--output_dir",
                args.output_dir,
            ]
            if args.visualize:
                cmd.append("--visualize")
            if not run_command(cmd):
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
