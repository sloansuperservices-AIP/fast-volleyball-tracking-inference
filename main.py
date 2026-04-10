#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
"""

import argparse
import os
import sys
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["detect", "track", "combined", "reels", "all", "pose", "analyze", "hub-track"],
        default="track",
        help="Processing mode: detect (ball detection), track (trajectory calculation), combined (horizontal assembly), reels (vertical 9:16), all (full pipeline)",
    )
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
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
        "--visualize", action="store_true", help="Enable visualization on display using cv2"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["onnx", "openvino"],
        default="onnx",
        help="Inference engine: onnx (default) or openvino",
    )
    parser.add_argument(
        "--only_csv",
        action="store_true",
        help="Save only CSV during detection, skip video output",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second")

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
            print("Error: --api_key or ULTRALYTICS_HUB_API_KEY env var is required for hub-track mode")
            return 1
        try:
            from src.hub_inference import run_hub_inference

            print(f"Hub tracking mode selected for {args.video_path}")
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
        if not args.video_path:
            print(f"Error: --video_path is required for {args.mode} mode")
            return 1

        script = (
            "src/inference_onnx_seq_gray_v2.py"
            if args.engine == "onnx"
            else "src/inference_openvino_seq_gray_v2.py"
        )
        cmd = [
            sys.executable,
            script,
            "--video_path",
            args.video_path,
            "--output_dir",
            args.output_dir,
        ]
        if args.engine == "onnx":
            cmd.extend(["--model_path", args.model_path])
        else:
            cmd.extend(["--model_xml", args.model_path])
        if args.visualize:
            cmd.append("--visualize")
        if args.only_csv:
            cmd.append("--only_csv")

        print(f"Step 1: Running detection with {args.engine} engine...")
        subprocess.run(cmd, check=True)

    if args.mode in ["track", "all"]:
        video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
        csv_path = os.path.join(args.output_dir, video_basename, "ball.csv")
        print(f"Step 2: Calculating tracks from {csv_path}...")
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
        subprocess.run(cmd, check=True)

    if args.mode in ["combined", "all"]:
        print(f"Step 3: Assembling rally clips...")
        cmd = [
            sys.executable,
            "src/track_processor.py",
            "--video_path",
            args.video_path,
            "--output_dir",
            args.output_dir,
        ]
        subprocess.run(cmd, check=True)

    if args.mode in ["reels", "all"]:
        video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
        json_dir = os.path.join(args.output_dir, video_basename, "tracks")
        print(f"Step 4: Generating vertical reels from {json_dir}...")
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
        subprocess.run(cmd, check=True)

    elif args.mode == "pose":
        if not args.track_file or not args.video_path:
            print("Error: --track_file and --video_path are required for pose mode")
            return 1
        try:
            from src.pose_detector import add_pose_to_track_json

            add_pose_to_track_json(
                track_file=args.track_file,
                video_path=args.video_path,
                output_dir=args.output_dir,
                visualize=args.visualize,
            )
        except Exception as e:
            print(f"Error during pose detection: {e}")
            return 1

    elif args.mode == "analyze":
        print("Analysis mode is currently a placeholder.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
