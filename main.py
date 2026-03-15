#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.

Modes
-----
track   - Offline ball detection on a video file (outputs CSV / annotated video)
live    - Real-time tracking from a webcam, RTSP stream, or video file
pose    - Adds pose keypoints to an existing track JSON
analyze - (future)
hub-track - Inference via Ultralytics Hub model
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["track", "live", "pose", "analyze", "hub-track"],
        default="track",
        help="Processing mode",
    )

    # ---- shared ----
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Directory for output files"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show live preview window during processing",
    )

    # ---- live mode ----
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Live mode: video source — camera index (0, 1, …), "
            "RTSP/RTMP/HTTP URL, or file path. "
            "Falls back to --video_path if not set."
        ),
    )
    parser.add_argument(
        "--trail_length",
        type=int,
        default=25,
        help="Live mode: number of past ball positions to draw as trail",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Live mode: record the annotated output to --output_dir",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Live mode: disable display window (headless recording)",
    )

    # ---- track mode ----
    parser.add_argument(
        "--only_csv",
        action="store_true",
        help="Track mode: skip annotated video and save only ball.csv",
    )
    parser.add_argument(
        "--track_length",
        type=int,
        default=8,
        help="Track mode: trail length for visualization",
    )

    # ---- pose mode ----
    parser.add_argument(
        "--track_file",
        type=str,
        help="Pose mode: path to track JSON file",
    )

    # ---- hub-track mode ----
    parser.add_argument(
        "--hub_model",
        type=str,
        default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
        help="Hub mode: Ultralytics Hub model URL or ID",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv(
            "ULTRALYTICS_HUB_API_KEY",
            "5ea02b4238fc9528408b8c36dcdb3834e11a9cbf58",
        ),
        help="Hub mode: Ultralytics Hub API key",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # live — real-time tracking from camera / RTSP / file                 #
    # ------------------------------------------------------------------ #
    if args.mode == "live":
        source = args.source or args.video_path
        if source is None:
            source = "0"  # default to first webcam
            print("No --source or --video_path given, defaulting to camera 0.")

        try:
            from src.live_tracker import LiveTracker
        except ImportError as exc:
            print(f"Error importing live_tracker: {exc}")
            return 1

        print(f"Live tracking — source: {source!r}, model: {args.model_path}")
        tracker = LiveTracker(
            source=source,
            model_path=args.model_path,
            trail_length=args.trail_length,
            display=not args.no_display,
            record=args.record,
            output_dir=args.output_dir,
        )
        try:
            tracker.run()
        except Exception as exc:
            print(f"Error during live tracking: {exc}")
            return 1

    # ------------------------------------------------------------------ #
    # track — offline ball detection on a video file                      #
    # ------------------------------------------------------------------ #
    elif args.mode == "track":
        if not args.video_path:
            print("Error: --video_path is required for track mode")
            return 1

        try:
            from src.inference_onnx_seq9_gray_v2 import main as track_main
        except ImportError as exc:
            print(f"Error importing inference module: {exc}")
            return 1

        print(f"Ball tracking — video: {args.video_path}, model: {args.model_path}")

        # inference_onnx_seq9_gray_v2.main() reads sys.argv, so inject args
        sys.argv = [
            sys.argv[0],
            "--video_path", args.video_path,
            "--model_path", args.model_path,
            "--output_dir", args.output_dir,
            "--track_length", str(args.track_length),
        ]
        if args.visualize:
            sys.argv.append("--visualize")
        if args.only_csv:
            sys.argv.append("--only_csv")

        try:
            track_main()
        except Exception as exc:
            print(f"Error during ball tracking: {exc}")
            return 1

    # ------------------------------------------------------------------ #
    # pose — add pose keypoints to an existing track JSON                 #
    # ------------------------------------------------------------------ #
    elif args.mode == "pose":
        if not args.track_file or not args.video_path:
            print("Error: --track_file and --video_path are required for pose mode")
            return 1

        try:
            from src.pose_detector import add_pose_to_track_json
        except ImportError as exc:
            print(f"Error importing pose_detector: {exc}")
            return 1

        print(f"Pose detection — track: {args.track_file}, video: {args.video_path}")
        try:
            add_pose_to_track_json(
                track_file=args.track_file,
                video_path=args.video_path,
                output_dir=args.output_dir,
                visualize=args.visualize,
            )
        except Exception as exc:
            print(f"Error during pose detection: {exc}")
            return 1

    # ------------------------------------------------------------------ #
    # hub-track — inference via Ultralytics Hub model                     #
    # ------------------------------------------------------------------ #
    elif args.mode == "hub-track":
        if not args.video_path:
            print("Error: --video_path is required for hub-track mode")
            return 1

        try:
            from src.hub_inference import run_hub_inference
        except ImportError as exc:
            print(f"Error importing hub_inference: {exc}")
            return 1

        print(f"Hub tracking — video: {args.video_path}, model: {args.hub_model}")
        try:
            run_hub_inference(
                video_path=args.video_path,
                model_url=args.hub_model,
                api_key=args.api_key,
                output_dir=args.output_dir,
                visualize=args.visualize,
            )
        except Exception as exc:
            print(f"Error during hub inference: {exc}")
            return 1

    # ------------------------------------------------------------------ #
    # analyze — placeholder                                                #
    # ------------------------------------------------------------------ #
    elif args.mode == "analyze":
        print("Analysis mode is not yet implemented.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
