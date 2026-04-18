#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "detect",
            "track",
            "combined",
            "reels",
            "all",
            "pose",
            "analyze",
            "hub-track",
        ],
        default="all",
        help="Processing mode",
    )
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument(
        "--track_file", type=str, help="Path to track JSON file (for pose mode)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Directory to save output files"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization on display using cv2",
    )
    parser.add_argument(
        "--only_csv", action="store_true", help="Save only CSV in detect mode"
    )
    parser.add_argument("--fps", type=int, default=30, help="FPS for track calculation")
    parser.add_argument(
        "--runtime",
        type=str,
        choices=["onnx", "openvino"],
        default="onnx",
        help="Inference runtime",
    )

    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY", "5ea02b4238fc9528408b8c36dcdb3834e11a9cbf58"),
                        help="Ultralytics Hub API key")

    args = parser.parse_args()

    if not args.video_path and args.mode != "analyze":
        print("Error: --video_path is required")
        return 1

    if args.video_path:
        video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
        csv_path = os.path.join(args.output_dir, video_basename, "ball.csv")
        json_dir = os.path.join(args.output_dir, video_basename, "tracks")
    else:
        video_basename = None
        csv_path = None
        json_dir = None

    if args.mode == "hub-track":
        # Hub tracking mode
        if not args.video_path:
            print("Error: --video_path is required for hub-track mode")
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

    elif args.mode in ["detect", "all"]:
        print(f"--- Step 1: Ball Detection ({args.runtime}) ---")
        try:
            if args.runtime == "onnx":
                import src.inference_onnx_seq_gray_v2 as detector
            else:
                import src.inference_openvino_seq_gray_v2 as detector

            # Override sys.argv to pass arguments to the script
            sys.argv = [
                detector.__file__,
                "--video_path",
                args.video_path,
                "--model_path",
                args.model_path,
                "--output_dir",
                args.output_dir,
            ]
            if args.visualize:
                sys.argv.append("--visualize")
            if args.only_csv or args.mode == "all":
                sys.argv.append("--only_csv")

            detector.main()
        except Exception as e:
            print(f"Error during detection: {e}")
            return 1

    if args.mode in ["track", "all"]:
        print("--- Step 2: Track Calculation ---")
        try:
            import src.track_calculator as tracker

            sys.argv = [
                tracker.__file__,
                "--csv_path",
                csv_path,
                "--output_dir",
                args.output_dir,
                "--fps",
                str(args.fps),
            ]
            tracker.main()
        except Exception as e:
            print(f"Error during track calculation: {e}")
            return 1

    if args.mode in ["combined", "all"]:
        print("--- Step 3: Horizontal Assembly ---")
        try:
            import src.track_processor as processor

            sys.argv = [
                processor.__file__,
                "--video_path",
                args.video_path,
                "--output_dir",
                args.output_dir,
                "--json_dir",
                json_dir,
            ]
            processor.main()
        except Exception as e:
            print(f"Error during horizontal assembly: {e}")
            return 1

    if args.mode in ["reels", "all"]:
        print("--- Step 4: Vertical Reels Generation ---")
        try:
            import src.make_reels as reel_maker

            sys.argv = [
                reel_maker.__file__,
                "--video_path",
                args.video_path,
                "--json_dir",
                json_dir,
                "--output_dir",
                args.output_dir,
            ]
            if args.visualize:
                sys.argv.append("--visualize")
            reel_maker.main()
        except Exception as e:
            print(f"Error during reels generation: {e}")
            return 1
            
    elif args.mode == "pose":
        # Pose detection mode
        if not args.track_file or not args.video_path:
            print("Error: --track_file and --video_path are required for pose mode")
            return 1
            
        # Import and run pose detection
        try:
            from src.pose_detector import add_pose_to_track_json
            print("Pose detection mode selected")
            print(f"Track file: {args.track_file}")
            print(f"Video: {args.video_path}")
            print(f"Visualize: {args.visualize}")
            
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
        print("Analysis mode selected")
        print("This mode is not yet implemented")
        
    else:
        print("Hello from fast-volleyball-tracking-inference!")
        print("Use --mode to specify the processing mode")
        print("Available modes: track, pose, analyze")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())