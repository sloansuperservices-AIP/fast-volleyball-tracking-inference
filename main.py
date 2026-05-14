#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument("--mode", type=str, choices=["track", "pose", "analyze", "hub-track"],
                        default="track", help="Processing mode")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    parser.add_argument("--model_path", type=str, default="models/vballNetV1.onnx", 
                        help="Path to ONNX model file")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable visualization on display using cv2")
    
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
        # Ball tracking mode
        if not args.video_path:
            print("Error: --video_path is required for tracking mode")
            return 1
            
        # Import and run ball tracking
        try:
            from src.inference_onnx_seq_gray_v2 import main as track_main
            # Forward arguments to the tracking module
            sys.argv = [sys.argv[0], "--video_path", args.video_path, "--model_path", args.model_path, "--output_dir", args.output_dir]
            if args.visualize:
                sys.argv.append("--visualize")

            print("Ball tracking mode selected")
            print(f"Video: {args.video_path}")
            print(f"Model: {args.model_path}")

            track_main()
        except ImportError as e:
            print(f"Error importing tracking module: {e}")
            return 1
        except Exception as e:
            print(f"Error during ball tracking: {e}")
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
        # Analysis mode (Track calculation from CSV)
        if not args.video_path:
            print("Error: --video_path is required for analyze mode to resolve track metadata")
            return 1

        csv_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.video_path))[0], "ball.csv")
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found at {csv_path}. Run 'track' mode first.")
            return 1

        try:
            from src.track_calculator import main as analyze_main
            sys.argv = [sys.argv[0], "--csv_path", csv_path, "--output_dir", args.output_dir]
            print("Track analysis mode selected")
            print(f"CSV: {csv_path}")
            analyze_main()
        except ImportError as e:
            print(f"Error importing analysis module: {e}")
            return 1
        except Exception as e:
            print(f"Error during track analysis: {e}")
            return 1
        
    else:
        print("Hello from fast-volleyball-tracking-inference!")
        print("Use --mode to specify the processing mode")
        print("Available modes: track, pose, analyze, hub-track")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
