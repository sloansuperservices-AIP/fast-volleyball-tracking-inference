#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument("--mode", type=str, choices=["track", "pose", "analyze"],
                        default="track", help="Processing mode")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    parser.add_argument("--model_path", type=str, default="models/vballNetV1.onnx", 
                        help="Path to ONNX model file")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable visualization on display using cv2")
    
    args = parser.parse_args()
    
    if args.mode == "track":
        # Ball tracking mode
        if not args.video_path:
            print("Error: --video_path is required for tracking mode")
            return 1
            
        # Import and run ball tracking
        try:
            from src.inference_onnx_seq_gray_v2 import main as track_main
            print("Ball tracking mode selected")
            print(f"Video: {args.video_path}")
            print(f"Model: {args.model_path}")
            print(f"Visualize: {args.visualize}")

            # Patch sys.argv to pass arguments to the tracking script
            sys.argv = [
                sys.argv[0],
                "--video_path", args.video_path,
                "--model_path", args.model_path,
                "--output_dir", args.output_dir,
            ]
            if args.visualize:
                sys.argv.append("--visualize")

            return track_main()
        except ImportError as e:
            print(f"Error importing tracking module: {e}")
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