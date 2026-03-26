#!/usr/bin/env python3
"""
Main entry point for the fast volleyball tracking inference system.
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Fast Volleyball Tracking Inference")
    parser.add_argument("--mode", type=str, choices=["track", "track-ov", "pose", "analyze", "hub-track"],
                        default="track", help="Processing mode")
    parser.add_argument("--video_path", type=str, help="Path to input video file")
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    parser.add_argument("--model_path", type=str, default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
                        help="Path to ONNX or OpenVINO model file")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable visualization on display using cv2")
    parser.add_argument("--only_csv", action="store_true",
                        help="Save only CSV, skip video output")
    
    # OpenVINO specific
    parser.add_argument("--device", type=str, default="CPU", help="Device for OpenVINO (CPU, GPU, AUTO)")

    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY"),
                        help="Ultralytics Hub API key")

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
            print("Hub tracking mode selected")
            run_hub_inference(
                video_path=args.video_path,
                model_url=args.hub_model,
                api_key=args.api_key,
                output_dir=args.output_dir,
                visualize=args.visualize
            )
        except Exception as e:
            print(f"Error during hub inference: {e}")
            return 1

    elif args.mode == "track":
        if not args.video_path:
            print("Error: --video_path is required for tracking mode")
            return 1
            
        try:
            from src.inference_onnx_seq_gray_v2 import main as track_main
            # Patch sys.argv to pass arguments to track_main
            sys.argv = [sys.argv[0], "--video_path", args.video_path, "--model_path", args.model_path, "--output_dir", args.output_dir]
            if args.visualize:
                sys.argv.append("--visualize")
            if args.only_csv:
                sys.argv.append("--only_csv")

            print(f"Starting ONNX tracking: {args.video_path}")
            track_main()
        except Exception as e:
            print(f"Error during tracking: {e}")
            return 1

    elif args.mode == "track-ov":
        if not args.video_path:
            print("Error: --video_path is required for track-ov mode")
            return 1
            
        try:
            from src.inference_openvino_seq_gray_v2 import main as track_ov_main
            sys.argv = [sys.argv[0], "--video_path", args.video_path, "--model_xml", args.model_path, "--output_dir", args.output_dir, "--device", args.device]
            if args.visualize:
                sys.argv.append("--visualize")
            if args.only_csv:
                sys.argv.append("--only_csv")

            print(f"Starting OpenVINO tracking: {args.video_path} on {args.device}")
            track_ov_main()
        except Exception as e:
            print(f"Error during OpenVINO tracking: {e}")
            return 1

    elif args.mode == "pose":
        if not args.track_file or not args.video_path:
            print("Error: --track_file and --video_path are required for pose mode")
            return 1
            
        try:
            from src.pose_detector import add_pose_to_track_json
            print(f"Starting pose detection for {args.track_file}")
            add_pose_to_track_json(
                track_file=args.track_file,
                video_path=args.video_path,
                output_dir=args.output_dir,
                visualize=args.visualize
            )
        except Exception as e:
            print(f"Error during pose detection: {e}")
            return 1
            
    elif args.mode == "analyze":
        print("Analysis mode selected - not yet fully implemented in main entry point")
        
    else:
        print("Available modes: track, track-ov, pose, analyze, hub-track")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
