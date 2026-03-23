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
    parser.add_argument("--model_path", type=str, default="models/VballNetGridV1b_seq9_grayscale_best.onnx",
                        help="Path to ONNX/OpenVINO model file")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable visualization on display using cv2")
    parser.add_argument("--device", type=str, default="CPU",
                        help="Device for OpenVINO inference (CPU, GPU, AUTO)")
    
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
        # Ball tracking mode (ONNX)
        if not args.video_path:
            print("Error: --video_path is required for tracking mode")
            return 1
            
        try:
            # Check if it's a seq grayscale model or legacy
            model_name = os.path.basename(args.model_path).lower()
            if "grayscale" in model_name and "seq" in model_name:
                from src.inference_onnx_seq_gray_v2 import run_tracking
                print(f"Ball tracking mode (ONNX Seq Gray) selected")
                print(f"Video: {args.video_path}")
                print(f"Model: {args.model_path}")

                run_tracking(
                    video_path=args.video_path,
                    model_path=args.model_path,
                    output_dir=args.output_dir,
                    visualize=args.visualize
                )
            else:
                from src.inference_onnx import main as track_legacy_main
                print(f"Ball tracking mode (ONNX Legacy) selected")
                print(f"Video: {args.video_path}")
                print(f"Model: {args.model_path}")

                # Legacy still uses sys.argv for now
                sys.argv = [
                    sys.argv[0],
                    "--video_path", args.video_path,
                    "--model_path", args.model_path,
                    "--output_dir", args.output_dir
                ]
                if args.visualize:
                    sys.argv.append("--visualize")
                track_legacy_main()

        except ImportError as e:
            print(f"Error importing tracking module: {e}")
            return 1
        except Exception as e:
            print(f"Error during tracking: {e}")
            return 1

    elif args.mode == "track-ov":
        # Ball tracking mode (OpenVINO)
        if not args.video_path:
            print("Error: --video_path is required for track-ov mode")
            return 1

        try:
            from src.inference_openvino_seq_gray_v2 import run_openvino_tracking
            print(f"Ball tracking mode (OpenVINO) selected")
            print(f"Video: {args.video_path}")
            print(f"Model: {args.model_path}")
            print(f"Device: {args.device}")

            run_openvino_tracking(
                video_path=args.video_path,
                model_xml=args.model_path,
                output_dir=args.output_dir,
                device=args.device,
                visualize=args.visualize
            )
        except ImportError as e:
            print(f"Error importing OpenVINO tracking module: {e}")
            return 1
        except Exception as e:
            print(f"Error during OpenVINO tracking: {e}")
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
        print("Available modes: track, track-ov, pose, analyze, hub-track")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
