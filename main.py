#!/usr/bin/env python3
"""
Unified entry point for the volleyball tracking pipeline.
Supports: detect, track, combined, reels, all, pose, and hub-track.
"""

import argparse
import os
import sys
import logging
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOG = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Volleyball Tracking Pipeline")
    parser.add_argument("--mode", type=str,
                        choices=["detect", "track", "combined", "reels", "all", "pose", "hub-track", "analyze"],
                        default="all", help="Processing mode")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--model_path", type=str, default="models/VballNetV1_seq9_grayscale_330_h288_w512.onnx",
                        help="Path to detection model file")
    parser.add_argument("--output_dir", type=str, default="output", 
                        help="Directory to save output files")
    parser.add_argument("--visualize", action="store_true", 
                        help="Enable visualization")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")

    # Pose mode arguments
    parser.add_argument("--track_file", type=str, help="Path to track JSON file (for pose mode)")
    
    # Hub specific arguments
    parser.add_argument("--hub_model", type=str, default="https://hub.ultralytics.com/models/ITKRtcQHITZrgT2ZNpRq",
                        help="Ultralytics Hub model URL or ID")
    parser.add_argument("--api_key", type=str,
                        default=os.getenv("ULTRALYTICS_HUB_API_KEY", "5ea02b4238fc9528408b8c36dcdb3834e11a9cbf58"),
                        help="Ultralytics Hub API key")

    args = parser.parse_args()

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    video_out_dir = os.path.join(args.output_dir, video_name)
    ball_csv = os.path.join(video_out_dir, "ball.csv")
    tracks_dir = os.path.join(video_out_dir, "tracks")

    if args.mode in ["detect", "all"]:
        LOG.info("--- Step 1: Ball Detection ---")
        from src.inference_onnx_seq_gray_v2 import load_onnx_model, initialize_video, setup_output_writer, setup_csv_file, BallTrackState, frame_reader, preprocess_frames, run_inference, decode_predictions, draw_track
        import cv2
        import numpy as np
        import threading
        import queue
        from tqdm import tqdm
        
        (model_session, has_gru, out_dim, h0_shape, batch_size, input_names, output_names, model_params) = load_onnx_model(args.model_path)
        cap, frame_width, frame_height, fps, total_frames = initialize_video(args.video_path)
        out_writer, _ = setup_output_writer(video_name, args.output_dir, frame_width, frame_height, int(args.fps), not args.visualize)
        csv_path = setup_csv_file(video_name, args.output_dir)
        
        frame_buffer = []
        track_state = BallTrackState(maxlen=8, max_missing=8)
        frame_index = 0
        frame_queue = queue.Queue(maxsize=2)
        error_queue = queue.Queue()
        stop_event = threading.Event()
        h0 = np.zeros(h0_shape, dtype=np.float32) if has_gru and h0_shape else None

        reader_thread = threading.Thread(target=frame_reader, args=(cap, frame_queue, batch_size, stop_event, error_queue), daemon=True)
        reader_thread.start()

        pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")
        try:
            while not stop_event.is_set():
                if not error_queue.empty(): raise error_queue.get()
                try: frames = frame_queue.get(timeout=0.5)
                except queue.Empty:
                    if not reader_thread.is_alive(): break
                    continue
                if frames is None: break

                proc_frames = preprocess_frames(frames, model_params["input_height"], model_params["input_width"])
                while len(frame_buffer) < batch_size: frame_buffer.append(proc_frames[0] if proc_frames else np.zeros((model_params["input_height"], model_params["input_width"]), dtype=np.float32))
                for pf in proc_frames: frame_buffer.append(pf)
                frame_buffer = frame_buffer[-batch_size:]

                input_tensor = np.transpose(np.expand_dims(np.stack(frame_buffer, axis=2), axis=0), (0, 3, 1, 2))
                output, new_h0 = run_inference(model_session, input_tensor, has_gru, h0, input_names, output_names)
                if has_gru and new_h0 is not None: h0 = new_h0

                predictions = decode_predictions(output, model_params, 0.5)
                for i, (vis, x, y) in enumerate(predictions[:len(frames)]):
                    xo = x * frame_width / model_params["input_width"] if vis else -1
                    yo = y * frame_height / model_params["input_height"] if vis else -1
                    track_state.update((int(xo), int(yo)) if vis else None)
                    if track_state.is_lost(): track_state.reset()
                    from src.inference_onnx_seq_gray_v2 import append_to_csv
                    append_to_csv({"Frame": frame_index + i, "Visibility": vis, "X": int(xo), "Y": int(yo)}, csv_path)
                    if args.visualize or out_writer:
                        vf = draw_track(frames[i].copy(), track_state.points())
                        if args.visualize:
                            cv2.imshow("Ball Detection", vf)
                            if cv2.waitKey(1) & 0xFF == ord("q"): stop_event.set(); break
                        if out_writer: out_writer.write(vf)
                pbar.update(len(frames))
                frame_index += len(frames)
        finally:
            stop_event.set(); reader_thread.join(timeout=2.0); pbar.close(); cap.release()
            if out_writer: out_writer.release()
            if args.visualize: cv2.destroyAllWindows()

    if args.mode in ["track", "all"]:
        LOG.info("--- Step 2: Track Calculation ---")
        from src.track_calculator import TrackCalculator, TrackCalculatorConfig
        config = TrackCalculatorConfig(csv_path=ball_csv, output_dir=args.output_dir, fps=args.fps, max_distance=200.0, min_duration_sec=1.0, max_x_displacement=20.0, min_y_displacement=50.0, bounce_frames=10, court_json_path=None, video_width=None, video_height=None)
        calculator = TrackCalculator(config)
        calculator.run()

    if args.mode in ["combined", "all"]:
        LOG.info("--- Step 3: Combined Video Assembly ---")
        from src.track_processor import TrackProcessor
        processor = TrackProcessor(json_dir=tracks_dir, video_path=args.video_path, output_path=os.path.join(video_out_dir, "combined.mp4"), fps=args.fps)
        processor._load_tracks_from_json()
        processor.visualize_tracks()

    if args.mode in ["reels", "all"]:
        LOG.info("--- Step 4: Vertical Reels Generation ---")
        from src.make_reels import crop_and_save_track, load_single_track
        reels_dir = os.path.join(video_out_dir, "reels")
        os.makedirs(reels_dir, exist_ok=True)
        if os.path.exists(tracks_dir):
            for f in sorted(os.listdir(tracks_dir)):
                if f.endswith(".json"):
                    track = load_single_track(os.path.join(tracks_dir, f))
                    crop_and_save_track(args.video_path, track, os.path.join(reels_dir, f"reel_{f.replace('.json', '.mp4')}"), visualize=args.visualize)

    if args.mode == "pose":
        LOG.info("--- Pose Detection ---")
        if not args.track_file:
            LOG.error("--track_file is required for pose mode")
            sys.exit(1)
        # Assuming src/process_track_pose.py exists and is integrated similarly
        pass

    if args.mode == "hub-track":
        LOG.info("--- Hub Tracking ---")
        from src.hub_inference import run_hub_inference
        run_hub_inference(video_path=args.video_path, model_url=args.hub_model, api_key=args.api_key, output_dir=args.output_dir, visualize=args.visualize)

    LOG.info("Processing complete.")

if __name__ == "__main__":
    main()
