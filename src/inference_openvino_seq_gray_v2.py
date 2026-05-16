import argparse
import cv2
import numpy as np
import pandas as pd
from openvino.runtime import Core
from collections import deque
import os
import time
from tqdm import tqdm
import threading
import queue
import re
from pathlib import Path


def infer_model_params(model_path):
    """
    Detects sequence length and model type from filename.
    """
    model_name = Path(model_path).name.lower()

    # Sequence length detection
    seq_match = re.search(r'seq(\d+)', model_name)
    input_seq = int(seq_match.group(1)) if seq_match else 3

    # Grid model detection
    is_grid = 'grid' in model_name

    return input_seq, is_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Volleyball ball detection and tracking with OpenVINO"
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--track_length", type=int, default=8, help="Length of the ball track"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output video and CSV",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to OpenVINO XML model file"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Enable visualization on display",
    )
    parser.add_argument(
        "--only_csv",
        action="store_true",
        default=False,
        help="Save only CSV, skip video output",
    )
    return parser.parse_args()


def load_openvino_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    ie = Core()
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    input_seq, is_grid = infer_model_params(model_path)
    out_dim = input_seq
    batch_size = input_seq

    print(f"✅ Model loaded (OpenVINO): {model_path}")
    print(
        f"   Sequence length: {batch_size}, Output heatmaps: {out_dim}, Grid mode: {is_grid}"
    )
    return compiled_model, input_layer, output_layer, out_dim, batch_size, is_grid


def initialize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frame_width, frame_height, fps, total_frames


def setup_output_writer(
    video_basename, output_dir, frame_width, frame_height, fps, only_csv
):
    if output_dir is None or only_csv:
        return None, None
    video_dir = os.path.join(output_dir, video_basename)
    os.makedirs(video_dir, exist_ok=True)
    output_path = os.path.join(video_dir, "predict_ov.mp4")
    out_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
    )
    return out_writer, output_path


def setup_csv_file(video_basename, output_dir):
    if output_dir is None:
        return None
    video_dir = os.path.join(output_dir, video_basename)
    os.makedirs(video_dir, exist_ok=True)
    csv_path = os.path.join(video_dir, "ball.csv")
    pd.DataFrame(columns=["Frame", "Visibility", "X", "Y", "Radius"]).to_csv(
        csv_path, index=False
    )
    return csv_path


def append_to_csv(result, csv_path):
    if csv_path is None:
        return
    cols = ["Frame", "Visibility", "X", "Y", "Radius"]
    data = {col: result.get(col, 0) for col in cols}
    pd.DataFrame([data]).to_csv(csv_path, mode="a", header=False, index=False)


def preprocess_frames(frames, input_height=288, input_width=512):
    processed = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (input_width, input_height))
        frame = frame.astype(np.float32) / 255.0
        processed.append(frame)
    return processed


def postprocess_output(
    output, threshold=0.5, input_height=288, input_width=512, out_dim=9
):
    results = []
    for frame_idx in range(out_dim):
        heatmap = output[0, frame_idx, :, :]
        _, binary = cv2.threshold(heatmap, threshold, 1.0, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            (binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                results.append((1, cx, cy, round(radius, 2)))
            else:
                results.append((0, 0, 0, 0))
        else:
            results.append((0, 0, 0, 0))
    return results


def draw_track(
    frame, track_points, current_color=(0, 0, 255), history_color=(255, 0, 0)
):
    for point in list(track_points)[:-1]:
        if point is not None:
            cv2.circle(frame, point, 5, history_color, -1)
    if track_points and track_points[-1] is not None:
        cv2.circle(frame, track_points[-1], 5, current_color, -1)
    return frame


def read_frames(cap, frame_queue, max_frames):
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    if frames:
        frame_queue.put(frames)
    else:
        frame_queue.put(None)


def main():
    args = parse_args()
    input_width, input_height = 512, 288

    compiled_model, input_layer, output_layer, out_dim, batch_size, is_grid = load_openvino_model(args.model_path)

    cap, frame_width, frame_height, fps, total_frames = initialize_video(
        args.video_path
    )
    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    out_writer, _ = setup_output_writer(
        video_basename, args.output_dir, frame_width, frame_height, fps, args.only_csv
    )
    csv_path = setup_csv_file(video_basename, args.output_dir)

    frame_buffer = deque(maxlen=batch_size)
    track_points = deque(maxlen=args.track_length)
    frame_index = 0
    frame_queue = queue.Queue(maxsize=2)

    def frame_reader():
        while cap.isOpened():
            read_frames(cap, frame_queue, batch_size)

    reader_thread = threading.Thread(target=frame_reader, daemon=True)
    reader_thread.start()

    pbar = tqdm(total=total_frames, desc="Processing video (OpenVINO)", unit="frame")
    exit_flag = False
    while True:
        frames = frame_queue.get()
        if frames is None:
            break

        processed_frames = preprocess_frames(frames, input_height, input_width)

        while len(frame_buffer) < batch_size:
            frame_buffer.append(processed_frames[0] if processed_frames else np.zeros((input_height, input_width), dtype=np.float32))

        for pf in processed_frames:
            frame_buffer.append(pf)

        input_tensor = np.stack(frame_buffer, axis=2)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))

        results_ov = compiled_model([input_tensor])[output_layer]

        predictions = postprocess_output(
            results_ov, input_height=input_height, input_width=input_width, out_dim=out_dim
        )

        for i, pred in enumerate(predictions[: len(frames)]):
            visibility, x, y, radius = pred[0], pred[1], pred[2], pred[3]
            x_orig = x * frame_width / input_width if visibility else -1
            y_orig = y * frame_height / input_height if visibility else -1

            if visibility:
                track_points.append((int(x_orig), int(y_orig)))
            else:
                if track_points: track_points.popleft()

            result = {
                "Frame": frame_index + i,
                "Visibility": visibility,
                "X": int(x_orig),
                "Y": int(y_orig),
                "Radius": radius
            }
            append_to_csv(result, csv_path)

            if args.visualize or out_writer is not None:
                vis_frame = frames[i].copy()
                vis_frame = draw_track(vis_frame, track_points)
                if args.visualize:
                    cv2.imshow("Ball Tracking (OV)", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        exit_flag = True
                        break
                if out_writer is not None:
                    out_writer.write(vis_frame)
        if exit_flag: break
        pbar.update(len(frames))
        frame_index += len(frames)

    pbar.close()
    cap.release()
    if out_writer is not None: out_writer.release()
    if args.visualize: cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
