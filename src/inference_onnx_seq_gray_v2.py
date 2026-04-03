import argparse
import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort
from collections import deque
import os
import time
from tqdm import tqdm
import threading
import queue


def parse_args():
    parser = argparse.ArgumentParser(
        description="Volleyball ball detection and tracking with ONNX"
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
        "--model_path", type=str, required=True, help="Path to ONNX model file"
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


def load_onnx_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    session = ort.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_meta = session.get_inputs()[0]
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    
    # Try to detect input resolution from model
    input_shape = input_meta.shape
    # input_shape can be (batch, seq, h, w)
    try:
        if len(input_shape) == 4:
            input_height = input_shape[2] if isinstance(input_shape[2], int) else 288
            input_width = input_shape[3] if isinstance(input_shape[3], int) else 512
        else:
            input_height, input_width = 288, 512
    except:
        input_height, input_width = 288, 512

    has_gru = "h0" in input_names
    h0_shape = None
    if has_gru:
        for inp in session.get_inputs():
            if inp.name == "h0":
                h0_shape = inp.shape
                break
        if h0_shape is None:
            raise ValueError("Could not determine h0 shape for GRU model.")
        resolved_shape = []
        for dim in h0_shape:
            if isinstance(dim, str) or dim is None:
                if dim in ["batch", "batch_size", None]:
                    resolved_shape.append(1)
                elif "hidden" in str(dim).lower():
                    resolved_shape.append(512)
                else:
                    raise ValueError(
                        f"Unknown symbolic dimension '{dim}' in h0_shape: {h0_shape}"
                    )
            else:
                resolved_shape.append(dim)
        h0_shape = tuple(resolved_shape)
    
    # Determine sequence length from model filename or input channels
    if "seq15" in model_path.lower():
        batch_size = 15
    elif "seq9" in model_path.lower():
        batch_size = 9
    elif isinstance(input_shape[1], int):
        batch_size = input_shape[1]
    else:
        batch_size = 9
        
    out_dim = batch_size
    print(f"✅ Model loaded: {model_path}")
    print(
        f"   Input: {input_width}x{input_height}, Batch: {batch_size}, Has GRU: {has_gru}"
    )
    return session, has_gru, out_dim, h0_shape, batch_size, input_names, output_names, input_width, input_height


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
    output_path = os.path.join(video_dir, "predict.mp4")
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
    pd.DataFrame(columns=["Frame", "Visibility", "X", "Y"]).to_csv(
        csv_path, index=False
    )
    return csv_path


def append_to_csv(result, csv_path):
    if csv_path is None:
        return
    pd.DataFrame([result]).to_csv(csv_path, mode="a", header=False, index=False)


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
    batch, channels, h, w = output.shape

    # Grid models have out_dim*3 channels and small h,w (grid size)
    is_grid = (channels == out_dim * 3)

    if is_grid:
        # Reshape to (batch, seq, 3, h, w) where 3 is (conf, x_offset, y_offset)
        grid_out = output.reshape(batch, out_dim, 3, h, w)
        grid_size_w = input_width / w
        grid_size_h = input_height / h

        for frame_idx in range(out_dim):
            conf_grid = grid_out[0, frame_idx, 0, :, :]
            x_offset_grid = grid_out[0, frame_idx, 1, :, :]
            y_offset_grid = grid_out[0, frame_idx, 2, :, :]

            max_conf = np.max(conf_grid)
            if max_conf >= threshold:
                pred_row, pred_col = np.unravel_index(np.argmax(conf_grid), conf_grid.shape)
                x_offset = x_offset_grid[pred_row, pred_col]
                y_offset = y_offset_grid[pred_row, pred_col]

                cx = int((pred_col + x_offset) * grid_size_w)
                cy = int((pred_row + y_offset) * grid_size_h)
                results.append((1, cx, cy))
            else:
                results.append((0, 0, 0))
    else:
        # Heatmap logic
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
                    results.append((1, cx, cy))
                else:
                    results.append((0, 0, 0))
            else:
                results.append((0, 0, 0))
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


def run_inference(session, input_tensor, has_gru, h0, input_names, output_names):
    """Helper для инференса с поддержкой GRU."""
    inputs = {input_names[0]: input_tensor}  # Основной инпут (кадры)
    if has_gru:
        if len(input_names) < 2:
            raise ValueError("GRU model expects at least 2 inputs: images and h0")
        inputs[input_names[1]] = h0  # Добавляем h0
    
    outputs = session.run(output_names, inputs)
    
    heatmaps = outputs[0]  # Первый аутпут — heatmaps
    new_h0 = None
    if has_gru:
        if len(outputs) < 2:
            raise ValueError("GRU model should output at least 2 values: heatmaps and hn")
        new_h0 = outputs[1]  # Второе — новое скрытое состояние
    return heatmaps, new_h0


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
    
    model_session, has_gru, out_dim, h0_shape, batch_size, input_names, output_names, input_width, input_height = load_onnx_model(args.model_path)

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
    
    # Инициализация скрытого состояния для GRU
    h0 = np.zeros(h0_shape, dtype=np.float32) if has_gru and h0_shape else None

    # Start frame reading thread
    def frame_reader():
        while cap.isOpened():
            read_frames(cap, frame_queue, batch_size)

    reader_thread = threading.Thread(target=frame_reader, daemon=True)
    reader_thread.start()

    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")
    exit_flag = False
    while True:
        start_time = time.time()

        # Get batch of frames
        frames = frame_queue.get()
        if frames is None:
            break

        # Preprocess frames in batch
        processed_frames = preprocess_frames(frames, input_height, input_width)

        # Fill buffer if not enough frames
        while len(frame_buffer) < batch_size:
            frame_buffer.append(
                processed_frames[0]
                if processed_frames
                else np.zeros((input_height, input_width), dtype=np.float32)
            )

        # Update buffer with new frames
        for pf in processed_frames:
            frame_buffer.append(pf)

        # Prepare input tensor
        input_tensor = np.stack(frame_buffer, axis=2)  # (height, width, seq_len)
        input_tensor = np.expand_dims(input_tensor, axis=0)  # (1, height, width, seq_len)
        input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))  # (1, seq_len, 288, 512)

        # Run inference with GRU support
        output, new_h0 = run_inference(model_session, input_tensor, has_gru, h0, input_names, output_names)
        if has_gru and new_h0 is not None:
            h0 = new_h0  # Обновляем состояние для следующего батча

        # Process predictions for all frames
        predictions = postprocess_output(
            output, input_height=input_height, input_width=input_width, out_dim=out_dim
        )

        # Save results and visualize for each frame in the batch
        for i, (visibility, x, y) in enumerate(predictions[: len(frames)]):
            x_orig = x * frame_width / input_width if visibility else -1
            y_orig = y * frame_height / input_height if visibility else -1

            if visibility:
                track_points.append((int(x_orig), int(y_orig)))
            else:
                if track_points:
                    track_points.popleft()

            result = {
                "Frame": frame_index + i,
                "Visibility": visibility,
                "X": int(x_orig),
                "Y": int(y_orig),
            }
            append_to_csv(result, csv_path)

            if args.visualize or out_writer is not None:
                vis_frame = frames[i].copy()
                vis_frame = draw_track(vis_frame, track_points)
                if args.visualize:
                    cv2.namedWindow("Ball Tracking", cv2.WINDOW_NORMAL)
                    cv2.imshow("Ball Tracking", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        exit_flag = True  # Set flag to exit
                        break
                if out_writer is not None:
                    out_writer.write(vis_frame)
        if exit_flag:
            break

        end_time = time.time()
        batch_time = end_time - start_time
        batch_fps = len(frames) / batch_time if batch_time > 0 else 0
        pbar.update(len(frames))
        frame_index += len(frames)

    pbar.close()
    cap.release()
    if out_writer is not None:
        out_writer.release()
    if args.visualize:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
