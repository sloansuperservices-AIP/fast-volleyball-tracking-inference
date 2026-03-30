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
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    
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
    
    # Determine sequence length from model filename or output shape
    output_shape = session.get_outputs()[0].shape
    # output_shape usually (1, C, H, W)
    channels = output_shape[1]

    is_grid = False
    if "grid" in model_path.lower() or (channels % 3 == 0 and output_shape[2] < 100):
        is_grid = True
        batch_size = channels // 3
        out_dim = batch_size
    else:
        out_dim = channels
        batch_size = channels

    if "seq15" in model_path.lower():
        batch_size = 15
        out_dim = 15
    elif "seq9" in model_path.lower():
        batch_size = 9
        out_dim = 9
    elif "seq5" in model_path.lower():
        batch_size = 5
        out_dim = 5
        
    print(f"✅ Model loaded: {model_path}")
    print(
        f"   Has GRU state: {has_gru}, Sequence length: {batch_size}, Output channels: {channels}, Type: {'Grid' if is_grid else 'Heatmap'}"
    )
    return session, has_gru, out_dim, h0_shape, batch_size, input_names, output_names, is_grid


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


def postprocess_heatmap(output, threshold, out_dim):
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
                results.append((1, cx, cy))
            else:
                results.append((0, 0, 0))
        else:
            results.append((0, 0, 0))
    return results


def postprocess_grid(output, threshold, out_dim, input_height, input_width):
    # output shape (1, out_dim*3, grid_H, grid_W)
    grid_h, grid_w = output.shape[2], output.shape[3]
    y_pred = np.split(output, out_dim, axis=1)
    y_pred = np.stack(y_pred, axis=2)
    y_pred = np.moveaxis(y_pred, 1, -1)

    conf_grid, x_offset_grid, y_offset_grid = np.split(y_pred, 3, axis=-1)
    conf_grid = np.squeeze(conf_grid, axis=-1)
    x_offset_grid = np.squeeze(x_offset_grid, axis=-1)
    y_offset_grid = np.squeeze(y_offset_grid, axis=-1)

    results = []
    # conf_grid shape (1, seq, grid_H, grid_W)
    for j in range(out_dim):
        curr_conf_grid = conf_grid[0][j]
        curr_x_offset_grid = x_offset_grid[0][j]
        curr_y_offset_grid = y_offset_grid[0][j]

        max_conf_val = np.max(curr_conf_grid)
        pred_row, pred_col = np.unravel_index(np.argmax(curr_conf_grid), curr_conf_grid.shape)

        if max_conf_val >= threshold:
            x_offset = curr_x_offset_grid[pred_row][pred_col]
            y_offset = curr_y_offset_grid[pred_row][pred_col]
            x_pred = int((x_offset + pred_col) * (input_width / grid_w))
            y_pred_val = int((y_offset + pred_row) * (input_height / grid_h))
            results.append((1, x_pred, y_pred_val))
        else:
            results.append((0, 0, 0))
    return results


def postprocess_output(output, threshold=0.5, input_height=288, input_width=512, out_dim=9, is_grid=False):
    if is_grid:
        return postprocess_grid(output, threshold, out_dim, input_height, input_width)
    else:
        return postprocess_heatmap(output, threshold, out_dim)


def draw_track(
    frame, track_points, current_color=(0, 0, 255), history_color=(255, 0, 0)
):
    import itertools
    points = list(itertools.islice(track_points, 0, len(track_points)-1))
    for point in points:
        if point is not None:
            cv2.circle(frame, point, 5, history_color, -1)
    if track_points and track_points[-1] is not None:
        cv2.circle(frame, track_points[-1], 5, current_color, -1)
    return frame


def run_inference(session, input_tensor, has_gru, h0, input_names, output_names):
    inputs = {input_names[0]: input_tensor}
    if has_gru:
        if len(input_names) < 2:
            raise ValueError("GRU model expects at least 2 inputs: images and h0")
        inputs[input_names[1]] = h0
    
    outputs = session.run(output_names, inputs)
    
    heatmaps = outputs[0]
    new_h0 = None
    if has_gru:
        if len(outputs) < 2:
            raise ValueError("GRU model should output at least 2 values: heatmaps and hn")
        new_h0 = outputs[1]
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
    
    model_session, has_gru, out_dim, h0_shape, batch_size, input_names, output_names, is_grid = load_onnx_model(args.model_path)

    if is_grid:
        input_width, input_height = 768, 432
    else:
        input_width, input_height = 512, 288

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
    
    h0 = np.zeros(h0_shape, dtype=np.float32) if has_gru and h0_shape else None

    def frame_reader():
        while cap.isOpened():
            read_frames(cap, frame_queue, batch_size)

    reader_thread = threading.Thread(target=frame_reader, daemon=True)
    reader_thread.start()

    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")
    exit_flag = False
    while True:
        start_time = time.time()

        frames = frame_queue.get()
        if frames is None:
            break

        processed_frames = preprocess_frames(frames, input_height, input_width)

        while len(frame_buffer) < batch_size:
            frame_buffer.append(
                processed_frames[0]
                if processed_frames
                else np.zeros((input_height, input_width), dtype=np.float32)
            )

        for pf in processed_frames:
            frame_buffer.append(pf)

        input_tensor = np.stack(frame_buffer, axis=2)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))

        output, new_h0 = run_inference(model_session, input_tensor, has_gru, h0, input_names, output_names)
        if has_gru and new_h0 is not None:
            h0 = new_h0

        predictions = postprocess_output(
            output, threshold=0.5, input_height=input_height, input_width=input_width, out_dim=out_dim, is_grid=is_grid
        )

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
                        exit_flag = True
                        break
                if out_writer is not None:
                    out_writer.write(vis_frame)
        if exit_flag:
            break

        end_time = time.time()
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
