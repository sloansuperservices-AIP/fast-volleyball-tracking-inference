import argparse
import cv2
import numpy as np
import pandas as pd
import openvino as ov
from collections import deque
import os
import time
from tqdm import tqdm
import threading
import queue


def is_headless():
    """Checks if the environment is headless."""
    return "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ


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
        "--model_xml", type=str, required=True, help="Path to OpenVINO model .xml file"
    )
    parser.add_argument(
        "--device", type=str, default="CPU", help="OpenVINO device: CPU, GPU, AUTO"
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


def load_ov_model(model_xml, device="CPU"):
    if not os.path.exists(model_xml):
        raise FileNotFoundError(f"Model not found: {model_xml}")

    core = ov.Core()
    model = core.read_model(model_xml)
    compiled_model = core.compile_model(model, device)

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    input_names = [inp.get_any_name() for inp in compiled_model.inputs]

    has_gru = "h0" in input_names
    h0_shape = None
    if has_gru:
        for inp in compiled_model.inputs:
            if inp.get_any_name() == "h0":
                h0_shape = inp.get_partial_shape()
                break

        # Convert PartialShape to tuple
        resolved_shape = []
        for dim in h0_shape:
            if dim.is_dynamic:
                resolved_shape.append(1) # Batch or hidden usually 1 or fixed
            else:
                resolved_shape.append(dim.get_length())
        h0_shape = tuple(resolved_shape)

    # Determine sequence length from model filename or shape
    if "seq15" in model_xml.lower():
        out_dim = 15
        batch_size = 15
    elif "seq9" in model_xml.lower():
        out_dim = 9
        batch_size = 9
    else:
        # Try to infer from input shape: [1, seq, H, W]
        input_shape = input_layer.get_shape()
        out_dim = input_shape[1]
        batch_size = input_shape[1]

    print(f"✅ OpenVINO Model loaded: {model_xml} on {device}")
    print(
        f"   Has GRU state: {has_gru}, Sequence length: {batch_size}, Output heatmaps: {out_dim}"
    )
    return compiled_model, has_gru, out_dim, h0_shape, batch_size, input_layer, output_layer


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
    csv_path = os.path.join(video_dir, "ball_ov.csv")
    pd.DataFrame(columns=["Frame", "Visibility", "X", "Y", "Radius"]).to_csv(
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
    # OpenVINO output might be [1, seq, H, W] or [1, H, W, seq] etc.
    # Usually it matches [1, out_dim, H, W] for our Heatmap models
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
                _, radius = cv2.minEnclosingCircle(largest_contour)
                results.append((1, cx, cy, radius))
            else:
                results.append((0, 0, 0, 0.0))
        else:
            results.append((0, 0, 0, 0.0))
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


def run_inference(compiled_model, input_tensor, has_gru, h0):
    inputs = {compiled_model.input(0): input_tensor}
    if has_gru:
        # Try to find h0 input by name if there are multiple inputs
        h0_input = None
        for inp in compiled_model.inputs:
            if "h0" in inp.get_any_name():
                h0_input = inp
                break
        if h0_input:
            inputs[h0_input] = h0
        else:
            inputs[compiled_model.input(1)] = h0

    infer_request = compiled_model.create_infer_request()
    results = infer_request.infer(inputs)

    # Results is a dictionary where keys are output ports
    # Find heatmap output (usually the largest one or containing 'heatmap'/'output')
    heatmaps = None
    new_h0 = None

    output_ports = compiled_model.outputs
    if len(output_ports) == 1:
        heatmaps = results[output_ports[0]]
    else:
        for port in output_ports:
            name = port.get_any_name()
            if "hn" in name or "output_1" in name: # Common names for GRU hidden state output
                new_h0 = results[port]
            else:
                # Default to heatmap for others, or specifically check name
                if heatmaps is None:
                    heatmaps = results[port]

    # Fallback if names didn't match
    if heatmaps is None:
        heatmaps = next(iter(results.values()))

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


def run_ov_inference_pipeline(video_path, model_xml, output_dir=None, visualize=False, only_csv=False, device="CPU", track_length=8):
    input_width, input_height = 512, 288

    compiled_model, has_gru, out_dim, h0_shape, batch_size, input_layer, output_layer = load_ov_model(
        model_xml, device
    )

    cap, frame_width, frame_height, fps, total_frames = initialize_video(
        video_path
    )
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    out_writer, _ = setup_output_writer(
        video_basename, output_dir, frame_width, frame_height, fps, only_csv
    )
    csv_path = setup_csv_file(video_basename, output_dir)

    frame_buffer = deque(maxlen=batch_size)
    track_points = deque(maxlen=track_length)
    frame_index = 0
    frame_queue = queue.Queue(maxsize=2)

    h0 = np.zeros(h0_shape, dtype=np.float32) if has_gru and h0_shape else None

    def frame_reader():
        while cap.isOpened():
            read_frames(cap, frame_queue, batch_size)

    reader_thread = threading.Thread(target=frame_reader, daemon=True)
    reader_thread.start()

    pbar = tqdm(total=total_frames, desc="Processing video (OpenVINO)", unit="frame")
    exit_flag = False
    if visualize and is_headless():
        print("Headless environment detected, disabling visualization.")
        visualize = False

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

        output, new_h0 = run_inference(compiled_model, input_tensor, has_gru, h0)
        if has_gru and new_h0 is not None:
            h0 = new_h0

        predictions = postprocess_output(
            output, input_height=input_height, input_width=input_width, out_dim=out_dim
        )

        for i, (visibility, x, y, radius) in enumerate(predictions[: len(frames)]):
            x_orig = x * frame_width / input_width if visibility else -1
            y_orig = y * frame_height / input_height if visibility else -1
            radius_orig = radius * (frame_width / input_width) if visibility else 0.0

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
                "Radius": radius_orig,
            }
            append_to_csv(result, csv_path)

            if visualize or out_writer is not None:
                vis_frame = frames[i].copy()
                vis_frame = draw_track(vis_frame, track_points)
                if visualize:
                    cv2.namedWindow("Ball Tracking (OV)", cv2.WINDOW_NORMAL)
                    cv2.imshow("Ball Tracking (OV)", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        exit_flag = True
                        break
                if out_writer is not None:
                    out_writer.write(vis_frame)
        if exit_flag:
            break

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
