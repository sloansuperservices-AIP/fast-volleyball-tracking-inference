import argparse
import cv2
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
import threading
import queue
import logging
from collections import deque

try:
    from openvino.runtime import Core
except ImportError:
    try:
        from openvino import Core
    except ImportError:
        Core = None

# Configure logging
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("inference_ov")

# Ball size detection constants
BALL_SIZE_HISTORY = 12
BALL_RAW_SIZE_HISTORY = 5
BALL_TREND_FRAMES = 3
BALL_RADIUS_MIN = 3
BALL_RADIUS_MAX = 40
BALL_ROI_HALF_SIZE = 48


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
        "--model_path", type=str, required=True, help="Path to OpenVINO model file (.xml)"
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
    parser.add_argument(
        "--device", type=str, default="CPU", help="OpenVINO device (CPU, GPU, etc.)"
    )
    return parser.parse_args()


def load_ov_model(model_path, device="CPU"):
    if Core is None:
        raise ImportError("OpenVINO is not installed.")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    core = Core()
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device)

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    input_names = [inp.get_any_name() for inp in compiled_model.inputs]
    output_names = [out.get_any_name() for out in compiled_model.outputs]

    has_gru = "h0" in input_names
    h0_shape = None
    if has_gru:
        for inp in compiled_model.inputs:
            if inp.get_any_name() == "h0":
                h0_shape = inp.get_partial_shape().get_shape()
                break

    # Determine sequence length from model filename
    if "seq15" in model_path.lower():
        out_dim = 15
        batch_size = 15
    elif "seq9" in model_path.lower():
        out_dim = 9
        batch_size = 9
    else:
        out_dim = 3
        batch_size = 3

    print(f"✅ OpenVINO Model loaded: {model_path} on {device}")
    print(
        f"   Has GRU state: {has_gru}, Sequence length: {batch_size}, Output heatmaps: {out_dim}, h0 shape: {h0_shape if has_gru else 'N/A'}"
    )
    return compiled_model, has_gru, out_dim, h0_shape, batch_size, input_names, output_names


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
    for frame_idx in range(out_dim):  # Process all heatmaps
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


def build_motion_mask(prev_gray, gray):
    diff = cv2.absdiff(prev_gray, gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, motion_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
    return motion_mask


def contour_narrow_radius(contour):
    if len(contour) < 5:
        (_, _), radius = cv2.minEnclosingCircle(contour)
        return float(radius)

    (_, _), (width, height), _ = cv2.minAreaRect(contour)
    narrow_diameter = min(width, height)
    if narrow_diameter <= 0:
        (_, _), radius = cv2.minEnclosingCircle(contour)
        return float(radius)
    return float(narrow_diameter) / 2.0


def fallback_radius(size_state):
    smoothed_radius = size_state["smoothed_radius"]
    if smoothed_radius > 0:
        return int(round(smoothed_radius))
    filtered_history = size_state["filtered_history"]
    if not filtered_history:
        return 0
    return int(round(float(np.median(filtered_history))))


def filter_ball_radius(radius, size_state):
    if radius <= 0:
        return fallback_radius(size_state)

    raw_history = size_state["raw_history"]
    filtered_history = size_state["filtered_history"]
    raw_history.append(radius)

    if not filtered_history:
        filtered_history.append(radius)
        size_state["smoothed_radius"] = float(radius)
        return radius

    baseline = size_state["smoothed_radius"]
    if baseline <= 0:
        baseline = float(np.median(filtered_history))

    trend_window = list(raw_history)[-BALL_TREND_FRAMES:]
    trend_confirmed = False
    if len(trend_window) == BALL_TREND_FRAMES:
        upper_shift = [value > baseline for value in trend_window]
        lower_shift = [value < baseline for value in trend_window]
        trend_confirmed = all(upper_shift) or all(lower_shift)

    target_radius = float(radius)
    if trend_confirmed:
        target_radius = float(np.median(trend_window))
    else:
        max_deviation = max(3.0, baseline * 0.55)
        target_radius = float(
            np.clip(radius, baseline - max_deviation, baseline + max_deviation)
        )

    alpha = 0.6 if trend_confirmed else 0.3
    smoothed_radius = baseline * (1.0 - alpha) + target_radius * alpha
    smoothed_radius = float(np.clip(smoothed_radius, BALL_RADIUS_MIN, BALL_RADIUS_MAX))

    accepted_radius = int(round(smoothed_radius))
    filtered_history.append(accepted_radius)
    size_state["smoothed_radius"] = smoothed_radius
    return accepted_radius


def estimate_ball_radius(prev_gray, gray, x_orig, y_orig, size_state):
    if prev_gray is None or x_orig < 0 or y_orig < 0:
        return 0, None

    motion_mask = build_motion_mask(prev_gray, gray)
    x1 = max(0, x_orig - BALL_ROI_HALF_SIZE)
    y1 = max(0, y_orig - BALL_ROI_HALF_SIZE)
    x2 = min(gray.shape[1], x_orig + BALL_ROI_HALF_SIZE)
    y2 = min(gray.shape[0], y_orig + BALL_ROI_HALF_SIZE)
    roi = motion_mask[y1:y2, x1:x2]
    if roi.size == 0:
        return fallback_radius(size_state), None

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback_radius(size_state), None

    center = np.array([x_orig - x1, y_orig - y1], dtype=np.float32)
    best_contour = None
    best_score = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 8:
            continue
        (cx, cy), _ = cv2.minEnclosingCircle(contour)
        radius = contour_narrow_radius(contour)
        if radius < BALL_RADIUS_MIN or radius > BALL_RADIUS_MAX:
            continue
        distance = np.linalg.norm(np.array([cx, cy], dtype=np.float32) - center)
        score = distance - area * 0.02
        if best_score is None or score < best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return fallback_radius(size_state), None

    radius = contour_narrow_radius(best_contour)
    filtered_radius = filter_ball_radius(int(round(radius)), size_state)
    contour_global = best_contour + np.array([[[x1, y1]]], dtype=np.int32)
    return filtered_radius, contour_global


def render_prediction(frame, points, visibility, x_orig, y_orig, radius, contour):
    vis_frame = draw_track(frame.copy(), points)
    if visibility:
        if contour is not None:
            cv2.drawContours(vis_frame, [contour], -1, (0, 255, 255), 1)
        draw_radius = radius if radius > 0 else 8
        cv2.circle(vis_frame, (x_orig, y_orig), draw_radius, (0, 255, 0), 2)
        cv2.putText(
            vis_frame,
            f"R:{radius}" if radius > 0 else "R:n/a",
            (x_orig + draw_radius + 4, y_orig - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return vis_frame


def run_inference(compiled_model, input_tensor, has_gru, h0, input_names):
    inputs = {input_names[0]: input_tensor}
    if has_gru:
        inputs["h0"] = h0

    results = compiled_model(inputs)

    # Heatmaps are usually the first output
    heatmaps = list(results.values())[0]
    new_h0 = None
    if has_gru:
        # Assuming h0 is the second output
        new_h0 = list(results.values())[1]

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
    input_width, input_height = 512, 288

    compiled_model, has_gru, out_dim, h0_shape, batch_size, input_names, output_names = load_ov_model(
        args.model_path, args.device
    )

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

    size_state = {
        "filtered_history": deque(maxlen=BALL_SIZE_HISTORY),
        "raw_history": deque(maxlen=BALL_RAW_SIZE_HISTORY),
        "smoothed_radius": 0.0,
    }
    prev_gray = None

    # Initialize GRU state
    h0 = np.zeros(h0_shape, dtype=np.float32) if has_gru and h0_shape else None

    def frame_reader():
        while cap.isOpened():
            read_frames(cap, frame_queue, batch_size)

    reader_thread = threading.Thread(target=frame_reader, daemon=True)
    reader_thread.start()

    pbar = tqdm(total=total_frames, desc="Processing video (OpenVINO)", unit="frame")
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

        output, new_h0 = run_inference(compiled_model, input_tensor, has_gru, h0, input_names)
        if has_gru and new_h0 is not None:
            h0 = new_h0

        predictions = postprocess_output(
            output, input_height=input_height, input_width=input_width, out_dim=out_dim
        )

        for i, (visibility, x, y) in enumerate(predictions[: len(frames)]):
            frame_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            x_orig = x * frame_width / input_width if visibility else -1
            y_orig = y * frame_height / input_height if visibility else -1

            if visibility:
                point = (int(x_orig), int(y_orig))
                track_points.append(point)
                radius, contour = estimate_ball_radius(
                    prev_gray, frame_gray, point[0], point[1], size_state
                )
            else:
                if track_points:
                    track_points.popleft()
                radius, contour = 0, None

            result = {
                "Frame": frame_index + i,
                "Visibility": visibility,
                "X": int(x_orig),
                "Y": int(y_orig),
                "Radius": radius,
            }
            append_to_csv(result, csv_path)

            if args.visualize or out_writer is not None:
                vis_frame = render_prediction(
                    frames[i],
                    track_points,
                    visibility,
                    int(x_orig),
                    int(y_orig),
                    radius,
                    contour,
                )
                if args.visualize:
                    cv2.namedWindow("Ball Tracking", cv2.WINDOW_NORMAL)
                    cv2.imshow("Ball Tracking", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        exit_flag = True
                        break
                if out_writer is not None:
                    out_writer.write(vis_frame)
            prev_gray = frame_gray
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
