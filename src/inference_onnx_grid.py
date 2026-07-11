import argparse
import csv
import os

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

WIDTH = 768
HEIGHT = 432
IMGS_PER_INSTANCE = 5
GRID_COLS = 48
GRID_ROWS = 27
GRID_SIZE_COL = WIDTH / GRID_COLS
GRID_SIZE_ROW = HEIGHT / GRID_ROWS


def is_headless():
    return not any(os.environ.get(name) for name in ("DISPLAY", "WAYLAND_DISPLAY"))


def build_session(onnx_path, provider):
    available = ort.get_available_providers()
    if provider == "gpu":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is not available in onnxruntime")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    return session, session.get_inputs()[0].name, session.get_providers()


def get_predictions(session, input_name, frames, is_bgr_format=False):
    output_height = frames[0].shape[0]
    output_width = frames[0].shape[1]

    batches = []
    for i in range(0, len(frames), IMGS_PER_INSTANCE):
        batch = frames[i:i + IMGS_PER_INSTANCE]
        if len(batch) == IMGS_PER_INSTANCE:
            batches.append(batch)

    units = []
    for batch in batches:
        unit = []
        for frame in batch:
            if is_bgr_format:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.moveaxis(frame, -1, 0)
            unit.append(frame[0])
            unit.append(frame[1])
            unit.append(frame[2])
        units.append(unit)

    units = np.asarray(units, dtype=np.float32)
    units /= 255.0

    y_pred = session.run(None, {input_name: units})[0]

    y_pred = np.split(y_pred, IMGS_PER_INSTANCE, axis=1)
    y_pred = np.stack(y_pred, axis=2)
    y_pred = np.moveaxis(y_pred, 1, -1)

    conf_grid, x_offset_grid, y_offset_grid = np.split(y_pred, 3, axis=-1)
    conf_grid = np.squeeze(conf_grid, axis=-1)
    x_offset_grid = np.squeeze(x_offset_grid, axis=-1)
    y_offset_grid = np.squeeze(y_offset_grid, axis=-1)

    ball_coordinates = []
    for i in range(conf_grid.shape[0]):
        for j in range(conf_grid.shape[1]):
            curr_conf_grid = conf_grid[i][j]
            curr_x_offset_grid = x_offset_grid[i][j]
            curr_y_offset_grid = y_offset_grid[i][j]

            max_conf_val = np.max(curr_conf_grid)
            pred_row, pred_col = np.unravel_index(np.argmax(curr_conf_grid), curr_conf_grid.shape)
            pred_has_ball = max_conf_val >= 0.5

            x_offset = curr_x_offset_grid[pred_row][pred_col]
            y_offset = curr_y_offset_grid[pred_row][pred_col]

            x_pred = int((x_offset + pred_col) * GRID_SIZE_COL)
            y_pred = int((y_offset + pred_row) * GRID_SIZE_ROW)

            if pred_has_ball:
                ball_coordinates.append((int((x_pred / WIDTH) * output_width), int((y_pred / HEIGHT) * output_height)))
            else:
                ball_coordinates.append((0, 0))

    return ball_coordinates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX inference for GridTrackNet")
    parser.add_argument("--video_path", required=True, type=str, help="Path to input .mp4 video file.")
    parser.add_argument("--model_path", required=False, default=os.path.join(os.getcwd(), "model_weights.onnx"), type=str, help="Path to ONNX model file.")
    parser.add_argument("--display_trail", required=False, default=1, type=int, help="Output a visible trail of the ball's trajectory. Default = 1")
    parser.add_argument("--output_dir", required=False, default=None, type=str, help="Directory to save output video and CSV.")
    parser.add_argument("--only_csv", action="store_true", default=False, help="Save only CSV, skip video output.")
    parser.add_argument("--chunk_size", required=False, default=5, type=int, help="Number of frames buffered before inference. Use multiples of 5. Default = 5")
    parser.add_argument("--provider", choices=("cpu", "gpu"), default="gpu", help="Execution provider for ONNX Runtime. Default = gpu")

    args = parser.parse_args()

    video_dir = args.video_path
    onnx_model = args.model_path
    display_trail = bool(args.display_trail)
    output_dir = args.output_dir
    only_csv = args.only_csv
    chunk_size = max(IMGS_PER_INSTANCE, args.chunk_size)

    if chunk_size % IMGS_PER_INSTANCE != 0:
        raise ValueError(f"--chunk_size must be a multiple of {IMGS_PER_INSTANCE}")

    if not os.path.exists(onnx_model):
        raise FileNotFoundError(f"ONNX model not found: {onnx_model}")

    session, input_name, providers = build_session(onnx_model, args.provider)
    print(f"ONNX providers: {providers}")

    cap = cv2.VideoCapture(video_dir)
    if not cap.isOpened():
        raise RuntimeError("Error opening video file")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps >= 57 and fps <= 62:
        num_frames_skip = 2
    elif fps >= 22 and fps <= 32:
        num_frames_skip = 1
    else:
        raise RuntimeError("ERROR: Video is not 30FPS or 60FPS")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // num_frames_skip
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    directory, filename = os.path.split(video_dir)
    name, extension = os.path.splitext(filename)
    if output_dir:
        output_base_dir = os.path.join(output_dir, name)
        os.makedirs(output_base_dir, exist_ok=True)
    else:
        output_base_dir = directory

    csv_path = os.path.join(output_base_dir, "ball.csv")
    csv_file = open(csv_path, "w", newline="", buffering=1)
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame", "Visibility", "X", "Y"])
    csv_file.flush()

    video_writer = None
    if not only_csv:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if output_dir:
            output_path = os.path.join(output_base_dir, "predict_onnx.mp4")
        else:
            output_path = os.path.join(output_base_dir, name + " Predicted ONNX" + extension)
        output_fps = fps if num_frames_skip == 1 else 30
        video_writer = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_width, frame_height))

    frames = []
    index = 0
    ball_coordinates_history = []
    predicted_frame_index = 0
    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")

    def write_predicted_frames(predicted_frames, ball_coordinates):
        nonlocal_predicted_frame_index = predicted_frame_index
        history_start_idx = len(ball_coordinates_history) - len(ball_coordinates)
        for i, frame in enumerate(predicted_frames):
            if i < len(ball_coordinates):
                x, y = ball_coordinates[i]
                visibility = 0 if (x == 0 and y == 0) else 1
                csv_writer.writerow([nonlocal_predicted_frame_index, visibility, x, y])
                nonlocal_predicted_frame_index += 1

                if display_trail:
                    current_history_idx = history_start_idx + i
                    for trail_offset in range(7, 0, -2):
                        idx = current_history_idx - trail_offset
                        if idx >= 0:
                            cv2.circle(frame, ball_coordinates_history[idx], 4, (0, 255, 255), -1)
                else:
                    cv2.circle(frame, ball_coordinates[i], 8, (0, 0, 255), 4)

            if video_writer is not None:
                video_writer.write(frame)

        csv_file.flush()
        return nonlocal_predicted_frame_index

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if index % num_frames_skip == 0:
                frames.append(frame)
                pbar.update(1)
                if len(frames) == chunk_size:
                    ball_coordinates = get_predictions(session, input_name, frames, True)
                    for ball in ball_coordinates:
                        ball_coordinates_history.append(ball)
                    predicted_frame_index = write_predicted_frames(frames, ball_coordinates)
                    frames = []

            index += 1

        if len(frames) >= IMGS_PER_INSTANCE:
            processable_frames = frames[:len(frames) - (len(frames) % IMGS_PER_INSTANCE)]
            if processable_frames:
                ball_coordinates = get_predictions(session, input_name, processable_frames, True)
                for ball in ball_coordinates:
                    ball_coordinates_history.append(ball)
                predicted_frame_index = write_predicted_frames(processable_frames, ball_coordinates)
    finally:
        pbar.close()
        cap.release()
        csv_file.close()
        if video_writer is not None:
            video_writer.release()
        if not is_headless():
            cv2.destroyAllWindows()
