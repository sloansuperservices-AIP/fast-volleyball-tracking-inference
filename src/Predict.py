import cv2
import argparse
import csv
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from src.model.GridTrackNet import GridTrackNet

WIDTH = 768
HEIGHT = 432
IMGS_PER_INSTANCE = 5
GRID_COLS = 48
GRID_ROWS = 27
GRID_SIZE_COL = WIDTH/GRID_COLS
GRID_SIZE_ROW = HEIGHT/GRID_ROWS

MODEL_DIR = os.path.join(os.getcwd(),"model_weights.h5")
model = None
loaded_model_dir = None


def is_headless():
    return not any(os.environ.get(name) for name in ("DISPLAY", "WAYLAND_DISPLAY"))


def configure_tensorflow():
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def getModel(model_dir=MODEL_DIR):
    global model
    global loaded_model_dir

    if model is None:
        model = GridTrackNet(IMGS_PER_INSTANCE, HEIGHT, WIDTH)

    if loaded_model_dir != model_dir:
        model.load_weights(model_dir)
        loaded_model_dir = model_dir

    return model

def getPredictions(frames, predict_batch_size, isBGRFormat = False):
    outputHeight = frames[0].shape[0]
    outputWidth = frames[0].shape[1]

    batches = []
    for i in range(0, len(frames), 5):
        batch = frames[i:i+5]
        if len(batch) == 5:
            batches.append(batch)

    units = []
    for batch in batches:
        unit = []
        for frame in batch:
            if(isBGRFormat):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame,(WIDTH,HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.moveaxis(frame, -1, 0)
            unit.append(frame[0])
            unit.append(frame[1])
            unit.append(frame[2])
        units.append(unit)

    units = np.asarray(units)
    units = units.astype(np.float32)
    units /= 255

    batch_size = max(1, min(predict_batch_size, len(batches)))
    y_pred = getModel().predict(units, batch_size=batch_size, verbose=0)

    y_pred = np.split(y_pred, IMGS_PER_INSTANCE, axis=1)
    y_pred = np.stack(y_pred, axis=2)
    y_pred = np.moveaxis(y_pred, 1, -1)

    confGrid, xOffsetGrid, yOffsetGrid = np.split(y_pred, 3, axis=-1)

    confGrid = np.squeeze(confGrid, axis=-1)
    xOffsetGrid = np.squeeze(xOffsetGrid, axis=-1)
    yOffsetGrid = np.squeeze(yOffsetGrid, axis=-1)

    ballCoordinates = []
    for i in range(0, confGrid.shape[0]):
        for j in range(0, confGrid.shape[1]):
            currConfGrid = confGrid[i][j]
            currXOffsetGrid = xOffsetGrid[i][j]
            currYOffsetGrid = yOffsetGrid[i][j]

            maxConfVal = np.max(currConfGrid)
            predRow, predCol = np.unravel_index(np.argmax(currConfGrid), currConfGrid.shape)

            threshold = 0.5
            predHasBall = maxConfVal >= threshold

            xOffset = currXOffsetGrid[predRow][predCol]
            yOffset = currYOffsetGrid[predRow][predCol]

            xPred = int((xOffset + predCol) * GRID_SIZE_COL)
            yPred = int((yOffset + predRow) * GRID_SIZE_ROW)

            if(predHasBall):
                ballCoordinates.append((int((xPred/WIDTH)*outputWidth), int((yPred/HEIGHT)*outputHeight)))
            else:
                ballCoordinates.append((0,0))

    return ballCoordinates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument Parser for GridTrackNet')

    parser.add_argument('--video_path', required=True, type=str, help="Path to input .mp4 video file.")
    parser.add_argument('--model_path', required=False, default=os.path.join(os.getcwd(), "model_weights.h5"), type=str, help="Path to TensorFlow weights file.")
    parser.add_argument('--display_trail', required=False, default=1, type=int, help="Output a visible trail of the ball's trajectory. Default = 1")
    parser.add_argument('--output_dir', required=False, default=None, type=str, help="Directory to save output video and CSV.")
    parser.add_argument('--only_csv', action='store_true', default=False, help="Save only CSV, skip video output.")
    parser.add_argument('--chunk_size', required=False, default=5, type=int, help="Number of frames buffered before inference. Use multiples of 5. Default = 5")
    parser.add_argument('--predict_batch_size', required=False, default=1, type=int, help="TensorFlow predict batch size. Lower values reduce VRAM usage. Default = 1")

    args = parser.parse_args()

    VIDEO_DIR = args.video_path
    MODEL_DIR = args.model_path
    DISPLAY_TRAIL = bool(args.display_trail)
    OUTPUT_DIR = args.output_dir
    ONLY_CSV = args.only_csv
    CHUNK_SIZE = max(IMGS_PER_INSTANCE, args.chunk_size)
    PREDICT_BATCH_SIZE = max(1, args.predict_batch_size)

    if CHUNK_SIZE % IMGS_PER_INSTANCE != 0:
        raise ValueError(f"--chunk_size must be a multiple of {IMGS_PER_INSTANCE}")

    configure_tensorflow()
    getModel(MODEL_DIR)

    cap = cv2.VideoCapture(VIDEO_DIR)

    if not cap.isOpened():
        print("Error opening video file")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if(fps >= 47 and fps <= 62):
        numFramesSkip = 2
    elif (fps >= 22 and fps <= 32):
        numFramesSkip = 1
    else:
        print("ERROR: Video is not 30FPS or 60FPS")
        exit(0)

    totalFrames = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // numFramesSkip)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    directory, filename = os.path.split(VIDEO_DIR)
    name, extension = os.path.splitext(filename)
    if OUTPUT_DIR:
        output_base_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(output_base_dir, exist_ok=True)
    else:
        output_base_dir = directory

    csv_path = os.path.join(output_base_dir, "ball.csv")
    csv_file = open(csv_path, "w", newline="", buffering=1)
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame", "Visibility", "X", "Y"])
    csv_file.flush()

    video_writer = None
    if not ONLY_CSV:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if OUTPUT_DIR:
            output_path = os.path.join(output_base_dir, "predict.mp4")
        else:
            output_filename = name + " Predicted" + extension
            output_path = os.path.join(output_base_dir, output_filename)

        if(numFramesSkip == 1):
            outputFPS = fps
        else:
            outputFPS = 30
        video_writer = cv2.VideoWriter(output_path, fourcc, outputFPS, (frame_width, frame_height))

    index = 0

    frames = []
    ballCoordinatesHistory = []
    predictedFrameIndex = 0
    pbar = tqdm(total=totalFrames, desc="Processing video", unit="frame")

    def writePredictedFrames(predicted_frames, ballCoordinates):
        nonlocal_predictedFrameIndex = predictedFrameIndex
        history_start_idx = len(ballCoordinatesHistory) - len(ballCoordinates)
        for i, frame in enumerate(predicted_frames):
            if(i < len(ballCoordinates)):
                x, y = ballCoordinates[i]
                visibility = 0 if (x == 0 and y == 0) else 1
                csv_writer.writerow([nonlocal_predictedFrameIndex, visibility, x, y])
                nonlocal_predictedFrameIndex += 1

                if(DISPLAY_TRAIL):
                    current_history_idx = history_start_idx + i
                    for trail_offset in range(7, 0, -2):
                        idx = current_history_idx - trail_offset
                        if idx >= 0:
                            cv2.circle(frame, ballCoordinatesHistory[idx], 4, (0, 255, 255),-1)
                else:
                    cv2.circle(frame, ballCoordinates[i], 8, (0, 0, 255),4)

            if video_writer is not None:
                video_writer.write(frame)

        csv_file.flush()
        return nonlocal_predictedFrameIndex

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            if(index % numFramesSkip == 0):
                frames.append(frame)
                pbar.update(1)
                if(len(frames) == CHUNK_SIZE):
                    ballCoordinates = getPredictions(frames, PREDICT_BATCH_SIZE, True)
                    for ball in ballCoordinates:
                        ballCoordinatesHistory.append(ball)
                    predictedFrameIndex = writePredictedFrames(frames, ballCoordinates)
                    frames = []

            index += 1

        if len(frames) >= 5:
            processable_frames = frames[:len(frames) - (len(frames) % 5)]
            if processable_frames:
                ballCoordinates = getPredictions(processable_frames, PREDICT_BATCH_SIZE, True)
                for ball in ballCoordinates:
                    ballCoordinatesHistory.append(ball)
                predictedFrameIndex = writePredictedFrames(processable_frames, ballCoordinates)
    finally:
        pbar.close()
        cap.release()
        csv_file.close()
        if video_writer is not None:
            video_writer.release()
        if not is_headless():
            cv2.destroyAllWindows()
