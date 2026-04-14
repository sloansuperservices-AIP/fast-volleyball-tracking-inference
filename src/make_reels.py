import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.signal import savgol_filter
from tqdm import tqdm

try:
    from constants import (
        DEFAULT_CROP_ASPECT_RATIO,
        DEFAULT_FPS,
        DEFAULT_IMAGE_HEIGHT,
        DEFAULT_IMAGE_WIDTH,
        DEFAULT_SMOOTH_WINDOW,
    )
except ImportError:
    from src.constants import (
        DEFAULT_CROP_ASPECT_RATIO,
        DEFAULT_FPS,
        DEFAULT_IMAGE_HEIGHT,
        DEFAULT_IMAGE_WIDTH,
        DEFAULT_SMOOTH_WINDOW,
    )

LOG = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def load_single_track(track_json_path: str) -> Dict[str, Any]:
    with open(track_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    positions = []
    # Support both formats: [[x, y], frame] and {"ball_position": [x, y], "frame_num": frame}
    for item in data.get("positions", []):
        if isinstance(item, list) and len(item) == 2:
            positions.append((float(item[0][0]), float(item[0][1]), int(item[1])))
        elif isinstance(item, dict):
            pos = item.get("ball_position", [0, 0])
            positions.append((float(pos[0]), float(pos[1]), int(item.get("frame_num", 0))))

    return {
        "start_frame": data["start_frame"],
        "last_frame": data["last_frame"],
        "positions": positions,
        "track_id": data.get("track_id", 0),
    }


def apply_kalman_filter(x_values: np.ndarray) -> np.ndarray:
    if len(x_values) < 2:
        return x_values
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[x_values[0]], [0.0]])
    kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
    kf.H = np.array([[1.0, 0.0]])
    kf.P *= 1000.0
    kf.R = 5
    kf.Q = np.array([[0.1, 0.1], [0.1, 0.1]])

    smoothed = []
    for z in x_values:
        kf.predict()
        kf.update(z)
        smoothed.append(kf.x[0, 0])
    return np.array(smoothed)


def smooth_trajectory(x_values: np.ndarray, method: str, window: int) -> np.ndarray:
    if len(x_values) < window:
        return x_values
    if method == "moving_avg":
        return np.convolve(x_values, np.ones(window) / window, mode="same")
    if method == "savitzky_golay":
        return savgol_filter(x_values, window, 3)
    if method == "kalman":
        return apply_kalman_filter(x_values)
    return x_values


def crop_and_save_track(
    video_path: str,
    track: Dict[str, Any],
    output_path: str,
    method: str = "savitzky_golay",
    window: int = DEFAULT_SMOOTH_WINDOW,
    visualize: bool = False,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_width = int(frame_height * DEFAULT_CROP_ASPECT_RATIO)
    crop_height = frame_height

    out = None
    if not visualize:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))

    frame_to_pos = {int(p[2]): p[0] for p in track["positions"]}
    start_f, end_f = track["start_frame"], track["last_frame"]

    x_centers = []
    last_x = track["positions"][0][0] if track["positions"] else frame_width / 2
    for f in range(start_f, end_f + 1):
        if f in frame_to_pos:
            last_x = frame_to_pos[f]
        x_centers.append(last_x)

    x_smooth = smooth_trajectory(np.array(x_centers), method, window)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    for i, frame_idx in enumerate(range(start_f, end_f + 1)):
        ret, frame = cap.read()
        if not ret:
            break

        center_x = int(x_smooth[i])
        left = np.clip(center_x - crop_width // 2, 0, frame_width - crop_width)
        cropped = frame[:, left : left + crop_width]

        if cropped.shape[1] != crop_width:
            cropped = cv2.resize(cropped, (crop_width, crop_height))

        if out:
            out.write(cropped)

        if visualize:
            cv2.imshow("Reel Preview", cropped)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if out:
        out.release()


def main():
    parser = argparse.ArgumentParser(description="Create vertical reels from tracking data")
    parser.add_argument("--video_path", required=True, help="Input video")
    parser.add_argument("--json_dir", help="Directory with track JSONs")
    parser.add_argument("--track_json", help="Single track JSON")
    parser.add_argument("--output_dir", default="output", help="Output root")
    parser.add_argument("--smoothing", choices=["none", "moving_avg", "savitzky_golay", "kalman"], default="savitzky_golay")
    parser.add_argument("--window", type=int, default=DEFAULT_SMOOTH_WINDOW)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    video_name = Path(args.video_path).stem
    json_paths = []
    if args.track_json:
        json_paths = [args.track_json]
    elif args.json_dir:
        json_paths = sorted(Path(args.json_dir).glob("track_*.json"))

    reels_dir = os.path.join(args.output_dir, video_name, "reels")
    os.makedirs(reels_dir, exist_ok=True)

    for jp in tqdm(json_paths, desc="Creating reels"):
        track = load_single_track(str(jp))
        output_path = os.path.join(reels_dir, f"reel_{video_name}_{track['track_id']:04d}.mp4")
        crop_and_save_track(
            args.video_path, track, output_path, method=args.smoothing, window=args.window, visualize=args.visualize
        )


if __name__ == "__main__":
    main()
