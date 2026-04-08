#!/usr/bin/env python3
import argparse
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from tqdm import tqdm

try:
    from constants import DEFAULT_CROP_ASPECT_RATIO, DEFAULT_FPS, DEFAULT_SMOOTH_WINDOW
except ImportError:
    from src.constants import DEFAULT_CROP_ASPECT_RATIO, DEFAULT_FPS, DEFAULT_SMOOTH_WINDOW

LOG = logging.getLogger(__name__)

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

def crop_frame(frame: np.ndarray, center_x: int, crop_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    left = max(0, min(center_x - crop_width // 2, w - crop_width))
    return frame[:, left : left + crop_width]

def crop_and_save_track(video_path: str, track: Dict, output_path: str, visualize: bool = False) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    crop_w = int(h * DEFAULT_CROP_ASPECT_RATIO)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (crop_w, h))

    frame_to_pos = {int(p[1]): p[0] for p in track["positions"]}
    cap.set(cv2.CAP_PROP_POS_FRAMES, track["start_frame"])

    for f_idx in range(track["start_frame"], track["last_frame"] + 1):
        ret, frame = cap.read()
        if not ret: break
        center_x = int(frame_to_pos.get(f_idx, [w/2, h/2])[0])
        cropped = crop_frame(frame, center_x, crop_w)
        if visualize:
            cv2.imshow("Crop", cropped)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
        out.write(cropped)
    cap.release()
    out.release()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--json_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    
    base = os.path.splitext(os.path.basename(args.video_path))[0]
    json_dir = args.json_dir or os.path.join(args.output_dir, base, "tracks")
    reels_dir = os.path.join(args.output_dir, base, "reels")
    os.makedirs(reels_dir, exist_ok=True)

    for f in sorted(os.listdir(json_dir)):
        if f.endswith(".json"):
            with open(os.path.join(json_dir, f)) as jf:
                track = json.load(jf)
            crop_and_save_track(args.video_path, track, os.path.join(reels_dir, f"reel_{f.replace('.json', '.mp4')}"), args.visualize)

if __name__ == "__main__":
    main()
