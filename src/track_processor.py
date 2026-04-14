#!/usr/bin/env python3
import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import cv2
from tqdm import tqdm

from ball_tracker import Track

try:
    from constants import DEFAULT_FADE_DURATION, DEFAULT_FPS
except ImportError:
    from src.constants import DEFAULT_FADE_DURATION, DEFAULT_FPS

LOG = logging.getLogger(__name__)


class TrackProcessor:
    def __init__(
        self,
        json_dir: str,
        video_path: str,
        output_path: Optional[str] = None,
        split_dir: Optional[str] = None,
        fps: float = DEFAULT_FPS,
    ):
        self.json_dir = json_dir
        self.video_path = video_path
        self.output_path = output_path
        self.split_dir = split_dir
        self.fps = fps
        self.tracks: List[Track] = []

    def _load_tracks(self) -> None:
        if not os.path.exists(self.json_dir):
            raise FileNotFoundError(f"JSON directory not found: {self.json_dir}")
        json_files = sorted(Path(self.json_dir).glob("track_*.json"))
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                self.tracks.append(Track.from_dict(json.load(f)))
        LOG.info("Loaded %s tracks", len(self.tracks))

    def process(self) -> None:
        self._load_tracks()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps

        combined_writer = None
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            combined_writer = cv2.VideoWriter(
                self.output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )

        fade_frames = int(fps * DEFAULT_FADE_DURATION)

        for track in tqdm(self.tracks, desc="Processing tracks"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, track.start_frame)
            last_frame = None
            for f_idx in range(track.start_frame, track.last_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw trajectory
                pos_map = {p[1]: p[0] for p in track.positions}
                for pf in range(f_idx - 15, f_idx + 1):
                    if pf in pos_map:
                        px, py = int(pos_map[pf][0]), int(pos_map[pf][1])
                        alpha = max(0.2, 1.0 - (f_idx - pf) / 15.0)
                        cv2.circle(frame, (px, py), 6, (0, 255, 255), -1)
                        if pf == f_idx:
                            cv2.circle(frame, (px, py), 8, (0, 0, 255), 2)

                if combined_writer:
                    combined_writer.write(frame)
                last_frame = frame

            if combined_writer and last_frame is not None:
                for i in range(fade_frames):
                    alpha = 1.0 - (i / fade_frames)
                    combined_writer.write(cv2.convertScaleAbs(last_frame, alpha=alpha))

        cap.release()
        if combined_writer:
            combined_writer.release()
            LOG.info("Combined video saved: %s", self.output_path)


def main():
    parser = argparse.ArgumentParser(description="Process and visualize tracks")
    parser.add_argument("--video_path", required=True, help="Input video")
    parser.add_argument("--json_dir", help="Directory with JSON tracks")
    parser.add_argument("--output_dir", help="Output root")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    video_name = Path(args.video_path).stem
    json_dir = args.json_dir or os.path.join(args.output_dir, video_name, "tracks")
    output_path = os.path.join(args.output_dir, video_name, "combined.mp4")

    processor = TrackProcessor(json_dir, args.video_path, output_path=output_path)
    processor.process()


if __name__ == "__main__":
    main()
