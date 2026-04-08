#!/usr/bin/env python3
import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple
import cv2
from tqdm import tqdm
from ball_tracker import Track
from constants import DEFAULT_FADE_DURATION

LOG = logging.getLogger(__name__)

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

def resolve_video_basename(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]

class BaseExporter:
    def open_track(self, track_id: int) -> bool: return True
    def write(self, frame) -> None: raise NotImplementedError
    def close_track(self) -> None: pass
    def close(self) -> None: pass

class CombinedVideoExporter(BaseExporter):
    def __init__(self, output_path: str, fps: float, size: Tuple[int, int]) -> None:
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    def write(self, frame) -> None: self._writer.write(frame)
    def close(self) -> None: self._writer.release()

class SplitClipsExporter(BaseExporter):
    def __init__(self, split_dir: str, fps: float, size: Tuple[int, int]) -> None:
        self.split_dir = split_dir
        self._fps = fps
        self._size = size
        self._writer = None
        os.makedirs(split_dir, exist_ok=True)
    def open_track(self, track_id: int) -> bool:
        self.close_track()
        path = os.path.join(self.split_dir, f"track_{track_id:04d}.mp4")
        self._writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self._fps, self._size)
        return self._writer.isOpened()
    def write(self, frame) -> None: self._writer.write(frame)
    def close_track(self) -> None:
        if self._writer: self._writer.release()
    def close(self) -> None: self.close_track()

class TrackProcessor:
    def __init__(self, json_dir: str, video_path: str, output_path: Optional[str] = None, split_dir: Optional[str] = None, fps: float = 30.0) -> None:
        self.json_dir, self.video_path, self.output_path, self.split_dir, self.fps = json_dir, video_path, output_path, split_dir, fps
        self.tracks: List[Track] = []

    def _load_tracks_from_json(self) -> None:
        json_files = sorted([f for f in os.listdir(self.json_dir) if f.startswith("track_") and f.endswith(".json")])
        for filename in json_files:
            with open(os.path.join(self.json_dir, filename), "r", encoding="utf-8") as f:
                self.tracks.append(Track.from_dict(json.load(f)))

    def visualize_tracks(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        exporter = SplitClipsExporter(self.split_dir, fps, (width, height)) if self.split_dir else CombinedVideoExporter(self.output_path, fps, (width, height)) if self.output_path else None

        for track in self.tracks:
            if exporter and not exporter.open_track(track.track_id): continue
            pos_by_frame = {int(pos[1]): (int(pos[0][0]), int(pos[0][1])) for pos in track.positions}
            cap.set(cv2.CAP_PROP_POS_FRAMES, track.start_frame)
            for frame_num in range(track.start_frame, track.last_frame + 1):
                ret, frame = cap.read()
                if not ret: break
                clean_frame = frame.copy()
                if frame_num in pos_by_frame:
                    cv2.circle(frame, pos_by_frame[frame_num], 10, (0, 255, 255), -1)
                if exporter: exporter.write(clean_frame)
                else:
                    cv2.imshow("Track", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"): return
            if exporter: exporter.close_track()
        cap.release()
        if exporter: exporter.close()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--split_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.verbose)
    base_name = resolve_video_basename(args.video_path)
    if args.json_dir is None and args.output_dir: args.json_dir = os.path.join(args.output_dir, base_name, "tracks")
    if args.output_path is None and args.output_dir and not args.split_dir: args.output_path = os.path.join(args.output_dir, base_name, "combined.mp4")
    processor = TrackProcessor(args.json_dir, args.video_path, args.output_path, args.split_dir, args.fps)
    processor._load_tracks_from_json()
    processor.visualize_tracks()

if __name__ == "__main__":
    main()
