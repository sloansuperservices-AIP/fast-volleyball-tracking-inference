#!/usr/bin/env python3
import argparse
import json
import logging
import os
from collections import deque
from typing import List

import numpy as np
import pandas as pd

from ball_tracker import BallTracker, Track
from track_utils import find_cyclic_sequences, find_rolling_sequences
from constants import DEFAULT_FPS, DEFAULT_MAX_DISTANCE, DEFAULT_MIN_DURATION_SEC

LOG = logging.getLogger(__name__)


class TrackCalculator:
    def __init__(
        self,
        csv_path: str,
        output_dir: str = "output",
        fps: float = DEFAULT_FPS,
        max_distance: float = DEFAULT_MAX_DISTANCE,
        min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    ):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.fps = fps
        self.max_distance = max_distance
        self.min_duration_sec = min_duration_sec
        self.tracks: List[Track] = []
        self.track_distances: dict[int, str] = {}

    def _validate_csv(self) -> None:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

    def _load_and_process_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
        df["Visibility"] = pd.to_numeric(df["Visibility"], errors="coerce")
        df["X"] = pd.to_numeric(df["X"], errors="coerce")
        df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
        if "Radius" not in df.columns:
            df["Radius"] = 0
        df["Radius"] = pd.to_numeric(df["Radius"], errors="coerce").fillna(0)
        df.loc[(df["X"] == -1) | (df["Visibility"] == 0), ["X", "Y"]] = np.nan
        return df

    def _is_overlapping(self, track1: Track, track2: Track) -> bool:
        return (
            track1.start_frame <= track2.last_frame
            and track2.start_frame <= track1.last_frame
        )

    def _trim_bounce_start(self, track: Track) -> Track:
        if not track.positions:
            return track

        sequences = find_cyclic_sequences(track.positions)
        if sequences:
            for start, end in sequences:
                track.start_frame = end
                break

        sequences = find_rolling_sequences(track.positions)
        if sequences:
            for start, end in sequences:
                track.last_frame = start
                break

        track.positions = deque(
            [
                pos
                for pos in track.positions
                if track.start_frame <= pos[1] <= track.last_frame
            ],
            maxlen=3000
        )
        return track

    def _filter_short_tracks(self, episodes: List[Track]) -> List[Track]:
        episodes = [self._trim_bounce_start(ep) for ep in episodes]
        long_tracks = [ep for ep in episodes if ep.duration_sec() >= self.min_duration_sec]
        sorted_tracks = sorted(
            long_tracks, key=lambda x: x.duration_sec(), reverse=True
        )

        filtered = []
        used = set()
        for i, track1 in enumerate(sorted_tracks):
            if i in used:
                continue
            filtered.append(track1)
            used.add(i)
            for j, track2 in enumerate(sorted_tracks):
                if j <= i or j in used:
                    continue
                if self._is_overlapping(track1, track2):
                    used.add(j)

        frames_to_extend = int(self.fps)
        extended = []
        for ep in filtered:
            ep.start_frame = max(0, ep.start_frame - frames_to_extend)
            ep.last_frame = ep.last_frame + frames_to_extend
            extended.append(ep)

        merged = []
        used = set()
        sorted_ext = sorted(extended, key=lambda x: x.start_frame)
        for i, track1 in enumerate(sorted_ext):
            if i in used:
                continue
            merged_track = track1
            merged_positions = list(merged_track.positions)
            used.add(i)
            for j, track2 in enumerate(sorted_ext):
                if j <= i or j in used:
                    continue
                if self._is_overlapping(merged_track, track2):
                    merged_track.start_frame = min(
                        merged_track.start_frame, track2.start_frame
                    )
                    merged_track.last_frame = max(
                        merged_track.last_frame, track2.last_frame
                    )
                    merged_positions.extend(track2.positions)
                    used.add(j)
            merged_track.positions = deque(sorted(merged_positions, key=lambda x: x[1]), maxlen=3000)
            merged.append(merged_track)

        merged = [self._trim_bounce_start(ep) for ep in merged]
        return sorted(merged, key=lambda x: x.start_frame)

    def _process_detections(self, df: pd.DataFrame) -> None:
        tracker = BallTracker(
            buffer_size=2500,
            max_disappeared=40,
            max_distance=self.max_distance,
            fps=self.fps,
        )
        close_tracks = []

        # Optimization: group by frame and iterate
        frames_group = df.groupby('Frame')
        for frame_num, group in sorted(frames_group):
            detections = []
            for _, row in group.iterrows():
                if not np.isnan(row["X"]) and not np.isnan(row["Y"]):
                    detections.append(
                        {
                            "x1": row["X"] - 20,
                            "y1": row["Y"] - 20,
                            "x2": row["X"] + 20,
                            "y2": row["Y"] + 20,
                            "confidence": row["Visibility"],
                            "radius": row["Radius"],
                            "cls_id": 0,
                        }
                    )
            _, _, close_track = tracker.update(detections, int(frame_num))
            close_tracks.extend(close_track)

        episodes = []
        for track in close_tracks:
            if not track.positions:
                continue
            episodes.append(track)

        for _, track in tracker.tracks.items():
            if not track.positions:
                continue
            episodes.append(track)

        self.tracks = self._filter_short_tracks(episodes)

    def _save_tracks_to_json(self) -> None:
        csv_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        video_basename = os.path.basename(os.path.dirname(self.csv_path)) or csv_name.replace("_predict_ball", "")
        tracks_dir = os.path.join(self.output_dir, video_basename, "tracks")
        os.makedirs(tracks_dir, exist_ok=True)

        for track in self.tracks:
            track.calculate_stats()
            file_path = os.path.join(tracks_dir, f"track_{track.track_id:04d}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(track.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.tracks)} tracks to: {tracks_dir}")

    def run(self) -> None:
        self._validate_csv()
        df = self._load_and_process_csv()
        self._process_detections(df)
        self._save_tracks_to_json()


def main():
    parser = argparse.ArgumentParser(description="Calculate tracks from CSV to JSON")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to ball.csv")
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Root output directory for JSON"
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frames per second")
    parser.add_argument(
        "--max_distance", type=float, default=DEFAULT_MAX_DISTANCE, help="Max tracking distance"
    )
    parser.add_argument(
        "--min_duration_sec", type=float, default=DEFAULT_MIN_DURATION_SEC, help="Minimum track duration"
    )
    args = parser.parse_args()

    calculator = TrackCalculator(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        fps=args.fps,
        max_distance=args.max_distance,
        min_duration_sec=args.min_duration_sec,
    )
    calculator.run()


if __name__ == "__main__":
    main()
