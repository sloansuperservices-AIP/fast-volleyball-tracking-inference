#!/usr/bin/env python3
import argparse
import json
import logging
import os
from typing import List

import numpy as np
import pandas as pd

from ball_tracker import BallTracker, Track
from constants import (
    DEFAULT_BOUNCE_FRAMES,
    DEFAULT_DETECTION_BOX_RADIUS,
    DEFAULT_FPS,
    DEFAULT_MAX_DISTANCE,
    DEFAULT_MIN_DURATION_SEC,
    DEFAULT_MAX_X_DISPLACEMENT,
    DEFAULT_MIN_Y_DISPLACEMENT,
)
from track_utils import find_cyclic_sequences, find_rolling_sequences

LOG = logging.getLogger(__name__)


class TrackCalculator:
    def __init__(
        self,
        csv_path: str,
        output_dir: str = "output",
        fps: float = DEFAULT_FPS,
        max_distance: float = DEFAULT_MAX_DISTANCE,
        min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
        max_x_displacement: float = DEFAULT_MAX_X_DISPLACEMENT,
        min_y_displacement: float = DEFAULT_MIN_Y_DISPLACEMENT,
        bounce_frames: int = DEFAULT_BOUNCE_FRAMES,
    ):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.fps = fps
        self.max_distance = max_distance
        self.min_duration_sec = min_duration_sec
        self.max_x_displacement = max_x_displacement
        self.min_y_displacement = min_y_displacement
        self.bounce_frames = bounce_frames
        self.tracks: List[Track] = []
        self.track_distances: dict[int, str] = {}

    def _validate_csv(self) -> None:
        """Check if the CSV file exists."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

    def _load_and_process_csv(self) -> pd.DataFrame:
        """Load CSV and preprocess columns."""
        df = pd.read_csv(self.csv_path)
        df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
        df["Visibility"] = pd.to_numeric(df["Visibility"], errors="coerce")
        df["X"] = pd.to_numeric(df["X"], errors="coerce")
        df["Y"] = pd.to_numeric(df["Y"], errors="coerce")

        if "Radius" not in df.columns:
            df["Radius"] = DEFAULT_DETECTION_BOX_RADIUS

        df["Radius"] = pd.to_numeric(df["Radius"], errors="coerce").fillna(
            DEFAULT_DETECTION_BOX_RADIUS
        )

        df.loc[(df["X"] == -1) | (df["Visibility"] == 0), ["X", "Y"]] = np.nan
        return df

    def _is_overlapping(self, track1: Track, track2: Track) -> bool:
        """Check if two tracks overlap in time."""
        return (
            track1.start_frame <= track2.last_frame
            and track2.start_frame <= track1.last_frame
        )

    def _trim_bounce_start(self, track: Track) -> Track:
        """Remove cyclic start and rolling end segments from the track."""
        if not track.positions:
            return track

        # Remove cyclic sequences at the beginning
        sequences = find_cyclic_sequences(track.positions)
        if sequences:
            for _, end in sequences:
                track.start_frame = end
                break

        # Remove rolling sequences at the end
        sequences = find_rolling_sequences(track.positions)
        if sequences:
            for start, _ in sequences:
                track.last_frame = start
                break

        # Trim positions if start/end frames were updated
        track.positions = [
            pos
            for pos in track.positions
            if track.start_frame <= pos[1] <= track.last_frame
        ]
        return track

    def _filter_short_tracks(self, episodes: List[Track]) -> List[Track]:
        """Filter, extend, merge, and clean up tracks."""
        episodes = [self._trim_bounce_start(ep) for ep in episodes]

        # 2. Keep only tracks longer than minimum duration
        long_tracks = [
            ep for ep in episodes if ep.duration_sec() >= self.min_duration_sec
        ]
        sorted_tracks = sorted(long_tracks, key=lambda x: x.duration_sec(), reverse=True)

        # 3. Remove overlapping tracks (keep longest)
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

        # 4. Extend each track by 1 second on both sides
        frames_to_extend = int(self.fps)
        extended = []
        for ep in filtered:
            ep.start_frame = max(0, ep.start_frame - frames_to_extend)
            ep.last_frame = ep.last_frame + frames_to_extend
            extended.append(ep)

        # 5. Merge overlapping extended tracks
        merged = []
        used = set()
        sorted_ext = sorted(extended, key=lambda x: x.start_frame)
        for i, track1 in enumerate(sorted_ext):
            if i in used:
                continue
            merged_track = track1
            merged_positions = list(merged_track.positions)
            merged_ids = [merged_track.track_id]
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
                    merged_ids.append(track2.track_id)
                    used.add(j)
            merged_track.positions = sorted(merged_positions, key=lambda x: x[1])
            duration_frames = merged_track.last_frame - merged_track.start_frame + 1
            merged_track.duration_sec = lambda: duration_frames / self.fps
            self.track_distances[merged_track.track_id] = (
                f"Merged: {', '.join(map(str, merged_ids))}"
                if len(merged_ids) > 1
                else self.track_distances.get(merged_track.track_id, "Unknown")
            )
            merged.append(merged_track)

        # 6. Final trim after merging
        merged = [self._trim_bounce_start(ep) for ep in merged]
        return sorted(merged, key=lambda x: x.start_frame)

    def _process_detections(self, df: pd.DataFrame) -> None:
        """Run tracker and collect completed tracks."""
        tracker = BallTracker(
            buffer_size=2500,
            max_disappeared=40,
            max_distance=self.max_distance,
            fps=self.fps,
        )
        close_tracks = []

        # Optimization: process detections per frame
        for frame_num, frame_group in df.groupby("Frame"):
            frame_num = int(frame_num)
            detections = []
            for row in frame_group.itertuples(index=False):
                if not np.isnan(row.X) and not np.isnan(row.Y):
                    r = getattr(row, "Radius", DEFAULT_DETECTION_BOX_RADIUS)
                    detections.append(
                        {
                            "x1": row.X - r,
                            "y1": row.Y - r,
                            "x2": row.X + r,
                            "y2": row.Y + r,
                            "radius": r,
                            "confidence": row.Visibility,
                            "cls_id": 0,
                        }
                    )
            _, _, close_track = tracker.update(detections, frame_num)
            close_tracks.extend(close_track)

        episodes = []
        for track in close_tracks:
            if not track.positions:
                continue
            self.track_distances[track.track_id] = (
                f"Distance > {self.max_distance}px"
                if track.reason == "Unknown"
                else track.reason
            )
            episodes.append(track)

        # Add remaining active tracks
        for _, track in tracker.tracks.items():
            if not track.positions:
                continue
            episodes.append(track)

        self.tracks = self._filter_short_tracks(episodes)

    def _save_tracks_to_json(self) -> None:
        """Save each track to a separate JSON file."""
        csv_name = os.path.splitext(os.path.basename(self.csv_path))[0]
        video_basename = (
            os.path.basename(os.path.dirname(self.csv_path))
            or csv_name.replace("_predict_ball", "")
        )
        tracks_dir = os.path.join(self.output_dir, video_basename, "tracks")
        os.makedirs(tracks_dir, exist_ok=True)

        for track in self.tracks:
            track.calculate_stats()  # Explicitly calculate stats before export
            file_path = os.path.join(tracks_dir, f"track_{track.track_id:04d}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(track.to_dict(), f, indent=2, ensure_ascii=False)
        LOG.info("Saved %s tracks to: %s", len(self.tracks), tracks_dir)

    def run(self) -> None:
        """Main execution flow."""
        self._validate_csv()
        df = self._load_and_process_csv()
        self._process_detections(df)
        self._save_tracks_to_json()
        print(f"Done. Found {len(self.tracks)} tracks.")


def main():
    parser = argparse.ArgumentParser(description="Calculate tracks from CSV to JSON")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to ball.csv")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Root output directory for JSON",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Frames per second")
    parser.add_argument(
        "--max_distance", type=float, default=DEFAULT_MAX_DISTANCE, help="Max tracking distance"
    )
    parser.add_argument(
        "--min_duration_sec",
        type=float,
        default=DEFAULT_MIN_DURATION_SEC,
        help="Minimum track duration",
    )
    parser.add_argument(
        "--max_x_displacement",
        type=float,
        default=DEFAULT_MAX_X_DISPLACEMENT,
        help="Max X displacement",
    )
    parser.add_argument(
        "--min_y_displacement",
        type=float,
        default=DEFAULT_MIN_Y_DISPLACEMENT,
        help="Min Y displacement",
    )
    parser.add_argument(
        "--bounce_frames",
        type=int,
        default=DEFAULT_BOUNCE_FRAMES,
        help="Frames to analyze bounce",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    calculator = TrackCalculator(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        fps=args.fps,
        max_distance=args.max_distance,
        min_duration_sec=args.min_duration_sec,
        max_x_displacement=args.max_x_displacement,
        min_y_displacement=args.min_y_displacement,
        bounce_frames=args.bounce_frames,
    )
    calculator.run()


if __name__ == "__main__":
    main()
