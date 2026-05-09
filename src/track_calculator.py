#!/usr/bin/env python3
import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from .ball_tracker import BallTracker, Track
    from .constants import (
        COURT_LENGTH_M,
        COURT_WIDTH_M,
        DEFAULT_BOUNCE_FRAMES,
        DEFAULT_DETECTION_BOX_RADIUS,
        DEFAULT_EXTEND_SECONDS,
        DEFAULT_FPS,
        DEFAULT_MAX_DISTANCE,
        DEFAULT_MAX_X_DISPLACEMENT,
        DEFAULT_MIN_DURATION_SEC,
        DEFAULT_MIN_Y_DISPLACEMENT,
        DEFAULT_NET_Y_THRESHOLD,
    )
    from .court_transformer import CoordinateTransformer, CourtTransformer
    from .models import CourtGeometry
    from .track_utils import find_cyclic_sequences, find_rolling_sequences
except ImportError:
    from ball_tracker import BallTracker, Track
    from constants import (
        COURT_LENGTH_M,
        COURT_WIDTH_M,
        DEFAULT_BOUNCE_FRAMES,
        DEFAULT_DETECTION_BOX_RADIUS,
        DEFAULT_EXTEND_SECONDS,
        DEFAULT_FPS,
        DEFAULT_MAX_DISTANCE,
        DEFAULT_MAX_X_DISPLACEMENT,
        DEFAULT_MIN_DURATION_SEC,
        DEFAULT_MIN_Y_DISPLACEMENT,
        DEFAULT_NET_Y_THRESHOLD,
    )
    from court_transformer import CoordinateTransformer, CourtTransformer
    from models import CourtGeometry
    from track_utils import find_cyclic_sequences, find_rolling_sequences

LOG = logging.getLogger(__name__)

REFERENCE_VIDEO_WIDTH = 1920.0
NET_HEIGHT_CM = 243.0
POST_PAUSE_TAIL_SECONDS = 0.5
MAX_MERGE_GAP_FRAMES = 40


@dataclass(frozen=True)
class TrackCalculatorConfig:
    csv_path: str
    output_dir: str
    fps: float
    max_distance: float
    min_duration_sec: float
    max_x_displacement: float
    min_y_displacement: float
    bounce_frames: int
    court_json_path: Optional[str]
    video_width: Optional[int]
    video_height: Optional[int]


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def resolve_video_basename(csv_path: str) -> str:
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    parent = os.path.basename(os.path.dirname(csv_path))
    if csv_name == "ball" and parent:
        return parent
    if csv_name.endswith("_predict_ball"):
        return csv_name[: -len("_predict_ball")]
    if csv_name.endswith("_ball"):
        return csv_name[: -len("_ball")]
    return csv_name


class TrackCalculator:
    def __init__(self, config: TrackCalculatorConfig) -> None:
        self.config = config
        self.tracks: List[Track] = []
        self._camera_position = "unknown"
        self._distance_unit = "px"
        self._cm_per_px_scale: Optional[float] = None
        self._frame_width_scale = self._compute_frame_width_scale()
        self._scaled_max_distance = self.config.max_distance * self._frame_width_scale

        transformer = CourtTransformer(config.court_json_path)
        result = transformer.load()
        self._court_geometry = result.geometry
        self._court_matrix = result.matrix
        self._coordinate_transformer = CoordinateTransformer(self._court_geometry, self._court_matrix)
        self._court_enabled = config.court_json_path is not None and self._court_geometry

        if config.court_json_path and not self._court_geometry:
            LOG.warning("Court JSON provided but could not be loaded, using image coordinates")
        else:
            self._camera_position = self._classify_camera_position()

        if self._court_enabled:
            self._cm_per_px_scale = self._calculate_cm_per_px_scale()
            if self._cm_per_px_scale is not None:
                self._distance_unit = "cm"

        LOG.info(
            "Tracking distance scale: width=%s ref_width=%s coeff=%.4f max_distance=%.2f->%.2f",
            self.config.video_width,
            int(REFERENCE_VIDEO_WIDTH),
            self._frame_width_scale,
            self.config.max_distance,
            self._scaled_max_distance,
        )
        if self._cm_per_px_scale is not None:
            LOG.info(
                "Court scale active: camera=%s cm_per_px=%.6f (unit=%s)",
                self._camera_position,
                self._cm_per_px_scale,
                self._distance_unit,
            )

    def _compute_frame_width_scale(self) -> float:
        width = self.config.video_width
        if width is None or width <= 0:
            return 1.0
        return width / REFERENCE_VIDEO_WIDTH

    @staticmethod
    def _distance_px(p1: Any, p2: Any) -> float:
        return float(np.hypot(float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1])))

    def _calculate_cm_per_px_scale(self) -> Optional[float]:
        if not self._court_geometry:
            return None
        keypoints = self._court_geometry.keypoints
        if len(keypoints) < 8:
            return None

        candidates_cm_per_px: List[float] = []
        p1, p3, p4 = keypoints[0], keypoints[2], keypoints[3]
        span_px = 0.0
        span_cm = 0.0
        if self._camera_position == "backline":
            span_px = self._distance_px(p1, p4)
            span_cm = COURT_WIDTH_M * 100.0
        elif self._camera_position == "sideline":
            span_px = self._distance_px(p3, p4)
            span_cm = COURT_LENGTH_M * 100.0

        if span_px > 1e-6 and span_cm > 0:
            candidates_cm_per_px.append(span_cm / span_px)

        p5, p6, p7, p8 = keypoints[4], keypoints[5], keypoints[6], keypoints[7]
        net_right_px = self._distance_px(p8, p6)
        net_left_px = self._distance_px(p7, p5)
        net_height_px_samples = [d for d in (net_right_px, net_left_px) if d > 1e-6]
        if net_height_px_samples:
            mean_net_height_px = float(np.mean(net_height_px_samples))
            candidates_cm_per_px.append(NET_HEIGHT_CM / mean_net_height_px)

        if not candidates_cm_per_px:
            return None
        return float(np.mean(candidates_cm_per_px))

    def _validate_csv(self) -> None:
        if not os.path.exists(self.config.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.config.csv_path}")

    def _load_and_process_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.csv_path)
        df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
        df["Visibility"] = pd.to_numeric(df["Visibility"], errors="coerce")
        df["X"] = pd.to_numeric(df["X"], errors="coerce")
        df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
        df.loc[(df["X"] == -1) | (df["Visibility"] == 0), ["X", "Y"]] = np.nan
        self._maybe_rescale_court_geometry(df)
        return df

    def _maybe_rescale_court_geometry(self, df: pd.DataFrame) -> None:
        if not self._court_enabled or not self._court_geometry:
            return
        target_w = self.config.video_width
        target_h = self.config.video_height

        if target_w is None or target_h is None:
            max_x = df["X"].max(skipna=True)
            max_y = df["Y"].max(skipna=True)
            if pd.notna(max_x) and pd.notna(max_y) and (max_x > self._court_geometry.image_width or max_y > self._court_geometry.image_height):
                target_w = max(int(max_x) + 1, self._court_geometry.image_width)
                target_h = max(int(max_y) + 1, self._court_geometry.image_height)
                LOG.warning("Rescaling court geometry to %sx%s", target_w, target_h)

        if target_w is None or target_h is None or target_w <= 0 or target_h <= 0:
            return
        if target_w == self._court_geometry.image_width and target_h == self._court_geometry.image_height:
            return

        scale_x = target_w / self._court_geometry.image_width
        scale_y = target_h / self._court_geometry.image_height
        scaled_keypoints = tuple((x * scale_x, y * scale_y) for x, y in self._court_geometry.keypoints)
        self._court_geometry = CourtGeometry(
            length_m=self._court_geometry.length_m,
            width_m=self._court_geometry.width_m,
            net_height_m=self._court_geometry.net_height_m,
            image_width=target_w,
            image_height=target_h,
            keypoints=scaled_keypoints,
        )
        self._court_matrix = CourtTransformer._calculate_transform(scaled_keypoints)
        self._coordinate_transformer = CoordinateTransformer(self._court_geometry, self._court_matrix)
        self._camera_position = self._classify_camera_position()
        self._cm_per_px_scale = self._calculate_cm_per_px_scale()
        if self._cm_per_px_scale is not None:
            self._distance_unit = "cm"

    def _classify_camera_position(self) -> str:
        if not self._court_geometry or len(self._court_geometry.keypoints) < 8:
            return "unknown"
        p1, p2, p3, p4 = self._court_geometry.keypoints[:4]
        p7, p8 = self._court_geometry.keypoints[6], self._court_geometry.keypoints[7]
        dx = p8[0] - p7[0]
        dy = p8[1] - p7[1]
        court_span = max(np.hypot(p4[0] - p1[0], p4[1] - p1[1]), np.hypot(p3[0] - p2[0], p3[1] - p2[1]), 1.0)
        net_span = np.hypot(dx, dy)
        net_span_ratio = net_span / court_span
        if abs(dx) < 1.0 or abs(dy) / (abs(dx) + 1e-6) > 0.7 or net_span_ratio < 0.28:
            return "sideline"
        left_depth = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        right_depth = np.hypot(p4[0] - p3[0], p4[1] - p3[1])
        depth_ratio = max(left_depth, right_depth) / max(1.0, min(left_depth, right_depth))
        net_mid_x = (p7[0] + p8[0]) / 2.0
        center_offset = abs(net_mid_x - self._court_geometry.image_width / 2.0) / max(1.0, self._court_geometry.image_width)
        if depth_ratio <= 1.35 and center_offset <= 0.12:
            return "backline"
        return "diagonal"

    @staticmethod
    def _is_overlapping(track1: Track, track2: Track) -> bool:
        return track1.start_frame <= track2.last_frame and track2.start_frame <= track1.last_frame

    def _trim_bounce_start(self, track: Track) -> Track:
        if not track.positions:
            return track
        original_start = track.start_frame
        original_end = track.last_frame
        sequences = find_cyclic_sequences(track.positions)
        if sequences:
            for _, end in sequences:
                track.start_frame = end
                break
        sequences = find_rolling_sequences(track.positions)
        if sequences:
            for start, _ in sequences:
                track.last_frame = start
                break
        if track.start_frame != original_start or track.last_frame != original_end:
            track.positions = [pos for pos in track.positions if track.start_frame <= pos[1] <= track.last_frame]
        return track

    def _net_y_at_x(self, x: float) -> float:
        if not self._court_geometry or len(self._court_geometry.keypoints) < 8:
            return DEFAULT_NET_Y_THRESHOLD
        net_left = self._court_geometry.keypoints[6]
        net_right = self._court_geometry.keypoints[7]
        dx = net_right[0] - net_left[0]
        if abs(dx) < 1e-6:
            return float(min(net_left[1], net_right[1]))
        t = (x - net_left[0]) / dx
        return float(net_left[1] + t * (net_right[1] - net_left[1]))

    def _is_above_net(self, x: float, y: float) -> bool:
        net_y = self._net_y_at_x(x)
        image_h = self._court_geometry.image_height if self._court_geometry else 720
        clearance = max(6.0, image_h * 0.01)
        if self._cm_per_px_scale is None:
            return y < (net_y - clearance)
        height_delta_cm = (net_y - y) * self._cm_per_px_scale
        clearance_cm = clearance * self._cm_per_px_scale
        return height_delta_cm > clearance_cm

    def _analyze_track_trajectory(self, track: Track) -> Dict[str, Any]:
        positions = sorted(track.positions, key=lambda p: p[1])
        if not positions:
            return {"camera_position": self._camera_position}
        above_flags = [self._is_above_net(pos[0][0], pos[0][1]) for pos in positions]
        last_above_idx = next((i for i, v in enumerate(reversed(above_flags)) if v), None)
        last_above_frame = positions[len(positions) - 1 - last_above_idx][1] if last_above_idx is not None else None
        return {
            "camera_position": self._camera_position,
            "last_above_net_frame": last_above_frame,
        }

    def _filter_short_tracks(self, episodes: List[Track]) -> List[Track]:
        episodes = [self._trim_bounce_start(ep) for ep in episodes]
        episodes = [track for track in episodes if track.duration_sec() >= self.config.min_duration_sec]

        # Remove overlapping
        sorted_tracks = sorted(episodes, key=lambda x: x.duration_sec(), reverse=True)
        filtered = []
        used = set()
        for i, t1 in enumerate(sorted_tracks):
            if i in used: continue
            filtered.append(t1)
            used.add(i)
            for j, t2 in enumerate(sorted_tracks):
                if j <= i or j in used: continue
                if self._is_overlapping(t1, t2): used.add(j)

        # Extend
        frames_to_extend = int(self.config.fps * DEFAULT_EXTEND_SECONDS)
        for track in filtered:
            track.start_frame = max(0, track.start_frame - frames_to_extend)
            track.last_frame = track.last_frame + frames_to_extend

        return sorted(filtered, key=lambda x: x.start_frame)

    def _process_detections(self, df: pd.DataFrame) -> None:
        tracker = BallTracker(
            buffer_size=2500,
            max_disappeared=40,
            max_distance=self._scaled_max_distance,
            fps=self.config.fps,
        )
        close_tracks: List[Track] = []
        all_frames = sorted(df["Frame"].dropna().astype(int).unique())
        for frame_num in all_frames:
            frame_rows = df[df["Frame"] == frame_num]
            detections = []
            for _, row in frame_rows.iterrows():
                if not np.isnan(row["X"]) and not np.isnan(row["Y"]):
                    detections.append(
                        {
                            "x1": row["X"] - DEFAULT_DETECTION_BOX_RADIUS,
                            "y1": row["Y"] - DEFAULT_DETECTION_BOX_RADIUS,
                            "x2": row["X"] + DEFAULT_DETECTION_BOX_RADIUS,
                            "y2": row["Y"] + DEFAULT_DETECTION_BOX_RADIUS,
                            "confidence": row["Visibility"],
                            "cls_id": 0,
                        }
                    )
            _, _, closed = tracker.update(detections, frame_num)
            close_tracks.extend(closed)

        for track_id in list(tracker.tracks.keys()):
            close_tracks.append(tracker.tracks[track_id])

        episodes = [track for track in close_tracks if track.positions]
        self.tracks = self._filter_short_tracks(episodes)

    def _save_tracks_to_json(self) -> None:
        video_basename = resolve_video_basename(self.config.csv_path)
        tracks_dir = os.path.join(self.config.output_dir, video_basename, "tracks")
        os.makedirs(tracks_dir, exist_ok=True)

        for track in self.tracks:
            track_dict = track.to_dict()
            track_dict["trajectory_analysis"] = self._analyze_track_trajectory(track)
            file_path = os.path.join(tracks_dir, f"track_{track.track_id:04d}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(track_dict, f, indent=2, ensure_ascii=False)
        LOG.info("Saved %s tracks to %s", len(self.tracks), tracks_dir)

    def run(self) -> None:
        self._validate_csv()
        df = self._load_and_process_csv()
        self._process_detections(df)
        self._save_tracks_to_json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate tracks from CSV to JSON")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to ball.csv")
    parser.add_argument("--court_json_path", type=str, help="Path to court keypoints JSON file")
    parser.add_argument("--output_dir", type=str, default="output", help="Root output directory")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="FPS")
    parser.add_argument("--max_distance", type=float, default=DEFAULT_MAX_DISTANCE, help="Max distance")
    parser.add_argument("--min_duration_sec", type=float, default=DEFAULT_MIN_DURATION_SEC, help="Min duration")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--video_width", type=int, default=1920)
    parser.add_argument("--video_height", type=int, default=1080)

    args = parser.parse_args()
    setup_logging(args.verbose)

    config = TrackCalculatorConfig(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        fps=args.fps,
        max_distance=args.max_distance,
        min_duration_sec=args.min_duration_sec,
        max_x_displacement=DEFAULT_MAX_X_DISPLACEMENT,
        min_y_displacement=DEFAULT_MIN_Y_DISPLACEMENT,
        bounce_frames=DEFAULT_BOUNCE_FRAMES,
        court_json_path=args.court_json_path,
        video_width=args.video_width,
        video_height=args.video_height,
    )
    calculator = TrackCalculator(config)
    calculator.run()


if __name__ == "__main__":
    main()
