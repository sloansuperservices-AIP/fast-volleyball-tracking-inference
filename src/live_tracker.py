"""
Live volleyball game tracker.

Supports webcam, RTSP streams, and video files with real-time HUD overlay:
  - Ball trajectory trail
  - FPS counter
  - Rally counter and duration timer
  - Estimated ball speed (km/h)
  - Optional recording to file

Usage:
    uv run src/live_tracker.py \
        --source 0 \
        --model_path models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx

    uv run src/live_tracker.py \
        --source rtsp://192.168.1.10:554/stream \
        --model_path models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx \
        --record
"""

import argparse
import os
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

# Allow running directly or as module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference_onnx_seq9_gray_v2 import (
    load_onnx_model,
    postprocess_output,
)


# ---------------------------------------------------------------------------
# Real-time rally state machine
# ---------------------------------------------------------------------------

class LiveRallyDetector:
    """Tracks rally state from a stream of ball visibility flags."""

    def __init__(self, fps: float = 30.0, max_gap_sec: float = 2.0):
        self.fps = max(fps, 1.0)
        self.max_gap_frames = int(self.fps * max_gap_sec)
        self.rally_count = 0
        self.in_rally = False
        self.rally_start_frame = 0
        self.current_frame = 0
        self.last_visible_frame = -9999

    def update(self, frame_number: int, visible: bool) -> None:
        self.current_frame = frame_number
        if visible:
            if not self.in_rally:
                self.in_rally = True
                self.rally_start_frame = frame_number
                self.rally_count += 1
            self.last_visible_frame = frame_number
        else:
            if self.in_rally:
                gap = frame_number - self.last_visible_frame
                if gap > self.max_gap_frames:
                    self.in_rally = False

    @property
    def rally_duration_sec(self) -> float:
        if not self.in_rally:
            return 0.0
        return (self.current_frame - self.rally_start_frame) / self.fps


# ---------------------------------------------------------------------------
# HUD drawing helpers
# ---------------------------------------------------------------------------

def _draw_hud(
    frame: np.ndarray,
    track_points: deque,
    visibility: int,
    ball_x: int,
    ball_y: int,
    display_fps: float,
    speed_kmh: float,
    rally_detector: LiveRallyDetector,
    is_live: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]

    # --- semi-transparent top bar ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # FPS
    cv2.putText(frame, f"FPS: {display_fps:.0f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 80), 2)

    # Speed
    speed_color = (0, 80, 255) if speed_kmh > 50 else (0, 200, 120)
    cv2.putText(frame, f"{speed_kmh:.0f} km/h", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, speed_color, 2)

    # Rally info (centred)
    rally_color = (0, 255, 130) if rally_detector.in_rally else (130, 130, 130)
    rally_text = f"Rally #{rally_detector.rally_count}"
    (tw, _), _ = cv2.getTextSize(rally_text, cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)
    cv2.putText(frame, rally_text, (w // 2 - tw // 2, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, rally_color, 2)

    if rally_detector.in_rally:
        dur_text = f"{rally_detector.rally_duration_sec:.1f}s"
        (dw, _), _ = cv2.getTextSize(dur_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(frame, dur_text, (w // 2 - dw // 2, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 220), 2)

    # LIVE badge (top-right)
    if is_live:
        badge_x = w - 85
        cv2.rectangle(frame, (badge_x, 8), (w - 8, 38), (0, 0, 210), -1)
        cv2.putText(frame, "LIVE", (badge_x + 8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    # --- trajectory trail ---
    points = list(track_points)
    n = len(points)
    for i in range(1, n):
        p0, p1 = points[i - 1], points[i]
        if p0 is None or p1 is None:
            continue
        alpha = i / max(n, 1)
        color = (int(50 * alpha), int(80 + 120 * alpha), int(255 * (1 - alpha) + 80 * alpha))
        thickness = max(1, int(4 * alpha))
        cv2.line(frame, p0, p1, color, thickness)

    # Current ball
    if visibility and ball_x > 0 and ball_y > 0:
        cv2.circle(frame, (ball_x, ball_y), 9, (0, 0, 255), -1)
        cv2.circle(frame, (ball_x, ball_y), 12, (255, 255, 255), 2)

    return frame


# ---------------------------------------------------------------------------
# Main live tracker class
# ---------------------------------------------------------------------------

class LiveTracker:
    """Real-time volleyball ball tracker for live games or video files."""

    # Approximate volleyball court width (m) – used for speed estimation
    COURT_WIDTH_M = 18.0

    def __init__(
        self,
        source,
        model_path: str,
        trail_length: int = 25,
        display: bool = True,
        record: bool = False,
        output_dir: str = "output",
    ):
        self.source = source
        self.model_path = model_path
        self.trail_length = trail_length
        self.display = display
        self.record = record
        self.output_dir = output_dir

        self.input_width = 512
        self.input_height = 288

        # Load ONNX model
        (
            self.session,
            self.has_gru,
            self.out_dim,
            self.h0_shape,
            self.batch_size,
        ) = load_onnx_model(model_path)

        # State
        self.track_points: deque = deque(maxlen=trail_length)
        self.frame_buffer: deque = deque(maxlen=self.batch_size)
        self.fps_history: deque = deque(maxlen=30)
        self.speed_history: deque = deque(maxlen=15)
        self._prev_ball_pos = None
        self._prev_ball_frame = 0

    # ------------------------------------------------------------------
    # Source handling
    # ------------------------------------------------------------------

    def _open_source(self):
        src = self.source
        is_live = False

        if isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
            cap = cv2.VideoCapture(int(src))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            is_live = True
        elif isinstance(src, str) and any(
            src.startswith(p) for p in ("rtsp://", "rtmp://", "http://", "https://")
        ):
            cap = cv2.VideoCapture(src)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            is_live = True
        else:
            cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            raise ValueError(f"Cannot open source: {src!r}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 240:
            fps = 30.0

        return cap, float(fps), is_live

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _setup_recorder(self, w: int, h: int, fps: float):
        if not self.record:
            return None
        os.makedirs(self.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"live_{ts}.mp4")
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )
        print(f"Recording to: {path}")
        return writer

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer(self) -> tuple[int, int, int]:
        """Run one inference pass on the current frame buffer.

        Returns (visibility, ball_x_model, ball_y_model) in model coordinates.
        """
        input_tensor = np.stack(list(self.frame_buffer), axis=2)   # (H, W, T)
        input_tensor = np.expand_dims(input_tensor, axis=0)         # (1, H, W, T)
        input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))     # (1, T, H, W)

        inputs = {self.session.get_inputs()[0].name: input_tensor}
        output = self.session.run(None, inputs)[0]

        predictions = postprocess_output(
            output,
            input_height=self.input_height,
            input_width=self.input_width,
            out_dim=self.out_dim,
        )
        return predictions[-1]  # (visibility, cx, cy) for the latest frame

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        cap, source_fps, is_live = self._open_source()
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        rally_detector = LiveRallyDetector(fps=source_fps)
        recorder = self._setup_recorder(frame_w, frame_h, source_fps)

        # Async frame reader – drops old frames for live sources
        frame_q: queue.Queue = queue.Queue(maxsize=4)

        def _reader():
            while True:
                ret, frame = cap.read()
                if not ret:
                    frame_q.put(None)
                    break
                if is_live and frame_q.full():
                    try:
                        frame_q.get_nowait()
                    except queue.Empty:
                        pass
                frame_q.put(frame)

        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        frame_number = 0
        last_time = time.perf_counter()
        speed_kmh = 0.0

        print("Volleyball live tracking started. Press 'q' to quit.")

        while True:
            frame = frame_q.get()
            if frame is None:
                break

            # Preprocess and buffer
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.resize(gray, (self.input_width, self.input_height))
            self.frame_buffer.append(gray_r.astype(np.float32) / 255.0)

            visibility, cx, cy = 0, 0, 0

            if len(self.frame_buffer) >= self.batch_size:
                visibility, cx, cy = self._infer()

            ball_x = int(cx * frame_w / self.input_width) if visibility else 0
            ball_y = int(cy * frame_h / self.input_height) if visibility else 0

            # Speed estimation
            if visibility and self._prev_ball_pos is not None:
                dt = max(1, frame_number - self._prev_ball_frame)
                dist_px = float(
                    np.linalg.norm(
                        np.array([ball_x, ball_y]) - np.array(self._prev_ball_pos)
                    )
                )
                px_per_frame = dist_px / dt
                # px/frame → m/s → km/h  (approximate using court width)
                speed_ms = px_per_frame * source_fps * self.COURT_WIDTH_M / frame_w
                speed_kmh = speed_ms * 3.6
                self.speed_history.append(speed_kmh)

            if visibility:
                self.track_points.append((ball_x, ball_y))
                self._prev_ball_pos = (ball_x, ball_y)
                self._prev_ball_frame = frame_number
            else:
                self.track_points.append(None)

            rally_detector.update(frame_number, bool(visibility))

            # Display FPS
            now = time.perf_counter()
            elapsed = now - last_time
            self.fps_history.append(1.0 / max(elapsed, 1e-6))
            last_time = now
            display_fps = float(np.mean(self.fps_history))
            avg_speed = float(np.mean(self.speed_history)) if self.speed_history else 0.0

            frame = _draw_hud(
                frame,
                self.track_points,
                visibility,
                ball_x,
                ball_y,
                display_fps,
                avg_speed,
                rally_detector,
                is_live,
            )

            if recorder:
                recorder.write(frame)

            if self.display:
                cv2.namedWindow("Volleyball Live Tracking", cv2.WINDOW_NORMAL)
                cv2.imshow("Volleyball Live Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_number += 1

        # Cleanup
        cap.release()
        if recorder:
            recorder.release()
        if self.display:
            cv2.destroyAllWindows()

        print(f"\nSession ended — total rallies detected: {rally_detector.rally_count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Live volleyball game tracker")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: camera index (0, 1, ...), RTSP URL, or file path",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--trail_length",
        type=int,
        default=25,
        help="Number of past ball positions to draw as trail",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record output video to --output_dir",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save recordings",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Disable live display window (useful for headless recording)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tracker = LiveTracker(
        source=args.source,
        model_path=args.model_path,
        trail_length=args.trail_length,
        display=not args.no_display,
        record=args.record,
        output_dir=args.output_dir,
    )
    tracker.run()
