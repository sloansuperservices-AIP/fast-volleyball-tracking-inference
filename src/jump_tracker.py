import cv2
import time
import numpy as np
from collections import deque
from ultralytics import YOLO
import argparse
import sys

# =========================
# CONFIG
# =========================

RTSP_URL = "rtsp://user:password@192.168.1.50:554/stream1"  # TODO: change this
MODEL_PATH = "models/yolov8n-pose.pt"  # pose model
WINDOW_NAME = "Volleyball Combine - Jump Tracking (YOLOv8n Pose)"

# Simple calibration: pixels per meter
# You should measure something on the court (e.g., net height in pixels vs 2.43m)
PIXELS_PER_METER = 300  # <-- adjust after calibration

# How many frames to keep in memory for jump detection
HISTORY_LEN = 120  # ~4 seconds at 30fps

# Minimum upward movement (in meters) to count as a jump
MIN_JUMP_DELTA_M = 0.20


# =========================
# Helper classes & functions
# =========================

class AthleteJumpTracker:
    """
    Tracks one athlete's vertical movement using pose keypoints
    and estimates jump height & events.
    """

    def __init__(self, history_len=HISTORY_LEN):
        self.ankle_ys = deque(maxlen=history_len)  # screen y-value (pixels)
        self.timestamps = deque(maxlen=history_len)
        self.baseline_ankle_y = None  # standing reference
        self.in_air = False
        self.last_jump_peak_m = 0.0
        self.best_jump_m = 0.0

    def update(self, ankle_y_px: float, timestamp: float):
        # y grows downward; smaller y => higher on screen
        self.ankle_ys.append(ankle_y_px)
        self.timestamps.append(timestamp)

        # Initialize baseline (average of first few frames)
        if self.baseline_ankle_y is None and len(self.ankle_ys) > 15:
            self.baseline_ankle_y = np.median(self.ankle_ys)

        if self.baseline_ankle_y is None:
            return None  # not enough data yet

        # Convert pixel difference to meters (rough)
        # baseline - current => positive when athlete moves upward
        delta_px = self.baseline_ankle_y - ankle_y_px
        current_jump_m = delta_px / PIXELS_PER_METER

        # Detect "in air" vs "on ground" roughly
        jump_threshold_m = MIN_JUMP_DELTA_M * 0.5
        if not self.in_air and current_jump_m > jump_threshold_m:
            # just took off
            self.in_air = True
            self.last_jump_peak_m = current_jump_m

        if self.in_air:
            # track max height during air time
            if current_jump_m > self.last_jump_peak_m:
                self.last_jump_peak_m = current_jump_m

            # detect landing: return near baseline
            if current_jump_m < jump_threshold_m * 0.3:
                # just landed
                self.in_air = False
                # finalize jump
                if self.last_jump_peak_m > MIN_JUMP_DELTA_M:
                    self.best_jump_m = max(self.best_jump_m, self.last_jump_peak_m)

        return {
            "current_jump_m": max(current_jump_m, 0.0),
            "best_jump_m": self.best_jump_m,
            "in_air": self.in_air,
        }


def draw_text(
    img,
    text,
    org,
    scale=0.7,
    thickness=2,
):
    """Simple text helper (keeps style consistent)."""
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


# =========================
# Main processing loop
# =========================

def main():
    parser = argparse.ArgumentParser(description="Volleyball Jump Tracker")
    parser.add_argument("--source", type=str, default=RTSP_URL, help="Video source (URL or file path)")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to YOLOv8 pose model")
    args = parser.parse_args()

    # Load YOLOv8n pose model
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"[ERROR] Could not load model {args.model}: {e}")
        return

    # Open IP camera stream or video file
    source = args.source
    # If source is an integer digit, convert to int for webcam index
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream: {source}")
        return

    tracker = AthleteJumpTracker()

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame (or end of stream)")
            break

        # Compute FPS
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

        # Run pose inference
        # imgsz can be tuned, e.g. 640
        results = model(frame, verbose=False)

        # We’ll assume the main athlete is the person with the largest bounding box
        person_boxes = []
        person_keypoints = []

        r = results[0]
        if r.boxes is not None and r.keypoints is not None:
            boxes = r.boxes.xyxy.cpu().numpy()  # (N, 4)
            kpts = r.keypoints.xy.cpu().numpy()  # (N, num_kpts, 2)

            for box, kpt in zip(boxes, kpts):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                area = w * h
                person_boxes.append((area, (x1, y1, x2, y2)))
                person_keypoints.append(kpt)

        jump_info = None

        if person_boxes:
            # pick largest
            max_idx = int(np.argmax([p[0] for p in person_boxes]))
            _, (x1, y1, x2, y2) = person_boxes[max_idx]
            kpt = person_keypoints[max_idx]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2,
            )

            # Draw keypoints (simple visualization)
            for (kx, ky) in kpt:
                cv2.circle(frame, (int(kx), int(ky)), 3, (0, 255, 255), -1)

            # Use ankle keypoint for vertical tracking.
            # YOLOv8-pose uses COCO skeleton; ankles are usually indices 15, 16 (right/left).
            # We'll take the lower of the two (bigger y) as "foot contact".
            ankle_indices = [15, 16]
            ankle_points = []
            for idx in ankle_indices:
                if 0 <= idx < len(kpt):
                    ax, ay = kpt[idx]
                    if ax > 0 and ay > 0:
                        ankle_points.append((ax, ay))

            if ankle_points:
                # choose ankle with greatest y value (closest to ground in image space)
                ankle_y = max(p[1] for p in ankle_points)
                # draw a circle where we’re sampling
                ankle_x = np.mean([p[0] for p in ankle_points]).astype(int)
                cv2.circle(frame, (int(ankle_x), int(ankle_y)), 6, (255, 0, 0), -1)

                # Update jump tracker
                jump_info = tracker.update(ankle_y, current_time)

        # =========================
        # HUD / On-screen stats
        # =========================
        hud_y = 30
        draw_text(frame, f"FPS: {fps:.1f}", (10, hud_y))
        hud_y += 30

        if tracker.baseline_ankle_y is None:
            draw_text(frame, "Calibrating baseline... stand still", (10, hud_y))
            hud_y += 30
        else:
            draw_text(frame, "Baseline calibrated", (10, hud_y))
            hud_y += 30

        if jump_info is not None:
            current_jump_cm = jump_info["current_jump_m"] * 100.0
            best_jump_cm = jump_info["best_jump_m"] * 100.0
            in_air = jump_info["in_air"]

            status = "IN AIR" if in_air else "ON GROUND"
            draw_text(frame, f"Status: {status}", (10, hud_y))
            hud_y += 30

            draw_text(frame, f"Current jump: {current_jump_cm:.1f} cm", (10, hud_y))
            hud_y += 30

            draw_text(frame, f"Best jump: {best_jump_cm:.1f} cm", (10, hud_y))
            hud_y += 30
        else:
            draw_text(frame, "No athlete detected", (10, hud_y))
            hud_y += 30

        # Show frame
        # In a headless environment like this sandbox, cv2.imshow might fail or do nothing.
        # But we keep it as the user requested code has it.
        # We might want to catch error if no display.
        try:
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):  # ESC or q to quit
                break
        except cv2.error:
            # Likely no display, just continue processing or break
            pass

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
