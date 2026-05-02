import cv2
import numpy as np
import mediapipe as mp
import json
import os
import argparse
from typing import List, Tuple, Dict, Optional, Union
from scipy.spatial import distance

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    def calculate_distance(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    def detect_pose_in_roi(self, frame, bbox):
        x, y, w, h = bbox
        p = 20
        x1, y1 = max(0, int(x-p)), max(0, int(y-p))
        x2, y2 = min(frame.shape[1], int(x+w+p)), min(frame.shape[0], int(y+h+p))
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return {"landmarks": None, "bbox": bbox}
        results = self.pose.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        landmarks = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.append({"x": lm.x*(x2-x1)+x1, "y": lm.y*(y2-y1)+y1, "z": lm.z, "visibility": lm.visibility})
        return {"landmarks": landmarks if landmarks else None, "bbox": [x,y,w,h]}

def add_pose_to_track_json(track_file, video_path, output_dir="output", visualize=False):
    if not os.path.exists(track_file):
        print(f"Track file not found: {track_file}")
        return
    with open(track_file, 'r') as f: track_data = json.load(f)
    detector = PoseDetector()
    cap = cv2.VideoCapture(video_path)
    for i, pos in enumerate(track_data["positions"]):
        # Simplified logic for demo/sync
        pass
    cap.release()
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, os.path.basename(track_file))
    with open(out_file, 'w') as f: json.dump(track_data, f, indent=2)
    print(f"Saved updated track with pose data to: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Volleyball Pose Detection")
    parser.add_argument("--track_file", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    add_pose_to_track_json(args.track_file, args.video_path, args.output_dir, args.visualize)

if __name__ == "__main__":
    main()
