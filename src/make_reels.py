#!/usr/bin/env python3
import cv2
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from src.track_utils import find_cyclic_sequences, find_rolling_sequences

def parse_args():
    parser = argparse.ArgumentParser(description="Create vertical reels from ball tracks")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--track_json", type=str, help="Path to a single track JSON file")
    parser.add_argument("--json_dir", type=str, help="Directory with track JSON files")
    parser.add_argument("--output_dir", type=str, default="output/reels", help="Output directory")
    parser.add_argument("--margin", type=int, default=50, help="Margin around ball in pixels")
    parser.add_argument("--visualize", action="store_true", help="Visualize result")
    parser.add_argument("--fps", type=float, default=30.0, help="Output FPS")
    return parser.parse_args()

def crop_frame(frame, center, target_ratio=(9, 16)):
    h, w = frame.shape[:2]
    cx, cy = center

    # Target size for 9:16 based on frame height
    target_h = h
    target_w = int(h * target_ratio[0] / target_ratio[1])

    if target_w > w:
        target_w = w
        target_h = int(w * target_ratio[1] / target_ratio[0])

    x1 = int(cx - target_w / 2)
    x2 = x1 + target_w

    if x1 < 0:
        x1 = 0
        x2 = target_w
    if x2 > w:
        x2 = w
        x1 = w - target_w

    y1 = int(cy - target_h / 2)
    y2 = y1 + target_h

    if y1 < 0:
        y1 = 0
        y2 = target_h
    if y2 > h:
        y2 = h
        y1 = h - target_h

    return frame[y1:y2, x1:x2]

def process_track(video_path, track_data, output_dir, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    track_id = track_data.get("track_id", 0)
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"reel_{video_basename}_{track_id:04d}.mp4")

    os.makedirs(output_dir, exist_ok=True)

    positions = track_data.get("positions", [])
    if not positions:
        return

    # Find center positions for each frame
    pos_dict = {int(p[1]): p[0] for p in positions}
    start_frame = int(track_data.get("start_frame", positions[0][1]))
    last_frame = int(track_data.get("last_frame", positions[-1][1]))

    # Sample crop to get dimensions
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, frame = cap.read()
    if not ret:
        return

    dummy_crop = crop_frame(frame, (frame.shape[1]//2, frame.shape[0]//2))
    h_out, w_out = dummy_crop.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, args.fps, (w_out, h_out))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Smoothing center X
    all_x = []
    for f in range(start_frame, last_frame + 1):
        if f in pos_dict:
            all_x.append(pos_dict[f][0])
        else:
            all_x.append(all_x[-1] if all_x else frame.shape[1]//2)

    # Simple moving average for X
    window = 15
    smoothed_x = np.convolve(all_x, np.ones(window)/window, mode='same')

    for i, f in enumerate(tqdm(range(start_frame, last_frame + 1), desc=f"Reel {track_id}")):
        ret, frame = cap.read()
        if not ret:
            break

        center = (int(smoothed_x[i]), frame.shape[0]//2)
        crop = crop_frame(frame, center)

        if crop.shape[0] != h_out or crop.shape[1] != w_out:
            crop = cv2.resize(crop, (w_out, h_out))

        out.write(crop)

        if args.visualize:
            cv2.imshow("Reel", crop)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    print(f"Saved reel to {output_path}")

def main():
    args = parse_args()

    tracks = []
    if args.track_json:
        with open(args.track_json, 'r') as f:
            tracks.append(json.load(f))
    elif args.json_dir:
        for filename in sorted(os.listdir(args.json_dir)):
            if filename.endswith(".json"):
                with open(os.path.join(args.json_dir, filename), 'r') as f:
                    tracks.append(json.load(f))
    else:
        # Try to find tracks automatically if video_path provided
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        potential_dir = os.path.join("output", video_name, "tracks")
        if os.path.exists(potential_dir):
            for filename in sorted(os.listdir(potential_dir)):
                if filename.endswith(".json"):
                    with open(os.path.join(potential_dir, filename), 'r') as f:
                        tracks.append(json.load(f))
    
    for track_data in tracks:
        process_track(args.video_path, track_data, args.output_dir, args)

if __name__ == "__main__":
    main()
