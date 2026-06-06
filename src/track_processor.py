#!/usr/bin/env python3
import cv2
import json
import os
import argparse
from tqdm import tqdm
from typing import List, Dict, Any

def parse_args():
    parser = argparse.ArgumentParser(description="Process ball tracks and create clips")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_dir", type=str, default="output", help="Root output directory")
    parser.add_argument("--json_dir", type=str, help="Directory with track JSON files")
    parser.add_argument("--split_dir", type=str, help="Directory to save individual rally clips")
    return parser.parse_args()

def process_video(video_path, tracks, output_path, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for track in tracks:
        start = track.get("start_frame", 0)
        end = track.get("last_frame", 0)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(start, end + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

    cap.release()
    out.release()

def split_rallies(video_path, tracks, split_dir, args):
    os.makedirs(split_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    for track in tracks:
        track_id = track.get("track_id", 0)
        start = track.get("start_frame", 0)
        end = track.get("last_frame", 0)

        out_path = os.path.join(split_dir, f"{video_basename}_rally_{track_id:04d}.mp4")
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(start, end + 1):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
    cap.release()

def main():
    args = parse_args()
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    json_dir = args.json_dir or os.path.join(args.output_dir, video_name, "tracks")

    tracks = []
    if os.path.exists(json_dir):
        for filename in sorted(os.listdir(json_dir)):
            if filename.endswith(".json"):
                with open(os.path.join(json_dir, filename), 'r') as f:
                    tracks.append(json.load(f))

    if not tracks:
        print(f"No tracks found in {json_dir}")
        return

    combined_path = os.path.join(args.output_dir, video_name, "combined.mp4")
    process_video(args.video_path, tracks, combined_path, args)
    print(f"Saved combined video to {combined_path}")

    if args.split_dir:
        split_rallies(args.video_path, tracks, args.split_dir, args)
        print(f"Saved individual clips to {args.split_dir}")

if __name__ == "__main__":
    main()
