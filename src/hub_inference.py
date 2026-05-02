import cv2
import numpy as np
import csv
from ultralytics import YOLO
import os
import sys
import argparse

def run_hub_inference(video_path, model_url, api_key, output_dir, visualize=False, save_video=True):
    """
    Runs inference using a model from Ultralytics Hub on a video.
    """
    if api_key:
        os.environ["ULTRALYTICS_HUB_API_KEY"] = api_key

    print(f"Loading model from {model_url}...")
    try:
        model = YOLO(model_url)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    out_writer = None
    if save_video and output_dir:
        output_video_path = os.path.join(output_dir, f"{video_basename}_hub_predict.mp4")
        print(f"Saving video to {output_video_path}")
        out_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    csv_file = None
    csv_writer = None
    if output_dir:
        csv_path = os.path.join(output_dir, f"{video_basename}_hub_predict.csv")
        print(f"Saving tracking data to {csv_path}")
        csv_file = open(csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Frame", "Visibility", "X", "Y", "W", "H", "Conf", "Class"])

    print(f"Starting inference on {video_path}...")
    results = model.track(source=video_path, stream=True, persist=True, conf=0.25, iou=0.45)

    try:
        for i, result in enumerate(results):
            annotated_frame = result.plot()
            visibility = 0
            bx, by, bw, bh = 0, 0, 0, 0
            conf = 0.0
            cls = -1

            if result.boxes and len(result.boxes) > 0:
                box = result.boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                bx = int((x1 + x2) / 2)
                by = int((y1 + y2) / 2)
                bw = int(x2 - x1)
                bh = int(y2 - y1)
                visibility = 1

            if csv_writer:
                csv_writer.writerow([i, visibility, bx, by, bw, bh, conf, cls])
            if out_writer:
                out_writer.write(annotated_frame)
            if visualize:
                cv2.imshow("Hub Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        if csv_file: csv_file.close()
        if out_writer: out_writer.release()
        if visualize: cv2.destroyAllWindows()
        cap.release()
    print(f"Processing complete.")

def main():
    parser = argparse.ArgumentParser(description="Ultralytics Hub Tracking")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--hub_model", type=str, required=True)
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    run_hub_inference(args.video_path, args.hub_model, args.api_key, args.output_dir, args.visualize)

if __name__ == "__main__":
    main()
