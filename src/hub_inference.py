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

    Args:
        video_path (str): Path to the input video.
        model_url (str): URL or ID of the model on Ultralytics Hub.
        api_key (str): Ultralytics Hub API key.
        output_dir (str): Directory to save output CSV and video.
        visualize (bool): Whether to show the video processing in a window.
        save_video (bool): Whether to save the processed video.
    """

    # Set API key
    if api_key:
        os.environ["ULTRALYTICS_HUB_API_KEY"] = api_key

    print(f"Loading model from {model_url}...")
    # Load model
    try:
        model = YOLO(model_url)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create output directory
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Video writer setup
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
        out_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    csv_file = None
    csv_writer = None
    if output_dir:
        csv_path = os.path.join(output_dir, f"{video_basename}_hub_predict.csv")
        print(f"Saving tracking data to {csv_path}")
        csv_file = open(csv_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Frame", "Visibility", "X", "Y", "W", "H", "Conf", "Class"])

    print(f"Starting inference on {video_path}...")

    # Run inference
    # stream=True returns a generator so we can process frame by frame
    # conf=0.25, iou=0.45 from user request
    results = model.track(source=video_path, stream=True, persist=True, conf=0.25, iou=0.45)

    try:
        for i, result in enumerate(results):
            annotated_frame = result.plot()

            visibility = 0
            bx, by, bw, bh = 0, 0, 0, 0
            conf = 0.0
            cls = -1

            # Check if there are detections
            if result.boxes and len(result.boxes) > 0:
                # We take the first detection (highest confidence) for the CSV tracking
                box = result.boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                bx = int((x1 + x2) / 2)
                by = int((y1 + y2) / 2)
                bw = int(x2 - x1)
                bh = int(y2 - y1)
                visibility = 1

            # Save to CSV
            if csv_writer:
                csv_writer.writerow([i, visibility, bx, by, bw, bh, conf, cls])

            # Write video
            if out_writer:
                out_writer.write(annotated_frame)

            # Visualize
            if visualize:
                cv2.imshow("Hub Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        if csv_file:
            csv_file.close()
        if out_writer:
            out_writer.release()
        if visualize:
            cv2.destroyAllWindows()
        cap.release()

    print(f"Processing complete.")

def main():
    parser = argparse.ArgumentParser(description="Ultralytics Hub Inference")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--model_url", type=str, required=True, help="Hub model URL or ID")
    parser.add_argument("--api_key", type=str, help="Hub API key")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")

    args = parser.parse_args()

    run_hub_inference(
        video_path=args.video_path,
        model_url=args.model_url,
        api_key=args.api_key,
        output_dir=args.output_dir,
        visualize=args.visualize
    )

if __name__ == "__main__":
    main()
