import json
import argparse
import os
import numpy as np

def analyze_trajectories(json_dir):
    """Analyzes ball trajectories from JSON files in a directory."""
    if not os.path.exists(json_dir):
        print(f"Directory not found: {json_dir}")
        return

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return

    print(f"Analyzing {len(json_files)} trajectories in {json_dir}")

    all_speeds = []
    all_heights = []

    for file_name in json_files:
        with open(os.path.join(json_dir, file_name), 'r') as f:
            data = json.load(f)
            avg_speed = data.get('avg_speed', 0)
            max_height = data.get('max_height', 0)

            if avg_speed > 0:
                all_speeds.append(avg_speed)
            if max_height > 0:
                all_heights.append(max_height)

    if all_speeds:
        print(f"Average Speed: {np.mean(all_speeds):.2f} px/frame")
        print(f"Max Average Speed: {np.max(all_speeds):.2f} px/frame")

    if all_heights:
        print(f"Average Max Height (min Y): {np.mean(all_heights):.2f} px")

def main():
    parser = argparse.ArgumentParser(description="Analyze ball trajectories in Zone 4")
    parser.add_argument("--json_dir", type=str, required=True, help="Path to directory with track JSONs")
    args = parser.parse_args()

    analyze_trajectories(args.json_dir)

if __name__ == "__main__":
    main()
