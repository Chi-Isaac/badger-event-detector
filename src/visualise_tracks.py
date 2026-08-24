import argparse
from pathlib import Path
import cv2
import json
import random
import colorsys

def parse_args():
    parser = argparse.ArgumentParser(description="Visualise tracks on video")
    parser.add_argument("--input", type=Path, required=True, help="Path to source video")
    parser.add_argument("--tracks", type=Path, required=True, help="Path to tracks.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Path to output annotated video")
    return parser.parse_args()

# Uses track ID as a seed to randomly generate a consistent hue
# Colour converted from HSV space to RGB
def track_colour(track_id):
    rng = random.Random(track_id)
    hue = rng.random()
    saturation = 0.75
    value = 0.90
    
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    
    return (
        round(r * 255),
        round(g * 255),
        round(b * 255),
    )

# Reads JSONL file and groups track records by frame_index
def load_tracks_by_frame(tracks_path):
    tracks_by_frame = {}

    with open(tracks_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            frame_index = record["frame_index"]
            if frame_index not in tracks_by_frame:
                tracks_by_frame[frame_index] = []
            tracks_by_frame[frame_index].append(record)

    return tracks_by_frame

def visualise(video_path, tracks_path, output_path):
    tracks_by_frame = load_tracks_by_frame(tracks_path)
    
    cap = cv2.VideoCapture(str(video_path))
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Prepare output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    frame_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_tracks = tracks_by_frame.get(frame_index, [])
        
        # Draw rectangle around each tracked object
        for record in frame_tracks:
            x1, y1, x2, y2 = [int(coord) for coord in record["box_xyxy"]]
            track_id = record["track_id"]
            state = record.get("state", "unknown")
            confidence = record.get("confidence", 0.0)
            
            colour = track_colour(track_id)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        
        # Draw frame index on the top left corner of the frame
        cv2.putText(
            frame,
            f"frame: {frame_index}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        
        writer.write(frame)
        frame_index += 1
    
    # Release resources
    cap.release()
    writer.release()

def main():
    args = parse_args()
    visualise(args.input, args.tracks, args.output)

if __name__ == "__main__":
    main()