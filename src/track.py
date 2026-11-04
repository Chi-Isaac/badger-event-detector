import json
import argparse
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class Track:
    track_id: str
    last_xyxy: list[float]
    last_xywh: list[float]
    hits: int
    misses: int
    state: str
    
@dataclass
class TrackRecord:
    video_id: str
    frame_index: int
    timestamp: float
    track_id: str
    hits: int
    misses: int
    state: str
    confidence: float
    box_xywh: list[float]
    box_xyxy: list[float]

# Takes jsonl with all detections and returns detections keyed by frame index
def load_detections(path):
    detections_by_frame = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            detections_by_frame[record["frame_index"]].append(record)

    return detections_by_frame

# Reads the stride used by inference from run_info.json beside the detections file
def load_stride(detections_path):
    run_info_path = detections_path.parent / "run_info.json"
    if not run_info_path.exists():
        return 1
    run_info = json.loads(run_info_path.read_text())
    return run_info["inference_parameters"]["stride"]

def iou(box_a, box_b):
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b
    
    # Calculate coords of intersecting box
    x1_i = max(x1_a, x1_b)
    y1_i = max(y1_a, y1_b)
    x2_i = min(x2_a, x2_b)
    y2_i = min(y2_a, y2_b)
    
    # Calculate area of intersection
    w_i = max(0, x2_i - x1_i)
    h_i = max(0, y2_i - y1_i)
    area_i = w_i * h_i
    
    # Calculate area of both boxes
    area_a = max(0, x2_a - x1_a) * max(0, y2_a - y1_a)
    area_b = max(0, x2_b - x1_b) * max(0, y2_b - y1_b)
    
    # Calculate union area and check if 0
    union = area_a + area_b - area_i
    if union <= 0:
        return 0.0
    
    # Calculate IoU
    iou = area_i / union
    
    # Ensure IoU is between 0 and 1
    if (0 <= iou <= 1):
        return iou
    return 0.0

# Matches detections to existing tracks
def detections_to_tracks(frame_detections, curr_tracks, iou_threshold):
    potential_matches = []
    
    for detection_id, detection in enumerate(frame_detections):
        for track_id, track in enumerate(curr_tracks):
            if track.state != "lost":
                # If IoU between detection and track is above threshold add as candidate
                iou_score = iou(detection["box_xyxy"], track.last_xyxy)
                if iou_score >= iou_threshold:
                    potential_matches.append((detection_id, track_id, iou_score))
    
    # Sort by largest IoU score
    potential_matches.sort(key = lambda match : match[2], reverse=True)
    
    matched_detections = set()
    matched_tracks = set()
    matches = []
    
    # Make matches if both detection and track are available (highest IoU first)
    for detection_id, track_id, iou_score in potential_matches:
        if detection_id not in matched_detections and track_id not in matched_tracks:
            matched_detections.add(detection_id)
            matched_tracks.add(track_id)
            matches.append((detection_id, track_id))
            
    unmatched_detections_ids = [idx for idx in range(len(frame_detections)) if idx not in matched_detections]
    unmatched_tracks_ids = [idx for idx in range(len(curr_tracks)) if idx not in matched_tracks]
    
    return matches, unmatched_detections_ids, unmatched_tracks_ids

def new_track_record(detection, track):
    record = TrackRecord(
        video_id=detection["video_id"],
        frame_index=detection["frame_index"],
        timestamp=detection["timestamp"],
        track_id=track.track_id,
        hits=track.hits,
        misses=track.misses,
        state=track.state,
        confidence=detection["confidence"],
        box_xywh=detection["box_xywh"],
        box_xyxy=detection["box_xyxy"]
    )

    return record

def update_tracks(frame_detections, curr_tracks, matches, unmatched_detections_ids, unmatched_tracks_ids, min_hits, max_misses, next_track_id):
    new_records = []
    
    # Update existing tracks with matched detections
    for detection_id, track_id in matches:
        detection = frame_detections[detection_id]
        track = curr_tracks[track_id]
        
        track.last_xyxy = detection["box_xyxy"]
        track.last_xywh = detection["box_xywh"]
        track.hits += 1
        track.misses = 0
        
        if track.hits >= min_hits:
            track.state = "active"
        else:
            track.state = "tentative"
        new_records.append(new_track_record(detection, track))
        
    # Create new tracks for unmatched detections
    for detection_id in unmatched_detections_ids:
        detection = frame_detections[detection_id]
        track = Track(
            track_id = f"track_{next_track_id}",
            last_xyxy = detection["box_xyxy"],
            last_xywh = detection["box_xywh"],
            hits = 1,
            misses = 0,
            state = "tentative",
        )
        next_track_id += 1
        curr_tracks.append(track)
        new_records.append(new_track_record(detection, track))
    
    # Update unmatched tracks
    for track_id in unmatched_tracks_ids:
        track = curr_tracks[track_id]
        track.misses += 1
        if track.misses > max_misses:
            track.state = "lost"
    
    # Remove lost tracks
    curr_tracks = [track for track in curr_tracks if track.state != "lost"]
    
    return curr_tracks, next_track_id, new_records

def write_tracks_jsonl(track_records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in track_records:
            f.write(json.dumps(asdict(record)) + "\n")
            
def parse_args():
    parser = argparse.ArgumentParser(description="Object Tracking: Frame detections -> Tracks")
    parser.add_argument("--detections", type=Path, required = True, help="Path to the detections JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output tracks JSONL file")
    
    parser.add_argument("--iou", type=float, default=0.3, help="IoU threshold for matching detections to existing tracks (default is 0.3)")
    parser.add_argument("--min_hits", type=int, default=3, help="Minimum hits required to consider a track as active (default is 3)")
    parser.add_argument("--max_misses", type=int, default=5, help="Maximum misses allowed before considering a track as lost (default is 5)")
    parser.add_argument("--stride", type=int, default=None, help="Stride used during inference (default is the value in run_info.json, or 1 if there is no run_info.json)")
    return parser.parse_args()

def run_tracking(detections_path, output_path, iou_threshold=0.3, min_hits=3, max_misses=5, stride=None):
    detections_by_frame = load_detections(detections_path)
    if stride is None:
        stride = load_stride(detections_path)
    curr_tracks = []
    next_track_id = 0
    track_records = []
    last_frame = max(detections_by_frame, default=-1)

    for frame_index in range(stride - 1, last_frame + 1, stride):
        frame_detections = detections_by_frame.get(frame_index, [])
        matches, unmatched_detections_ids, unmatched_tracks_ids = detections_to_tracks(frame_detections, curr_tracks, iou_threshold)
        curr_tracks, next_track_id, new_records = update_tracks(frame_detections, curr_tracks, matches, unmatched_detections_ids, unmatched_tracks_ids, min_hits, max_misses, next_track_id)
        track_records.extend(new_records)
    
    write_tracks_jsonl(track_records, output_path)
    
def main():
    args = parse_args()
    run_tracking(args.detections, args.output, args.iou, args.min_hits, args.max_misses, args.stride)
    
if __name__ == "__main__":
    main()