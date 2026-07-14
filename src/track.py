import json
import argparse
from pathlib import Path

class Track:
    track_id: str
    last_xyxy: list[float]
    last_xywh: list[float]
    hits: int
    misses: int
    state: str
    
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

# Takes jsonl with all detections and returns list of detections by frame
def load_detections(path):
    detections_by_frame = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["frame_index"] >= len(detections_by_frame):
                detections_by_frame.extend([[] for _ in range(record["frame_index"] - len(detections_by_frame) + 1)])
            detections_by_frame[record["frame_index"]].append(record)
    
    return detections_by_frame

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
    potential_matches.sort(reverse=True)
    
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
    return TrackRecord(
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
        track = Track()
        track.track_id = f"track_{next_track_id}"
        next_track_id += 1
        track.last_xyxy = detection["box_xyxy"]
        track.last_xywh = detection["box_xywh"]
        track.hits = 1
        track.misses = 0
        track.state = "tentative"
        curr_tracks.append(track)
        new_records.append(new_track_record(detection, track))
    
    # Update unmatched tracks
    for track_id in unmatched_tracks_ids:
        track = curr_tracks[track_id]
        track.misses += 1
        if track.misses > max_misses:
            track.state = "lost"
            
    return curr_tracks, next_track_id, new_records

def write_tracks_jsonl(tracks, path):
    with open(path, "w", encoding="utf-8") as f:
        for track in tracks:
            f.write(json.dumps(track.__dict__) + "\n")
            
def parse_args():
    parser = argparse.ArgumentParser(description="Object Tracking: Frame detections -> Tracks")
    parser.add_argument("--detections", type=Path, required = True, help="Path to the detections JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Path to the output tracks JSONL file")
    
    parser.add_argument("--iou", type=float, default=0.3)
    parser.add_argument("--min_hits", type=int, default=3) # Tentative -> Active track threshold
    parser.add_argument("--max_misses", type=int, default=5) # Active -> Lost track threshold
    return parser.parse_args()

def run_tracking(detections_path, output_path, iou_threhold=0.3, min_hits=3, max_misses=5):
    detections_by_frame = load_detections(detections_path)
    curr_tracks = []
    next_track_id = 0
    
    for frame_index, frame_detections in enumerate(detections_by_frame):
        matches, unmatched_detections_ids, unmatched_tracks_ids = detections_to_tracks(frame_detections, curr_tracks, iou_threhold)
        curr_tracks, next_track_id = update_tracks(frame_detections, curr_tracks, matches, unmatched_detections_ids, unmatched_tracks_ids, min_hits, max_misses, next_track_id)
    
    write_tracks_jsonl(curr_tracks, output_path)
    
def main():
    args = parse_args()
    run_tracking(args.detections, args.output, args.iou, args.min_hits, args.max_misses)
    
if __name__ == "__main__":
    main()