import json

class Track:
    track_id: str
    last_xyxy: list[float]
    last_xywh: list[float]
    hits: int
    misses: int
    state: str

# Takes jsonl with all detections and returns list of detections by frame
def load_detections(path):
    detections_by_frame = []
    
    with open(path, "r") as f:
        for line in f:
            record = json.loads(line)
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
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    
    # Calculate IoU
    iou = area_i / (area_a + area_b - area_i)
    
    # Ensure IoU is between 0 and 1
    if (0 <= iou <= 1):
        return iou
    else:
        return 0.0

# Matches detections to existing tracks
def detections_to_tracks(frame_detections, curr_tracks, iou_threhold):
    potential_matches = []
    
    for detection_id, detection in enumerate(frame_detections):
        for track_id, track in enumerate(curr_tracks):
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
    for detection_id, track_id, iou_score in potential matches:
        if detection_id not in matched_detections and track_id not in matched_tracks:
            matched_detections.add(detection_id)
            matched_tracks.add(track_id)
            matches.append((detection_id, track_id))
            
    unmatched_detections_ids = [idx for idx in range(len(frame_detections)) if idx not in matched_detections]
    unmatched_tracks_ids = [idx for idx in range(len(curr_tracks)) if idx not in matched_tracks]
    
    return matches, unmatched_detections_ids, unmatched_tracks_ids