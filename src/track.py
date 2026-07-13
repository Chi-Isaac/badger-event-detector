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