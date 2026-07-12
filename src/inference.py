import argparse
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}

class DetectionRecord:
    video_id: str
    frame_index: int
    timestamp: float
    image_width: int
    image_height: int
    category_name: str
    category_id: int
    confidence: float
    box_xywh: list[float]
    box_xyxy: list[float]
    model_name: str

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO inference: video -> JSONL")
    parser.add_argument("--model", type=Path, required=True, help="Path to model weights (.pt file)")
    parser.add_argument("--source", type=Path, required=True, help="Path to input video")
    parser.add_argument("--outdir", type=Path, required=True, help="Path to output directory")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--save", action="save_true")
    
    return parser.parse_args()