import argparse
import cv2
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
    parser.add_argument("--conf", type=float, default=0.3) # Confidence threshold for detections
    parser.add_argument("--iou", type=float, default=0.7) # Intersection over Union threshold (IoU = Area of Overlap / Area of Union)
    parser.add_argument("--imgsz", type=int, default=640) # Inference image size (pixels)
    parser.add_argument("--device", type=str, default="0") # Device to run inference on ("cpu" or "0" for GPU)
    parser.add_argument("--stride", type=int, default=1) # Stride for fram sampling
    parser.add_argument("--save", action="save_true") # Save inference video with bounding boxes drawn
    
    return parser.parse_args()

def load_metadata(video_path):
    if not video_path.exists():
        raise FileNotFoundError(f"inference.py: {video_path} does not exist")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Following three return floats, so cast to int
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height
    }