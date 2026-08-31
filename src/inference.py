import argparse
import cv2
import json
from pathlib import Path
from ultralytics import YOLO
from dataclasses import dataclass, asdict

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}

@dataclass
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
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for detections (default is 0.3)")
    parser.add_argument("--iou", type=float, default=0.7, help="Intersection over Union threshold (IoU = Area of Overlap / Area of Union) (default is 0.7)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size (pixels) (default is 640)")
    parser.add_argument("--device", type=str, default="0", help="Device to run inference on ('cpu' or '0' for GPU) (default is '0')")
    parser.add_argument("--stride", type=int, default=1, help="Stride for frame sampling (default is 1)")
    parser.add_argument("--save", action="store_true", help="Save individual frames with bounding boxes drawn")
    
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

# Prepares output directory for results
def prepare_output_dir(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "detections": output_dir / "detections.jsonl",
        "annotated_dir": output_dir / "annotated",
        "run_info": output_dir / "run_info.json"
    }
    paths["annotated_dir"].mkdir(parents=True, exist_ok=True)
    return paths

def run_inference():
    args = parse_args()
    
    if args.source.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"inference.py: {args.source} is not a supported video format. Supported formats: {VIDEO_EXTENSIONS}")
    
    paths = prepare_output_dir(args.outdir)
    metadata = load_metadata(args.source)
    
    model = YOLO(str(args.model))
    
    results = model.predict(
        source=str(args.source),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        vid_stride=args.stride,
        save=False,
        stream=True,
        verbose=True
    )
    
    video_id = args.source.stem
    model_name = args.model.stem
    
    frames_seen = 0
    detections = 0
    
    with open(paths["detections"], "w") as f:
        for frame_index, result in enumerate(results):
            frames_seen += 1
            timestamp = frame_index / metadata["fps"]
            image_width = metadata["width"]
            image_height = metadata["height"]
            
            # For each detection, DetectionRecord made for JSONL file
            for box in result.boxes:
                detections += 1
                record = DetectionRecord(
                    video_id=video_id,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    image_width=image_width,
                    image_height=image_height,
                    category_name=result.names[int(box.cls)],
                    category_id=int(box.cls),
                    confidence=float(box.conf),
                    box_xywh=[float(i) for i in box.xywh[0].tolist()],
                    box_xyxy=[float(i) for i in box.xyxy[0].tolist()],
                    model_name=model_name
                )
                
                f.write(json.dumps(asdict(record)) + "\n")
                
            # Saves annotated frame with bounding box if --save flag set
            if args.save:
                annotated = result.plot()
                output_path = paths["annotated_dir"] / f"{video_id}_frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(output_path), annotated)
    
    run_info = {
        "video_id": video_id,
        "model_name": model_name,
        "weights": str(args.model),
        "source": str(args.source),
        "output_dir": str(args.outdir),
        "video_metadata": metadata,
        "inference_parameters": {
            "conf": args.conf,
            "iou": args.iou,
            "imgsz": args.imgsz,
            "device": args.device,
            "stride": args.stride,
            "save": args.save
        },
        "results_summary": {
            "frames_seen": frames_seen,
            "detections": detections,
        }
    }
    paths["run_info"].write_text(json.dumps(run_info, indent=4))

if __name__ == "__main__":
    run_inference()