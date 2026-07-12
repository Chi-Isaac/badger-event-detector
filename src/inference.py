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
