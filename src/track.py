class Track:
    track_id: str
    last_xyxy: list[float]
    last_xywh: list[float]
    hits: int
    misses: int
    state: str
