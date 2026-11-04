import json

from src.track import Track, detections_to_tracks, run_tracking

BOX = [0.0, 0.0, 100.0, 100.0]

def make_detection(frame_index, box=BOX):
    return {
        "video_id": "video",
        "frame_index": frame_index,
        "timestamp": 0.0,
        "confidence": 0.9,
        "box_xywh": [50.0, 50.0, 100.0, 100.0],
        "box_xyxy": box
    }

# Writes detections for given frames, runs tracker, returns records
def track_frames(tmp_path, frames, **tracking_args):
    detections_path = tmp_path / "detections.jsonl"
    output_path = tmp_path / "tracks.jsonl"
    lines = [json.dumps(make_detection(frame)) for frame in frames]
    detections_path.write_text("\n".join(lines) + "\n")

    run_tracking(detections_path, output_path, **tracking_args)

    return [json.loads(line) for line in output_path.read_text().splitlines()]

# Check that a track can be created and keep its ID across multiple frames
def test_still_object_keeps_one_track(tmp_path):
    records = track_frames(tmp_path, range(5))

    assert [record["track_id"] for record in records] == ["track_0"] * 5

# Check that candidates are sorted by IoU
def test_track_matches_detection_with_highest_iou():
    track = Track(
        track_id="track_0",
        last_xyxy=BOX,
        last_xywh=[50.0, 50.0, 100.0, 100.0],
        hits=1,
        misses=0,
        state="tentative"
    )
    detections = [
        make_detection(0, box=BOX), # IoU 1.0
        make_detection(0, box=[50.0, 0.0, 150.0, 100.0]) # IoU 0.33
    ]

    matches, _, _ = detections_to_tracks(detections, [track], 0.3)

    assert matches == [(0, 0)]

# Ensure frames skipped by strides are not counted as misses (would lose track)
def test_stride_does_not_count_skipped_frames_as_misses(tmp_path):
    # Stride 3 samples frames 2, 5, 8, 11, so 5 and 8 are the only real misses
    records = track_frames(tmp_path, [2, 11], max_misses=5, stride=3)

    assert [record["track_id"] for record in records] == ["track_0", "track_0"]
