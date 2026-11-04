import csv
import random
import re
from collections import defaultdict
from pathlib import Path
from shutil import copyfile

BADGER_CLASSES = {"badger_esp", "badger_iaco", "badger_looi", "badger_strik"}

# Yolo requires box centres and dimensions normalised to [0, 1] range
def coords_to_yolo(xmin, ymin, xmax, ymax, width, height):
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height

# Maps images to annotations
def load_csv_annotations(csv_path):
    annotations = defaultdict(list)
    
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            width = int(row["width"])
            height = int(row["height"])
            class_name = row["class"].strip().lower()
            
            xmin = int(row["xmin"])
            ymin = int(row["ymin"])
            xmax = int(row["xmax"])
            ymax = int(row["ymax"])
            
            annotations[filename].append({
                "class_name": class_name,
                "width": width,
                "height": height,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            })

    return annotations

# Filenames are <video>_image_<n>.jpg, with some using a double underscore
def video_id(filename):
    match = re.match(r"(.+?)_+image_\d+\.jpg$", filename)
    if match is None:
        raise ValueError(f"yolo_prep.py: Cannot parse video id from {filename}")
    return match.group(1)

# Holds out whole videos per site, as adjacent frames are near-identical
def split_by_video(annotations, val_fraction=0.2, seed=0):
    frames_by_video = defaultdict(list)
    for filename in annotations:
        frames_by_video[video_id(filename)].append(filename)

    videos_by_site = defaultdict(list)
    for video in sorted(frames_by_video):
        videos_by_site[video.split("_")[0]].append(video)

    rng = random.Random(seed)
    val_videos = set()
    for site, videos in sorted(videos_by_site.items()):
        rng.shuffle(videos)
        target = val_fraction * sum(len(frames_by_video[v]) for v in videos)
        held_out = 0
        # Last video is never held out so each site stays in train
        for video in videos[:-1]:
            n_frames = len(frames_by_video[video])
            closer = abs(held_out + n_frames - target) < abs(held_out - target)
            if held_out == 0 or closer:
                val_videos.add(video)
                held_out += n_frames

    train = {}
    val = {}
    for filename, boxes in annotations.items():
        split = val if video_id(filename) in val_videos else train
        split[filename] = boxes
    return train, val

# Creates annotations and copies images into new folders
def prepare_split_positives(split_name, annotations, images_path, output_path):
    output_imgs_dir = output_path / split_name / "images"
    output_imgs_dir.mkdir(parents=True, exist_ok=True)
    
    output_labels_dir = output_path / split_name / "labels"
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, boxes in annotations.items():
        inp_img_path = images_path / filename
        if not inp_img_path.exists():
            raise FileNotFoundError(f"yolo_prep.py: {inp_img_path} not found")
        
        out_img_path = output_imgs_dir / filename
        copyfile(inp_img_path, out_img_path)
        
        label_path = output_labels_dir / f"{Path(filename).stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            for box in boxes:
                if box["class_name"] not in BADGER_CLASSES:
                    raise ValueError(f"yolo_prep.py: Unknown badger class {box['class_name']}")
            
                class_id = 0
                x_center, y_center, box_width, box_height = coords_to_yolo(
                    box["xmin"], box["ymin"], box["xmax"], box["ymax"],
                    box["width"], box["height"]
                )
                
                f.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} "
                    f"{box_width:.6f} {box_height:.6f}\n"
                )