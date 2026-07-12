import csv
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

# Creates annotations and copies images into new folders
def prepare_split_positives(split_name, csv_path, images_path, output_path):
    annotations = load_csv_annotations(csv_path)
    
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