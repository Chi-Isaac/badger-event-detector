import random
from pathlib import Path
from shutil import copyfile

# Splits images into ratios specified for training, validation, and testing
def split_image_paths(images_path, train_ratio, val_ratio, test_ratio):
    image_paths = [
        path for path in images_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".jpg"
    ]
    
    random.shuffle(image_paths)
    
    num_imgs = len(image_paths)
    train_end = int(num_imgs * train_ratio)
    val_end = train_end + int(num_imgs * val_ratio)

    return {
        "train": image_paths[:train_end],
        "val": image_paths[train_end:val_end],
        "test": image_paths[val_end:]
    }

def copy_negative_images(split_paths, output_path):
    for split_name, image_paths in split_paths.items():
        output_imgs_dir = output_path / split_name / "images"
        output_imgs_dir.mkdir(parents=True, exist_ok = True)
        
        output_labels_dir = output_path / split_name / "labels"
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        for inp_img_path in image_paths:
            out_img_path = output_imgs_dir / inp_img_path.name
            copyfile(inp_img_path, out_img_path)
            
            label_path = output_labels_dir / (inp_img_path.stem + ".txt")
            label_path.touch()  # Creates empty label file

def prepare_split_negatives(split_name, images_path, output_path, train_ratio, val_ratio, test_ratio):
    split_paths = split_image_paths(images_path, train_ratio, val_ratio, test_ratio)
    copy_negative_images(split_paths, output_path)