from pathlib import Path
import shutil
import yaml

from src.yolo_prep.prep_positives import (
    load_csv_annotations,
    prepare_split_positives,
    split_by_video
)
from src.yolo_prep.prep_negatives import prepare_split_negatives

YOLO_PREP_PATH = Path(__file__).resolve().parent
SRC_PATH = YOLO_PREP_PATH.parent
ROOT_PATH = SRC_PATH.parent

DATA_PATH = ROOT_PATH / "data"
POSITIVES_DATA_PATH = DATA_PATH / "Badger_Dataset_for_ICANN"
POSITIVES_TRAIN_CSV = POSITIVES_DATA_PATH / "train_labels.csv"
POSITIVES_VAL_CSV = POSITIVES_DATA_PATH / "val_labels.csv"
POSITIVES_TRAIN_IMAGES = POSITIVES_DATA_PATH / "train_images"
POSITIVES_TEST_IMAGES = POSITIVES_DATA_PATH / "test_images"

POSITIVES_TEST_CSV = POSITIVES_DATA_PATH / "test_labels.csv"

NEGATIVES_DATA_PATH = DATA_PATH / "nighttime_driving_dataset"

OUTPUT_PATH = DATA_PATH / "prepared_yolo"

def write_yaml(output_path):
    yaml_data = {
        "path": str(output_path.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {
            0: "badger"
        }
    }
    
    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)

def main():
    if OUTPUT_PATH.exists():
        shutil.rmtree(OUTPUT_PATH)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # The val CSV shares videos with train, so re-split them by whole video
    train_val = (
        load_csv_annotations(POSITIVES_TRAIN_CSV)
        | load_csv_annotations(POSITIVES_VAL_CSV)
    )
    train, val = split_by_video(train_val)
    test = load_csv_annotations(POSITIVES_TEST_CSV)

    prepare_split_positives("train", train, POSITIVES_TRAIN_IMAGES, OUTPUT_PATH)
    prepare_split_positives("val", val, POSITIVES_TRAIN_IMAGES, OUTPUT_PATH)
    prepare_split_positives("test", test, POSITIVES_TEST_IMAGES, OUTPUT_PATH)

    prepare_split_negatives("train", NEGATIVES_DATA_PATH, OUTPUT_PATH, 0.8, 0.1, 0.1)

    write_yaml(OUTPUT_PATH)

    print("build_dataset.py: Images prepared for YOLO")
    
if __name__ == "__main__":
    main()