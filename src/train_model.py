import argparse
from pathlib import Path
import subprocess
import sys
import torch
from ultralytics import YOLO

ROOT_PATH = Path(__file__).resolve().parent.parent
YAML_PATH = ROOT_PATH / "data" / "prepared_yolo" / "dataset.yaml"

MODEL_NAME = "yolo11s.pt"

RUNS_STORE_PATH = ROOT_PATH / "runs"
RUN_NAME = f"badger_{MODEL_NAME.split('.')[0]}"

def parse_args():
    parser = argparse.ArgumentParser(description="Train the YOLO badger model")
    parser.add_argument(
        "--cpu", action="store_true",
        help="Train on the CPU instead of the GPU (much slower)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Checked before the dataset rebuild, which deletes the prepared dataset
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError(
            "train_model.py: No GPU found. Training on the CPU is much "
            "slower, so to use it anyway rerun with --cpu"
        )

    subprocess.run([sys.executable, "-m", "src.yolo_prep.build_dataset"], check=True, cwd=ROOT_PATH)
    if not YAML_PATH.exists():
        raise FileNotFoundError(f"train_model.py: dataset.yaml not found")

    model = YOLO(MODEL_NAME)
    model.train(
        data=str(YAML_PATH),
        project=str(RUNS_STORE_PATH),
        name=str(RUN_NAME),
        epochs=20,
        imgsz=640,
        batch=-1,
        device="cpu" if args.cpu else 0,
        workers=4
    )

    # Test on best weights (chosen by val)
    test_metrics = model.val(
        data=str(YAML_PATH),
        split="test",
        project=str(RUNS_STORE_PATH),
        name=f"{RUN_NAME}_test"
    )
    print(
        f"train_model.py: test mAP50={test_metrics.box.map50:.3f}, "
        f"mAP50-95={test_metrics.box.map:.3f}"
    )
    
if __name__ == "__main__":
    main()