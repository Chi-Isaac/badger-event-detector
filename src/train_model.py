from pathlib import Path
import subprocess
import sys
from ultralytics import YOLO

ROOT_PATH = Path(__file__).resolve().parent.parent
YAML_PATH = ROOT_PATH / "data" / "prepared_yolo" / "dataset.yaml"

MODEL_NAME = "yolo11s.pt"

RUNS_STORE_PATH = ROOT_PATH / "runs"
RUN_NAME = f"badger_{MODEL_NAME.split('.')[0]}"

def main():
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
        device=0,
        workers=4
    )
    
if __name__ == "__main__":
    main()