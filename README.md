# Badger Event Detector
> **A computer vision pipeline for detecting and tracking badgers in camera footage**

This project takes camera footage as input, which feeds the input frame by frame to a YOLO object detection model trained on badgers in low light settings.
The current pipeline also uses IoU-based object tracking to associate detections across frames, producing both detection and track records which are saved in JSONL format.

---

## Project Structure
```text
├── README.md                  // Project documentation
├── requirements.txt           // Project dependencies
├── .gitignore                 // Files excluded from version control
├── src/
│   ├── inference.py           // Runs YOLO inference over input footage
│   ├── track.py               // Associates detections across frames
│   ├── train_model.py         // Trains the YOLO detection model
│   └── yolo_prep/
│       ├── __init__.py        // YOLO dataset-preperation package
│       ├── build_dataset.py   // Builds prepared YOLO dataset
│       ├── prep_negatives.py  // Prepares negatives training examples
│       └── prep_positives.py  // Prepares positive training examples
├── data/                      // Dataset and wildlife footage
└── runs/                      // Training and inference outputs
```

---

## Installation
1) Clone the repository:
```bash
git clone https://github.com/Chi-Isaac/badger-event-detector.git
cd badger-event-detector
```

2) Create a Python virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3) Install dependencies:
```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

For GPU support, install the appropriate PyTorch build for your system.

---

## Dataset Preparation

This project uses two datasets:
- [Badger Dataset for ICANN](https://osf.io/d6vrf/overview)
- [Nighttime Driving Images Dataset](https://www.kaggle.com/datasets/adembakrc/nighttime-driving)

The raw data is stored in the data folder:

```text
data/
├── Badger_Dataset_for_ICANN/
└── nighttime-driving/
```

The scripts in `src/yolo_prep/` are used to prepare the data for YOLO training:
- `prep_positives.py` prepares positive images (images containing badgers).
- `prep_negatives.py` prepares negative images (images not including badgers).
- `build_dataset.py` creates the final YOLO dataset.

The prepared YOLO dataset is written to `data/prepared_yolo/`. It stores the dataset using images as well as YOLO-format `.txt` annotation files for labelled badger images, organised into separate folders for training, validation, and test splits. 

From the repository root, prepare the dataset with:
```bash
python3 -m src.yolo_prep.build_dataset
```

The data folder following the dataset preparation:

```text
data/
├── Badger_Dataset_for_ICANN/
├── nighttime-driving/
└── prepared_yolo/
    ├── test/
    │   ├── images/
    │   |   ├── ESP_EK000058_image_00001.jpg
    │   |   ├── ...
    │   |   └── STRIK_EK000016_image_00062.jpg
    │   ├── labels/
    │   |   ├── ESP_EK000058_image_00001.txt
    │   |   ├── ...
    │   |   └── STRIK_EK000016_image_00062.txt
    ├── train/
    │   ├── images/
    │   |   ├── ESP_EK000027__image_00003.jpg
    │   |   ├── ...
    │   |   └── STRIK_EK000019_image_00062.jpg
    │   ├── labels/
    │   |   ├── ESP_EK000027__image_00003.txt
    │   |   ├── ...
    │   |   └── STRIK_EK000019_image_00062.txt
    ├── val/
    │   ├── images/
    │   |   ├── ESP_EK000027__image_00001.jpg
    │   |   ├── ...
    │   |   └── STRIK_EK000019_image_00058.jpg
    │   └── labels/
    │       ├── ESP_EK000027__image_00001.txt
    │       ├── ...
    │       └── STRIK_EK000019_image_00058.txt
    └── dataset.yaml 
```

---

## Training
Please ensure the dataset is properly prepared before attempting to train the model.

To train the YOLO model from the repository root:
```bash
python3 src/train_model.py
```

Training outputs, including the model weights and metrics will be saved in the `runs/` folder.

The best-performing and last model weights will be stored in:
```text
runs/badger_yolo11s/weights/best.pt
runs/badger_yolo11s/weights/last.pt
```

---

## Acknowledgments
This project uses the following datasets:
- [Badger Dataset for ICANN](https://osf.io/d6vrf/overview), licensed under CC BY 4.0.
- [Nighttime Driving Images Dataset](https://www.kaggle.com/datasets/adembakrc/nighttime-driving), licensed under the MIT License.

Thanks to the dataset creators and maintainers, and to the developers of:
- [NumPy](https://numpy.org) for numerical operations.
- [OpenCV](https://opencv.org) for image and video processing.
- [PyYAML](https://pyyaml.org) for reading dataset configuration files.
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO training and inference.
