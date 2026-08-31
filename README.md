# Badger Event Detector
> **A computer vision pipeline for detecting and tracking badgers in camera footage**

This project takes camera footage as input, which feeds it frame by frame to a YOLO object detection model trained on badgers in low-light settings.
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
│       ├── __init__.py        // YOLO dataset-preparation package
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

Training outputs, including the model weights and metrics, will be saved in the `runs/` folder.

The best-performing and last model weights will be stored in:
```text
runs/badger_yolo11s/weights/best.pt
runs/badger_yolo11s/weights/last.pt
```

---

## Inference and Tracking

### Inference

To run object detection on an input video from the repository root:
```bash
python3 src/inference.py \
    --model <PATH_TO_MODEL_WEIGHTS> \
    --source <PATH_TO_INPUT_VIDEO> \
    --outdir <OUTPUT_DIRECTORY> \
    [--save]
```

| Argument | Description | Required | Default Value |
| --- | --- | --- | --- |
| `--model` | Path to trained YOLO model weights. | Yes | N/A |
| `--source` | Path to the input video file. | Yes | N/A |
| `--outdir` | Path to directory to store inference outputs. | Yes | N/A | 
| `--save` | Flag to save annotated frames. | No | False |

This step runs the YOLO model over the input footage frame by frame and writes detection results to `detections.jsonl`. Run metadata is written to `run_info.json`.

### Tracking

To run object tracking using the detections generated during inference from the repository root:
```bash
python3 src/track.py \
    --detections <PATH_TO_DETECTIONS> \
    --output <PATH_TO_TRACKS> \
    [--iou <IOU_THRESHOLD>] \
    [--min_hits <MIN_HITS>] \
    [--max_misses <MAX_MISSES>]
```

| Argument | Description | Required | Default Value |
| --- | --- | --- | --- |
| `--detections` | Path to detection records generated by the inference step | Yes | N/A |
| `--output` | Path where tracking results will be written. | Yes | N/A |
| `--iou` | IoU threshold used to match detections to existing tracks (decimal from 0.0 to 1.0). | No | 0.3 |
| `--min_hits` | Minimum number of hits/matches before a tentative track becomes active. | No | 3 |
| `--max_misses` | Number of consecutive unmatched frames before an active track becomes lost. | No | 5 |

This step reads `detections.jsonl` and associates detections across frames using IoU-based tracking. Resulting object tracks are written to `tracks.jsonl`.

---

## Output Format

### Inference Outputs

Running `src/inference.py` produces:
- `detections.jsonl` - Frame-by-frame detection records
- `run_info.json` - Metadata describing the inference run
   
Each line in `detections.jsonl` is a separate JSON object.
For example:
```json
{"video_id": "EK000058", "frame_index": 0, "timestamp": 0.0, "image_width": 1280, "image_height": 720, "category_name": "badger", "category_id": 0, "confidence": 0.8888955116271973, "box_xywh": [367.72979736328125, 402.4609069824219, 284.1888427734375, 268.35565185546875], "box_xyxy": [225.63539123535156, 268.2830810546875, 509.82421875, 536.6387329101562], "model_name": "best"}
```

Detection records include:
- Video identifier and frame number
- Timestamp in seconds
- Image dimensions
- Confidence score
- Bounding box coordinates (both xywh and xyxy formats)
- Name of model used

The `run_info.json` file stores metadata including:
- Video identifier
- Name of model used and path to weights
- Source video path
- Output directory
- Video metadata (FPS, frame count, width, height)
- Inference parameters (confidence threshold, IoU threshold, image size, device, stride, save settings)
- Frames seen and number of detections

### Tracking Outputs

Running `src/track.py` produces:
- `tracks.jsonl` - Track records generated from detections.jsonl

Each line in `tracks.jsonl` is a separate JSON object.
For example:
```json
{"video_id": "EK000058", "frame_index": 0, "timestamp": 0.0, "track_id": "track_0", "hits": 1, "misses": 0, "state": "tentative", "confidence": 0.8888955116271973, "box_xywh": [367.72979736328125, 402.4609069824219, 284.1888427734375, 268.35565185546875], "box_xyxy": [225.63539123535156, 268.2830810546875, 509.82421875, 536.6387329101562]}
```

Tracking records include:
- Video identifier and frame number
- Timestamp in seconds
- Assigned track identifier
- Hit and miss counters
- Current track state
- Confidence score
- Bounding box coordinates (both xywh and xyxy formats)

#### Track States
The `state` field in each JSON object describes the current stage of a track:
1) `tentative` - Track is newly created and has not received enough matching detections to be confirmed.
2) `active` - Once enough hits/matches have been made, the track is upgraded to this state.
3) `lost` - Once enough frames have passed with no matching detection, the state is updated as `lost`.

---

## Visualisation

This step takes a source video and the `tracks.jsonl` file from the tracking stage to write a new video with bounding boxes.

To run from the repository root:
```bash
python3 src/visualise_tracks.py \
    --input <PATH_TO_INPUT_VIDEO> \
    --tracks <PATH_TO_TRACKS> \
    --output <PATH_TO_OUTPUT>
```

| Argument | Description | Required |
| --- | --- | --- |
| `--input` | Path to the original source video | Yes |
| `--tracks` | Path to `tracks.jsonl` file produced by `src/track.py` | Yes |
| `--output` | Path where the output will be written | Yes |

The output video will:
- Have the same resolution and FPS as the input video.
- Show a coloured rectangle around each detection.
- Have the same colour for each track.
- Displays a label for each track (track_id, state, confidence).
- Displays the frame index.

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

## TODO / Future Improvements
- [ ] **Improve tracking consistency**: Track fragmentation can occur when badgers are not detected for extended periods due to use of IoU-based matching.
- [ ] **Optimise inference speed**: Profile and improve FPS for real-time performance.
- [ ] **Add MOT evaluation metrics**: For example MOTA, IDF1, and HOTA could be used to quantify tracking quality.
