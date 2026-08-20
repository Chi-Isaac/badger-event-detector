# Badger Event Detector
> **A computer vision pipeline for detecting and tracking badgers in camera footage**

This project takes camera footage as input, which feeds the input frame by frame to a YOLO object detection model trained on badgers in low light settings.
The current pipeline also uses IoU-based object tracking to associate detections across frames, producing both detection and track records which are saved in JSONL format.

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
