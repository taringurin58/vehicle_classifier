# YOLOv8 Vehicle Detector & Cropper

This project detects vehicles in videos using YOLOv8 and crops individual vehicles to create a dataset for further classification (e.g., CNN-based vehicle type classifier).

## Features

- Detect vehicles: car, truck, bus, motorcycle
- Crop and save unique vehicle images
- Reduce duplicate images with object tracking (Norfair tracker)
- Prepare dataset for downstream classification tasks

## Project Structure
```
yolo8_vehicle_classifier/
├── training/ # YOLO dataset & training scripts
│ ├── datasets/ # Cropped images / YOLO labels
│ │ ├── images/
│ │ │ ├── train/
│ │ │ └── val/
│ │ └── labels/
│ │ ├── train/
│ │ └── val/
│ ├── dataset.yaml # Dataset config for YOLO
│ └── extract_unique_vehicles.py # Script to crop unique vehicles from video
├── testing/
│ └── videos/ # Test videos
├── runs/ # YOLO training outputs
├── yolov8n.pt # Pre-trained YOLOv8 model
├── yolo_env/ # Python virtual environment
├── requirements.txt # Python dependencies
└── README.md
```

## Create and activate a virtual environment (recommended):
```
python3 -m venv yolo_env
source yolo_env/bin/activate
```

## Install dependencies:
```
pip install --upgrade pip
pip install -r requirements.txt
```

# Usage
## Crop vehicles from video:
```
python3 training/extract_unique_vehicles.py
```

## Train YOLOv8 on your dataset (optional):
```
yolo detect train data=training/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640 project=./runs name=vehicle_detect
```
