"""
Baseline Model Training Script
YOLOv8 Segmentation Model Training

Usage:
    python train.py
"""

from ultralytics import YOLO
import os

# Load model
model = YOLO("yolov8m-seg.pt")  # Model will auto-download if not present

# Train model
results = model.train(
    data="baseline_data.yaml",    # Dataset configuration file
    epochs=300,          # Number of training epochs
    imgsz=1280,          # Input image size
    batch=6,             # Batch size (reduce if OOM)
    device=0,            # GPU device (0, 1, 2, 3, or 'cpu')
    task='segment'       # Segmentation task
)