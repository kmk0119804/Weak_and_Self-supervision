"""
Baseline Model Inference Script
Validate trained YOLOv8 model

Usage:
    python inference.py
"""

from ultralytics import YOLO
import os

# Load trained model
model_path = 'runs/segment/train/weights/best.pt'  # Path to trained weights
model = YOLO(model_path)

# Run validation
results = model.val(
    data='data.yaml',    # Dataset configuration file
    imgsz=1280,          # Input image size
    conf=0.25,           # Confidence threshold
    iou=0.5,             # IoU threshold for NMS
    save_json=True       # Save results to JSON
)
