# Self-Training with Noisy Student

Train YOLOv8 segmentation model using Noisy Student self-training with pseudo labels and strong data augmentation.

---

## 📋 Overview

This module implements **Noisy Student self-training** using:
- **Weak supervision model** predictions as pseudo labels (teacher)
- **Target domain images** with pseudo labels
- **Small portion of source domain** with ground truth labels
- **Strong data augmentation** for student model (noise injection)

---

## 📁 Files

```
self-supervision/
├── self_inference.py      # Step 1 & 3: Generate predictions / Evaluate model
├── self_yolo_to_yolo.py   # Step 1: Remove confidence from predictions
├── self_train.py          # Step 2: Train student model
├── self_data.yaml         # Dataset configuration
├── self_config.yaml       # Training hyperparameters (optional)
└── README.md              # This file
```

---

## 📊 Training Data Composition

```
Training Set:
├── Target domain + Pseudo labels (from weak model)
└── Source domain + Ground truth labels

Validation Set:
├── Target domain validation
└── Source domain validation

Test Set:
└── Target domain (for final evaluation)
```

---

## 🚀 3-Step Self-Training Process

### Step 1: Generate Pseudo Labels with Weak Model

**Purpose:** Use weak supervision model to generate pseudo labels on test set

#### 1.1 Rename Folders Temporarily

```bash
# Backup original validation set
mv dataset/images/val dataset/images/val_origin
mv dataset/labels/val dataset/labels/val_origin

# Use test set as validation for inference
mv dataset/images/test dataset/images/val
mv dataset/labels/test dataset/labels/val  # Can be empty or dummy
```

#### 1.2 Generate Predictions

```bash
python self_inference.py
```

**Edit `self_inference.py` before running:**

```python
from ultralytics import YOLO

# Load trained WEAK SUPERVISION model (not self-training model!)
model = YOLO('path/to/weak/supervision/best.pt')  # ← CHANGE THIS PATH!

# Generate predictions on test set (temporarily named val)
results = model.predict(
    source='dataset/images/val',    # ← CHANGE THIS PATH!
    save=True,
    save_txt=True,      # Save as YOLO format
    save_conf=True,     # Include confidence scores
    conf=0.25,
    iou=0.5,
    imgsz=1280
)
```

**Output:**
```
runs/segment/predict/
└── labels/
    ├── img001.txt    # Format: class x1 y1 x2 y2 ... xn yn conf
    └── ...
```

#### 1.3 Remove Confidence Scores

**Edit paths in `self_yolo_to_yolo.py`:**

```python
from pathlib import Path

# INPUT: Predictions with confidence (from step 1.2)
LABEL_DIR = Path("runs/segment/predict/labels")  # ← CHANGE THIS!

# OUTPUT: Labels without confidence
OUT_DIR = Path("dataset/labels/pseudo_labels")   # ← CHANGE THIS!

# The script removes last value (confidence) from each line
```

**Run the script:**

```bash
python self_yolo_to_yolo.py
```

**Output:**
```
dataset/labels/pseudo_labels/
├── img001.txt    # Format: class x1 y1 x2 y2 ... xn yn (no conf)
└── ...
```

#### 1.4 Restore Original Folders

```bash
# Restore test set
mv dataset/images/val dataset/images/test
mv dataset/labels/val dataset/labels/test

# Restore original validation set
mv dataset/images/val_origin dataset/images/val
mv dataset/labels/val_origin dataset/labels/val
```

---

### Step 2: Train Student Model

**Purpose:** Train student model with pseudo labels + source labels + strong augmentation

#### 2.1 Prepare Training Data

**Organize dataset with pseudo labels:**

```
dataset/
├── images/
│   ├── train/
│   │   ├── [target domain images]    
│   │   └── [source domain images]     
│   └── val/
│       ├── [target domain val]
│       └── [source domain val]
└── labels/
    ├── train/
    │   ├── [pseudo labels from step 1] # For target domain
    │   └── [ground truth labels]       # For source domain
    └── val/
        └── [validation labels]
```

#### 2.2 Configure Dataset

**Edit `self_data.yaml`:**

```yaml
# Paths (use absolute paths)
train: /absolute/path/to/dataset/images/train  # ← CHANGE THIS!
val: /absolute/path/to/dataset/images/val      # ← CHANGE THIS!

# Class definitions
names:
  0: worker
  1: hardhat
  2: strap
  3: hook

nc: 4
```

#### 2.3 Train Student Model

**Edit `self_train.py`:**

```python
from ultralytics import YOLO

# Load weak supervision model as pretrained weights
model = YOLO('path/to/weak/supervision/best.pt')  # ← CHANGE THIS!

# Train with stronger augmentation (Noisy Student)
results = model.train(
    data="self_data.yaml",
    epochs=300,
    imgsz=1280,
    batch=6,
    device=0,
    task='segment',
    
    # Adjust these hyperparameters for stronger augmentation
    # Increase values compared to weak supervision training
    hsv_h=...,      # Hue augmentation
    hsv_s=...,      # Saturation augmentation
    hsv_v=...,      # Value augmentation
    degrees=...,    # Rotation
    translate=...,  # Translation
    scale=...,      # Scaling
    fliplr=...,     # Horizontal flip
    mosaic=...,     # Mosaic augmentation
    mixup=...       # Mixup augmentation
)
```

**Run training:**

```bash
python self_train.py
```

**Output:**
```
runs/segment/train_self/
├── weights/
│   ├── best.pt     ⭐ Self-trained model
│   └── last.pt
├── results.png
└── confusion_matrix.png
```

---

### Step 3: Evaluate Student Model

**Purpose:** Evaluate final self-trained model on test set

#### 3.1 Rename Folders Temporarily (Again)

```bash
# Same as Step 1.1
mv dataset/images/val dataset/images/val_origin
mv dataset/labels/val dataset/labels/val_origin
mv dataset/images/test dataset/images/val
mv dataset/labels/test dataset/labels/val
```

#### 3.2 Run Evaluation

**Edit `self_inference.py`:**

```python
from ultralytics import YOLO

# Load trained SELF-TRAINING model (not weak model!)
model = YOLO('runs/segment/train_self/weights/best.pt')  # ← CHANGE THIS!

# Evaluate on test set (temporarily named val)
results = model.val(
    data='self_data.yaml',
    imgsz=1280,
    conf=0.25,
    iou=0.5,
    save_json=True
)
```

**Run evaluation:**

```bash
python self_inference.py
```

**Evaluation metrics:**
- mAP@50, mAP@50-95
- Precision, Recall per class
- Confusion matrix

#### 3.3 Restore Original Folders

```bash
# Restore folders
mv dataset/images/val dataset/images/test
mv dataset/labels/val dataset/labels/test
mv dataset/images/val_origin dataset/images/val
mv dataset/labels/val_origin dataset/labels/val
```

---

## 🔄 Complete Workflow Summary

```bash
# ============================================
# Step 1: Generate Pseudo Labels
# ============================================

# 1.1 Rename folders
mv dataset/images/val dataset/images/val_origin
mv dataset/labels/val dataset/labels/val_origin
mv dataset/images/test dataset/images/val

# 1.2 Generate predictions (edit paths in self_inference.py first!)
python self_inference.py

# 1.3 Remove confidence (edit paths in self_yolo_to_yolo.py first!)
python self_yolo_to_yolo.py

# 1.4 Restore folders
mv dataset/images/val dataset/images/test
mv dataset/images/val_origin dataset/images/val
mv dataset/labels/val_origin dataset/labels/val

# ============================================
# Step 2: Train Student Model
# ============================================

# 2.1 Prepare training data with pseudo labels + source labels
# 2.2 Edit paths in self_data.yaml
# 2.3 Edit paths and augmentation in self_train.py
# 2.4 Train model
python self_train.py

# ============================================
# Step 3: Evaluate Student Model
# ============================================

# 3.1 Rename folders
mv dataset/images/val dataset/images/val_origin
mv dataset/labels/val dataset/labels/val_origin
mv dataset/images/test dataset/images/val

# 3.2 Edit paths in self_inference.py
# 3.3 Evaluate
python self_inference.py

# 3.4 Restore folders
mv dataset/images/val dataset/images/test
mv dataset/images/val_origin dataset/images/val
mv dataset/labels/val_origin dataset/labels/val
```

---

## ⚙️ Path Configuration Checklist

**Before running, update these paths:**

### Step 1: Generate Pseudo Labels
- [ ] `self_inference.py`: weak model path
- [ ] `self_inference.py`: dataset image path
- [ ] `self_yolo_to_yolo.py`: LABEL_DIR (input)
- [ ] `self_yolo_to_yolo.py`: OUT_DIR (output)

### Step 2: Train Model
- [ ] `self_data.yaml`: train path
- [ ] `self_data.yaml`: val path
- [ ] `self_train.py`: weak model pretrained weights path
- [ ] `self_train.py`: augmentation hyperparameters

### Step 3: Evaluate Model
- [ ] `self_inference.py`: self-trained model path
- [ ] `self_inference.py`: dataset path

---

## 📊 Expected Performance

**Improvement over weak supervision model:**
- mAP@50: +3-8% improvement
- mAP@50-95: +2-5% improvement

**Performance depends on:**
- Weak model quality (teacher quality)
- Pseudo label accuracy
- Augmentation strength
- Source data portion size

---

## 💡 Noisy Student Key Concepts

### 1. Teacher-Student Framework
- **Teacher**: Weak supervision model (less augmentation)
- **Student**: Self-training model (more augmentation)

### 2. Data Augmentation
- Student uses **stronger augmentation** than teacher
- Adjust hyperparameters in `self_train.py` or `self_config.yaml`
- Balance: strong enough for noise, but not too extreme

### 3. Pseudo Labels + Real Labels
- Pseudo labels from teacher model (noisy but abundant)
- Real labels from source domain (clean but limited)
- Combined training improves robustness

---

## 🐛 Troubleshooting

### Low Pseudo Label Quality

**Solutions:**
1. Increase confidence threshold in Step 1: `conf=0.25`
2. Train weak model longer
3. Filter pseudo labels by confidence

### Performance Degradation

**Solutions:**
1. Reduce augmentation strength
2. Increase training epochs 
3. Add more source domain labels 

### Path Errors

**Solutions:**
1. Use absolute paths in all yaml files
2. Double-check paths before each step
3. Verify output directories exist after each step

---

## 📚 References

- **Noisy Student**: Xie et al. "Self-training with Noisy Student" (2020)
- **Weak Supervision**: `../weak-supervision/README.md`
- **YOLOv8**: https://docs.ultralytics.com/

---

**Version:** 1.0  
**Last Updated:** 2025
