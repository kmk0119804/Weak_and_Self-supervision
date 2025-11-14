# Weak-supervision Model Training

Train YOLOv8 segmentation model using SAM-generated pseudo labels combined with a small portion of source domain labels.

---

## 📋 Overview

This module trains a **baseline segmentation model** using:
- **SAM-generated pseudo labels** from target domain images
- **Small portion (~10-20%) of labeled source domain data**

**Purpose:** Train a robust segmentation model that learns from both target domain features (via SAM) and high-quality source annotations.

---

## 📁 Files

```
yolo-V8/weak-supervision/
├── weak_train.py          # Training script
├── weak_inference.py      # Test evaluation script
├── weak_data.yaml         # Dataset configuration
├── weak_config.yaml       # Training hyperparameters (optional)
└── README.md              # This file
```

---

## 📊 Training Data

### Data Composition

```
Training Set:
├── Target domain images + SAM pseudo labels
└── Source domain images + ground truth labels

Validation Set:
├── Target domain images + SAM pseudo labels
└── Source domain images + ground truth labels

Test Set:
└── Target domain images (for final evaluation)
```

### Why Mix Source + Target?

1. **SAM labels**: Capture target domain-specific features
2. **Source labels**: Provide high-quality supervision
3. **Small source portion**: Prevent overfitting to source domain

---

## 🚀 Quick Start

### Prerequisites

**Complete SAM data preparation:**

Run in `segment-anything/preparing_data/` to generate YOLO segmentation labels:
```bash
python step1_json_to_detection.py
python step2_detection_to_voc.py
cd .. && python generate_sam_masks.py && cd preparing_data
python step3_binary_to_json.py
python step4_combine_json.py
python step5_json_to_yolo.py
```

**Expected dataset structure:**
```
dataset/
├── images/
│   ├── train/   (target + small source portion)
│   ├── val/     (target + small source portion)
│   └── test/    (target domain only)
└── labels/
    ├── train/   (YOLO segmentation format)
    ├── val/     (YOLO segmentation format)
    └── test/    (YOLO segmentation format)
```

---

### Step 1: Configure Dataset

**Edit `weak_data.yaml`:**

```yaml
# Paths (use absolute paths)
train: /absolute/path/to/dataset/images/train
val: /absolute/path/to/dataset/images/val
test: /absolute/path/to/dataset/images/test  # Optional

# Class definitions
names:
  0: worker
  1: hardhat
  2: strap
  3: hook

nc: 4
```

**Note:** YOLO automatically finds labels at `labels/train/` when given `images/train/`

---

### Step 2: Train Model

```bash
python weak_train.py
```

**Training configuration in `weak_train.py`:**

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO("path/to/pretrained/Baseline/model.pt")  # or "yolov8m-seg.pt"

# Train
results = model.train(
    data="weak_data.yaml",
    epochs=300,
    imgsz=1280,
    batch=6,
    device=0,
    task='segment'
)
```

**Output:**
```
runs/segment/train/
├── weights/
│   ├── best.pt     ⭐ Best model checkpoint
│   └── last.pt     Last epoch checkpoint
├── results.png     Training curves
└── confusion_matrix.png
```

---

### Step 3: Evaluate on Test Set

```bash
python weak_inference.py
```

**Evaluation configuration in `weak_inference.py`:**

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/segment/train/weights/best.pt')

# Evaluate on test set
results = model.val(
    data='weak_data.yaml',
    imgsz=1280,
    conf=0.25,
    iou=0.5,
    save_json=True
)
```

**Evaluation metrics:**
- mAP@50, mAP@50-95
- Precision, Recall per class
- Confusion matrix

---

## ⚙️ Model Configuration

### Change Model Size

Edit `weak_train.py`:

```python
# Smaller models (faster)
model = YOLO("yolov8n-seg.pt")  # Nano
model = YOLO("yolov8s-seg.pt")  # Small

# Recommended
model = YOLO("yolov8m-seg.pt")  # Medium

# Larger models (more accurate)
model = YOLO("yolov8l-seg.pt")  # Large
model = YOLO("yolov8x-seg.pt")  # Extra Large
```

### Adjust Training Parameters

Edit `weak_train.py`:

```python
results = model.train(
    data="weak_data.yaml",
    epochs=300,      # Training duration
    imgsz=1280,      # Input size (640, 960, 1280)
    batch=6,         # Batch size (reduce if OOM)
    device=0,        # GPU device (0, 1, 2, 3, 'cpu')
    task='segment'
)
```

### Use Custom Hyperparameters

Pass `weak_config.yaml`:
```bash
yolo segment train cfg=weak_config.yaml
```

Or override in `weak_train.py`:
```python
results = model.train(
    data="weak_data.yaml",
    epochs=300,
    lr0=0.01,        # Initial learning rate
    lrf=0.01,        # Final learning rate
    patience=50,     # Early stopping
    # ... more parameters in weak_config.yaml
)
```

---

## 📊 Expected Performance

**Typical metrics on target domain test set:**
- mAP@50: 45-60%
- mAP@50-95: 28-38%

*Performance depends on:*
- Domain gap between source and target
- SAM pseudo label quality
- Source data portion size

**Model comparison:**
- Better than SAM-only training (lacks source supervision)
- Better than source-only training (poor target generalization)

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

```python
# In weak_train.py
batch=4  # Reduce batch size
imgsz=960  # Reduce image size
model = YOLO("yolov8s-seg.pt")  # Use smaller model
```

### Dataset Not Found

```yaml
# In weak_data.yaml - use ABSOLUTE paths
train: /absolute/path/to/images/train  # ✅
train: ./images/train                  # ❌
```

Verify:
- `labels/` folder exists parallel to `images/`
- Label files match image names: `img001.jpg` → `img001.txt`

### Low Performance

1. **Check SAM label quality**: Visualize masks in `segment-anything/preparing_data/`
2. **Increase source portion**: Try 20-30% instead of 10%
3. **Train longer**: 500-1000 epochs
4. **Try larger model**: YOLOv8l-seg or YOLOv8x-seg

---

## 📝 Training Output

```
runs/segment/train/
├── weights/
│   ├── best.pt           ⭐ Best validation mAP
│   └── last.pt           Last epoch
│
├── results.png           Training/validation curves
├── confusion_matrix.png  Per-class performance
├── val_batch*_pred.jpg   Validation predictions
└── args.yaml             Training arguments
```

---

## 🔗 Related Documentation

- **SAM Data Preparation**: `segment-anything/preparing_data/README.md`
- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **YOLO Segmentation Guide**: https://docs.ultralytics.com/tasks/segment/

---

## 💡 Tips

1. **Start with medium model** (YOLOv8m-seg) for good balance
2. **Monitor training curves** in `results.png` for convergence
3. **Validate on target domain** to ensure good adaptation
4. **Save multiple checkpoints** for comparison
5. **Visualize predictions** on test set to verify quality

---

**Version:** 1.0  
**Last Updated:** 2024