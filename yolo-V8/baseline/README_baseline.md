# Baseline Model Training

Train YOLOv8 Segmentation Model on source domain data.

This baseline model serves as a **comparison benchmark** to measure performance improvement of the proposed weak supervision method.

---

## 📌 Reference

This implementation is based on **Ultralytics YOLOv8** segmentation model.

For detailed information about YOLOv8, please refer to:
- **Repository**: https://github.com/ultralytics/ultralytics
- **Documentation**: https://docs.ultralytics.com/
- **Paper**: [YOLOv8 Technical Report](https://github.com/ultralytics/ultralytics)

---

## 📁 Files

```
baseline/
├── baseline_train.py          # Training script
├── baseline_inference.py      # Validation/Test script
├── baseline_data.yaml         # Dataset configuration ← EDIT THIS!
├── baseline_config.yaml       # Training hyperparameters (optional)
└── README.md                  # This file
```

---

## 📋 Prerequisites

Prepare your dataset in YOLO segmentation format:

```
your_dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── labels/
    ├── train/
    │   ├── img001.txt    # YOLO format: class x1 y1 x2 y2 ... xn yn
    │   └── ...
    └── val/
        └── ...
```

---

## 🚀 Training

### Step 1: Configure Dataset Paths

**Edit `baseline_data.yaml`:**

```yaml
# Training set - use your absolute path
train: /your/path/to/dataset/images/train

# Validation set - use your absolute path
val: /your/path/to/dataset/images/val

# Class names
names:
  0: worker
  1: hardhat
  2: strap
  3: hook

nc: 4
```

**Important**: 
- Use **absolute paths**
- YOLO automatically finds labels: `images/train/` → `labels/train/`

### Step 2: (Optional) Customize Training Settings

**Edit `baseline_config.yaml`** to adjust hyperparameters:
- Training epochs
- Batch size
- Learning rate
- Data augmentation
- etc.

See `baseline_config.yaml` for available options.

### Step 3: Run Training

```bash
python baseline_train.py
```

**Training output:**
```
runs/segment/train/
├── weights/
│   ├── best.pt      ⭐ Best model
│   └── last.pt      Last epoch
├── results.png      Training curves
└── ...
```

---

## 📊 Evaluation

### Validate on Validation Set

```bash
python baseline_inference.py
```

**Output**: Validation metrics (mAP@50, mAP@50-95, Precision, Recall)

---

### Evaluate on Test Set

**Folder renaming required!**

#### Step 1: Rename Folders

```bash
# In your dataset/images/
mv val val_origin      # Backup original val
mv test val            # Rename test to val
```

#### Step 2: Run Inference

```bash
python baseline_inference.py
```

#### Step 3: Restore Folders

```bash
# In your dataset/images/
mv val test            # Restore test
mv val_origin val      # Restore val
```

**Why?** `baseline_inference.py` uses paths from `baseline_data.yaml`, which points to `val/` folder.

---

## ⚙️ Customization

### Change Model Size

Edit `baseline_train.py`:

```python
# Options: yolov8n-seg, yolov8s-seg, yolov8m-seg, yolov8l-seg, yolov8x-seg
model = YOLO("yolov8m-seg.pt")  # Medium (default)
```

| Model | Accuracy | Speed | Memory |
|:------|:---------|:------|:-------|
| yolov8n-seg | Lower | ⚡⚡⚡ Fast | 2GB |
| yolov8s-seg | Low | ⚡⚡ Fast | 4GB |
| yolov8m-seg | Medium | ⚡ Medium | 8GB |
| yolov8l-seg | High | 🐌 Slow | 12GB |
| yolov8x-seg | Higher | 🐌🐌 Slow | 16GB |

### Adjust Training Parameters

Edit `baseline_train.py`:

```python
results = model.train(
    data="baseline_data.yaml",
    epochs=300,      # Training duration
    imgsz=1280,      # Input size
    batch=6,         # Batch size
    device=0,        # GPU device
    task='segment'
)
```

Or use `baseline_config.yaml` for more options.

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

```python
# In baseline_train.py
batch=4  # Reduce batch size
imgsz=960  # Reduce image size
```

### Dataset not found

- Check paths in `baseline_data.yaml` are absolute
- Verify folder structure: `images/` and `labels/`
- Ensure label files exist and match image names

---

## 📝 Notes

### Purpose of Baseline Model

This baseline model is trained on **source domain data** and serves as:
- **Performance benchmark** for comparison
- **Starting point** to measure improvement from weak supervision

### Test Set Evaluation

- `baseline_inference.py` uses `val` path from `baseline_data.yaml`
- To evaluate on test set: temporarily rename `test → val`
- **Always restore** original folder names after testing

---

## 🔗 Next Steps

After training baseline model:

1. **Record performance**: Note mAP scores for comparison
2. **Save weights**: `runs/segment/train/weights/best.pt`
3. **Proceed to weak supervision**: Apply proposed method
4. **Compare results**: Measure performance improvement

---

**Last Updated**: November 2024