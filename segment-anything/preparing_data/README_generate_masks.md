# SAM-Based Segmentation Data Preparation Pipeline

Complete pipeline for generating YOLO segmentation labels from LabelMe polygon annotations using Segment Anything Model (SAM).

> **Note:** This pipeline utilizes the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) developed by Meta AI Research. Please refer to the official SAM repository for model details, installation, and usage guidelines.

---

## 📋 Overview

This pipeline converts polygon annotations from LabelMe into high-quality YOLO segmentation labels using SAM (Segment Anything Model). The workflow consists of 5 main steps plus SAM mask generation.

### Workflow

```
LabelMe JSON (polygon)
    ↓  [Step 1]
YOLO Detection (bbox)
    ↓  [Step 2]
Pascal VOC (bbox)
    ↓  [SAM]
Binary Masks (per object)
    ↓  [Step 3]
LabelMe JSON (per class)
    ↓  [Step 4]
LabelMe JSON (combined)
    ↓  [Step 5]
YOLO Segmentation
```

---

## 🛠️ Requirements

### Python Environment
```bash
pip install numpy opencv-python pillow tqdm
```

### SAM Installation
For SAM model setup and checkpoint download, please refer to the official repository:
- **SAM Repository:** https://github.com/facebookresearch/segment-anything
- **Installation Guide:** https://github.com/facebookresearch/segment-anything#installation
- **Model Checkpoints:** https://github.com/facebookresearch/segment-anything#model-checkpoints

Required packages for SAM:
- `segment-anything`
- `torch` and `torchvision` (CUDA recommended for GPU acceleration)

---

## 📁 Directory Structure

### Initial Setup
```
project/
├── data/
│   ├── images/
│   │   ├── train/
│   │   │   ├── img001.jpg
│   │   │   └── ...
│   │   └── val/
│   │       └── ...
│   └── labelme_json/
│       ├── train/
│       │   ├── img001.json  # LabelMe polygon annotations
│       │   └── ...
│       └── val/
│           └── ...
├── segment-anything/
│   ├── preparing_data/
│   │   ├── step1_json_to_detection.py
│   │   ├── step2_detection_to_voc.py
│   │   ├── step3_binary_to_json.py
│   │   ├── step4_combine_json.py
│   │   ├── step5_json_to_yolo.py
│   │   └── README.md  (this file)
│   ├── generate_sam_masks.py
│   └── sam_vit_h_4b8939.pth  # Download from SAM repo
└── output/
    ├── detection_labels/
    ├── pascal_voc/
    ├── sam_output/
    ├── labels_json/
    ├── labels_combined/
    └── yolo_seg/  (final output)
```

---

## 🚀 Quick Start

### 1. Configure Paths

Edit the `DEFAULT_*` variables in each script to match your directory structure:

**step1_json_to_detection.py:**
```python
DEFAULT_JSON_FOLDER = Path("data/labelme_json/train")
DEFAULT_IMAGE_FOLDER = Path("data/images/train")
DEFAULT_OUTPUT_DIR = Path("output/detection_labels")
DEFAULT_CLASS_ORDER = ["worker", "hardhat", "strap", "hook"]
```

**step2_detection_to_voc.py:**
```python
DEFAULT_INPUT_ROOT = Path("output/detection_labels")
DEFAULT_OUTPUT_ROOT = Path("output/pascal_voc")
DEFAULT_IMAGE_FOLDER = Path("data/images/train")
```

**generate_sam_masks.py:**
```python
SAM_CHECKPOINT = "segment-anything/sam_vit_h_4b8939.pth"
DEVICE = "cuda"  # or "cpu"

JOBS = [
    {
        "split": "train",
        "pascal_voc_root": "output/pascal_voc/train",
        "image_folder": "data/images/train",
        "output_root": "output/sam_output/train",
        "classes": ["worker", "hardhat", "strap", "hook"],
    },
]
```

**step3_binary_to_json.py:**
```python
DEFAULT_SAM_OUTPUT_ROOT = Path("output/sam_output")
DEFAULT_IMAGE_ROOT = Path("data/images")
DEFAULT_OUTPUT_ROOT = Path("output/labels_json")
DEFAULT_SUBSETS = ["train", "val"]
DEFAULT_CLASSES = ["worker", "hardhat", "strap", "hook"]
```

**step4_combine_json.py:**
```python
DEFAULT_INPUT_ROOT = Path("output/labels_json")
DEFAULT_OUTPUT_ROOT = Path("output/labels_combined")
DEFAULT_IMAGE_ROOT = Path("data/images")
DEFAULT_SUBSETS = ["train", "val"]
DEFAULT_CLASSES = ["worker", "hardhat", "strap", "hook"]
```

**step5_json_to_yolo.py:**
```python
DEFAULT_JSON_DIR = Path("output/labels_combined")
DEFAULT_IMAGE_ROOT = Path("data/images")
DEFAULT_SUBSETS = ["train", "val"]
DEFAULT_CLASSES = ["worker", "hardhat", "strap", "hook"]
```

### 2. Run Pipeline

```bash
# Navigate to preparing_data directory
cd segment-anything/preparing_data

# Step 1: Convert LabelMe JSON to YOLO Detection
python step1_json_to_detection.py

# Step 2: Convert YOLO Detection to Pascal VOC
python step2_detection_to_voc.py

# Step 3: Generate SAM masks (move to parent directory)
cd ..
python generate_sam_masks.py

# Step 4: Convert SAM binary masks to JSON
cd preparing_data
python step3_binary_to_json.py

# Step 5: Combine class-separated JSONs
python step4_combine_json.py

# Step 6: Convert to YOLO segmentation format
python step5_json_to_yolo.py
```

---

## 📖 Detailed Step Descriptions

### Step 1: LabelMe JSON → YOLO Detection

**Purpose:** Extract bounding boxes from polygon annotations

**Input:**
```
labelme_json/train/
└── img001.json  (LabelMe polygon format)
```

**Output:**
```
detection_labels/
├── worker/
│   └── img001.txt  (0 0.5 0.5 0.2 0.3)
├── hardhat/
├── strap/
└── hook/
```

**Format:** YOLO detection (`class_id cx cy w h`, normalized 0-1)

---

### Step 2: YOLO Detection → Pascal VOC

**Purpose:** Convert to absolute pixel coordinates for SAM

**Input:**
```
detection_labels/worker/img001.txt
```

**Output:**
```
pascal_voc/
├── worker/
│   └── img001.txt  ([[100, 200, 300, 400], ...])
├── hardhat/
├── strap/
└── hook/
```

**Format:** JSON array of bounding boxes (`[[xmin, ymin, xmax, ymax], ...]`)

---

### SAM: Generate Segmentation Masks

**Purpose:** Generate precise segmentation masks from bounding boxes

**Input:**
```
pascal_voc/train/worker/img001.txt
images/train/img001.jpg
```

**Output:**
```
sam_output/
└── train/
    └── worker/
        ├── binary/
        │   ├── img001_1.png  (individual object masks)
        │   ├── img001_2.png
        │   └── ...
        └── binary_sum/
            └── img001.png  (combined mask)
```

**Note:** This step uses GPU and may take significant time

---

### Step 3: Binary Masks → LabelMe JSON

**Purpose:** Convert binary masks to polygon coordinates

**Input:**
```
sam_output/train/worker/binary/img001_1.png
```

**Output:**
```
labels_json/
└── train/
    └── worker/
        └── img001.json  (LabelMe format, worker objects only)
```

---

### Step 4: Combine Class-Separated JSONs

**Purpose:** Merge per-class JSONs into single files

**Input:**
```
labels_json/train/worker/img001.json
labels_json/train/hardhat/img001.json
labels_json/train/strap/img001.json
labels_json/train/hook/img001.json
```

**Output:**
```
labels_combined/
└── train/
    └── img001.json  (all classes combined)
```

---

### Step 5: LabelMe JSON → YOLO Segmentation

**Purpose:** Convert to final YOLO segmentation format

**Input:**
```
labels_combined/train/img001.json
```

**Output:**
```
labels_combined/
└── train/
    └── img001.txt  (0 0.1 0.2 0.15 0.25 ...)
```

**Format:** YOLO segmentation (`class_id x1 y1 x2 y2 ...`, normalized 0-1)

---

## 💡 Usage as Modules

All scripts can be imported and used programmatically:

```python
from pathlib import Path
import step1_json_to_detection as step1
import step2_detection_to_voc as step2
# ... etc

# Step 1
result = step1.convert_json_to_detection(
    json_folder=Path("data/labelme_json/train"),
    image_folder=Path("data/images/train"),
    output_dir=Path("output/detection_labels"),
    class_order=["worker", "hardhat", "strap", "hook"],
    verbose=True
)

# Step 2
result = step2.convert_detection_to_voc(
    input_root=Path("output/detection_labels"),
    output_root=Path("output/pascal_voc"),
    image_folder=Path("data/images/train"),
    verbose=True
)

# ... continue with other steps
```

---

## ⚙️ Configuration

### Class Order
**IMPORTANT:** Class order must be consistent across all steps!

```python
DEFAULT_CLASSES = ["worker", "hardhat", "strap", "hook"]
# Class IDs: worker=0, hardhat=1, strap=2, hook=3
```

### Image Extensions
Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`
(both lowercase and uppercase)

### SAM Settings
```python
MODEL_TYPE = "vit_h"  # Options: vit_h, vit_l, vit_b
DEVICE = "cuda"       # Use GPU for faster processing
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Image Not Found Errors
**Problem:** Scripts cannot find corresponding images

**Solution:**
- Verify image file extensions match
- Check that image filenames match label filenames (without extension)
- Ensure image folders are correctly specified

#### 2. Missing Image Dimensions
**Problem:** Step 5 fails with "Cannot get image dimensions"

**Solution:**
- Ensure Step 4 successfully found all images
- Provide `image_root` parameter to Step 5
- Check that image files are accessible

#### 3. CUDA Out of Memory (SAM)
**Problem:** GPU memory exhausted during SAM processing

**Solutions:**
- Process fewer images at a time
- Use smaller SAM model (`vit_b` instead of `vit_h`)
- Use CPU instead of GPU (slower): `DEVICE = "cpu"`
- Clear CUDA cache between batches

#### 4. Empty Output Files
**Problem:** Generated txt files are empty

**Solution:**
- Verify class names match exactly (case-sensitive)
- Check that original annotations have valid polygons
- Ensure SAM generated masks successfully

#### 5. Coordinate Out of Bounds Warnings
**Problem:** Normalized coordinates exceed [0, 1] range

**Solution:**
- Usually safe to ignore (coordinates are clamped)
- May indicate annotation errors in original data

---

## 📊 Output Validation

### Check Conversion Success

```bash
# Count files at each stage
find output/detection_labels -name "*.txt" | wc -l
find output/pascal_voc -name "*.txt" | wc -l
find output/sam_output -name "*.png" | wc -l
find output/labels_combined -name "*.txt" | wc -l

# Verify final YOLO format
head -n 5 output/labels_combined/train/img001.txt
# Expected: "0 0.123456 0.234567 0.345678 ..."
```

### Visualize Results

Use tools like:
- [LabelMe](https://github.com/wkentaro/labelme) for JSON visualization
- [CVAT](https://github.com/opencv/cvat) for annotation review
- Custom visualization scripts with OpenCV/PIL

---

## 📝 Important Notes

### File Naming Convention
- All label files must match image filenames (without extension)
- Example: `img001.jpg` → `img001.txt`, `img001.json`

### Class-Specific Processing
- Steps 1-3 process each class separately
- Step 4 combines all classes per image
- This allows parallel processing if needed

### Data Integrity
- Original images are never modified
- All intermediate outputs are preserved
- Each step is idempotent (can be re-run safely)

### Performance Tips
- Use SSD for faster I/O
- Enable GPU for SAM (10-50x faster than CPU)
- Process train/val splits in parallel
- Monitor disk space (SAM outputs can be large)

---

## 🔗 References

- **Segment Anything Model (SAM):** https://github.com/facebookresearch/segment-anything
  - Kirillov, A., et al. (2023). "Segment Anything." arXiv:2304.02643
- **LabelMe:** https://github.com/wkentaro/labelme
- **YOLO Format:** https://docs.ultralytics.com/datasets/segment/

---

**Version:** 1.0  
**Last Updated:** 2024