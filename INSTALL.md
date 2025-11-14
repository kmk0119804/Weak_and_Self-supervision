# Installation

## Prerequisites

- Python 3.8+
- CUDA 11.0+ (recommended)

## Quick Install

```bash
# Clone repository
git clone https://github.com/kmk0119804/Minkyu-KOO
cd Weak_and_Self-supervision

# Install PyTorch (choose your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install YOLOv8 and SAM
pip install ultralytics segment-anything

# Install other dependencies
pip install opencv-python pillow numpy tqdm
```

## Download SAM Checkpoint

```bash
cd segment-anything
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
```

## Verify Installation

```python
import torch
from ultralytics import YOLO
print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
```

---

**Next:** See [GETTING_STARTED.md](GETTING_STARTED.md)