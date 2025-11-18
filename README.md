# Domain-Adaptive Instance Segmentation Using SAM-Based Weak Supervision and Noisy Student Self-Training

A complete pipeline for training robust segmentation models on construction site CCTV data using weak and self-supervision approaches.

---

## 📋 Overview

This repository implements a domain adaptation framework for instance segmentation in far-field monitoring scenarios where:
- Manual pixel-level annotation is impractical
- Domain shift exists between source and target data
- Small objects require precise segmentation

**Training Pipeline:**
1. **Baseline Model**: Train on well-annotated source domain
2. **SAM Data Preparation**: Generate pseudo labels for target domain using SAM
3. **Weak Supervision**: Train with SAM labels + small source portion
4. **Self-Training**: Improve with Noisy Student approach

---
## 📦 Dataset Setup

### 1. Download Dataset

**📋 Request dataset access:**

👉 **[Complete Dataset Request Form](https://docs.google.com/forms/d/e/1FAIpQLScpNdkz1ylxaJLIlcUPoM4LkQPIlNcA2Twbl4Yvp7d6OMgMfQ/viewform?usp=header)**

You can download the dataset immediately after completing the form.

---

**By downloading the dataset, you agree to:**
- Use the dataset solely for research purposes
- Not distribute or share the dataset with others
- Cite this repository if you use the dataset in publications

**For commercial use inquiries, please contact: kmk0119804@yonsei.ac.kr**

---

## 🛠️ Installation

This project requires both **YOLOv8** and **SAM (Segment Anything Model)** environments.

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (recommended for GPU support)
- Conda or Miniconda

### Step 1: Create Conda Environment

```bash
# Create new conda environment
conda create -n weak_self python=3.8 -y
conda activate weak_self
```

### Step 2: Install PyTorch

```bash
cd Weak_and_Self-supervision
pip install -r requirements.txt
```

### Step 3: Install YOLOv8(optional)

```bash
pip install ultralytics
```

**For detailed installation and usage, refer to:**
- **YOLOv8 Repository**: https://github.com/ultralytics/ultralytics
- **YOLOv8 Documentation**: https://docs.ultralytics.com/

### Step 4: Install Segment Anything Model (SAM)

```bash
pip install segment-anything
```

**For SAM setup, model checkpoints, and usage, refer to:**
- **SAM Repository**: https://github.com/facebookresearch/segment-anything
- **Installation Guide**: https://github.com/facebookresearch/segment-anything#installation
- **Model Checkpoints**: https://github.com/facebookresearch/segment-anything#model-checkpoints

### Step 5: Install Additional Dependencies(optional)

```bash
pip install numpy opencv-python pillow tqdm
```

### Step 6: Download SAM Checkpoint

```bash
cd segment-anything
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
```

---

## 🔄 Complete Pipeline

### Stage 1: Baseline Model Training

**Purpose:** Train initial segmentation model on source domain with ground truth labels

**Location:** `yolo-V8/baseline/`

**Training Data:**
- Source domain images with ground truth labels

**Process:**
```bash
cd yolo-V8/baseline
python baseline_train.py
python baseline_inference.py
```

**📖 Detailed Guide:** See `yolo-V8/baseline/README_baseline.md`

---

### Stage 2: SAM-Based Data Preparation

**Purpose:** Generate high-quality YOLO segmentation labels from polygon annotations using SAM

**Location:** `segment-anything/preparing_data/`

**Process:**
```bash
cd segment-anything/preparing_data

python step1_json_to_detection.py
python step2_detection_to_voc.py
cd .. && python generate_sam_masks.py && cd preparing_data
python step3_binary_to_json.py
python step4_combine_json.py
python step5_json_to_yolo.py
```

**Note:** Edit paths in each script before running.

**📖 Detailed Guide:** See `segment-anything/preparing_data/README.md`

---

### Stage 3: Weak Supervision Training

**Purpose:** Train robust model using SAM pseudo labels + small source portion

**Location:** `yolo-V8/weak-supervision/`

**Training Data:**
- Target domain: SAM-generated pseudo labels
- Source domain: Ground truth labels

**Process:**
```bash
cd yolo-V8/weak-supervision

python weak_train.py
python weak_inference.py
```

**📖 Detailed Guide:** See `yolo-V8/weak-supervision/README.md`

---

### Stage 4: Noisy Student Self-Training

**Purpose:** Improve model through iterative self-training with strong augmentation

**Location:** `yolo-V8/self-training/`

**Process:**
```bash
cd yolo-V8/self-training

# Step 1: Generate pseudo labels
python self_inference.py
python self_yolo_to_yolo.py

# Step 2: Train student model
python self_train.py

# Step 3: Evaluate
python self_inference.py
```

**📖 Detailed Guide:** See `yolo-V8/self-training/README.md`

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n weak_self python=3.8 -y
conda activate weak_self

# Install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install ultralytics segment-anything opencv-python pillow numpy tqdm
```

### 2. Download SAM Checkpoint

```bash
cd segment-anything
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
```

### 3. Download Dataset

See `dataset/README.md` for dataset download instructions.

### 4. Run Pipeline

```bash
# Stage 1: Baseline
cd yolo-V8/baseline
python train.py

# Stage 2: SAM Data Preparation  
cd ../../segment-anything/preparing_data
python step1_json_to_detection.py
# ... (run all steps)

# Stage 3: Weak Supervision
cd ../../yolo-V8/weak-supervision
python weak_train.py

# Stage 4: Self-Training
cd ../self-training
python self_inference.py
python self_yolo_to_yolo.py
python self_train.py
```

---


### Key Advantages

1. **No Manual Annotation**: SAM generates high-quality pseudo labels
2. **Domain Adaptation**: Handles source-target domain shift
3. **Iterative Improvement**: Self-training progressively improves performance
4. **Practical**: Works with limited labeled source data

---

## 📖 Documentation

Detailed documentation for each stage:

- **Installation Guide**: [INSTALLATION.md](INSTALLATION.md)
- **Getting Started**: [GETTING_STARTED.md](GETTING_STARTED.md)
- **Baseline Training**: [yolo-V8/baseline/README.md](yolo-V8/baseline/README.md)
- **SAM Data Preparation**: [segment-anything/preparing_data/README.md](segment-anything/preparing_data/README.md)
- **Weak Supervision**: [yolo-V8/weak-supervision/README.md](yolo-V8/weak-supervision/README.md)
- **Self-Training**: [yolo-V8/self-training/README.md](yolo-V8/self-training/README.md)
- **Dataset Guide**: [dataset/README.md](dataset/README.md)

---

## 🔗 References

### Core Technologies

- **YOLOv8**: https://github.com/ultralytics/ultralytics
  - Ultralytics YOLOv8 Documentation: https://docs.ultralytics.com/
- **Segment Anything Model (SAM)**: https://github.com/facebookresearch/segment-anything
  - Kirillov, A., et al. (2023). "Segment Anything." arXiv:2304.02643

### Methodology

- **Noisy Student**: Xie, Q., et al. (2020). "Self-training with Noisy Student improves ImageNet classification." CVPR 2020.
- **Weak Supervision**: Ratner, A., et al. (2020). "Weak Supervision: A New Programming Paradigm for Machine Learning."

---

## 💡 Tips for Best Results

1. **Start with baseline** to verify source domain data quality
2. **Check SAM outputs** visually before weak supervision training
3. **Monitor training curves** at each stage for convergence
4. **Adjust augmentation** in self-training based on domain characteristics
5. **Save checkpoints** from each stage for comparison and ablation studies

---

## 🐛 Common Issues

### Installation Issues
- **CUDA version mismatch**: Ensure PyTorch CUDA version matches your system
- **SAM import error**: Verify segment-anything installation: `pip install segment-anything`
- **YOLOv8 import error**: Update ultralytics package: `pip install -U ultralytics`

### Training Issues
- **Out of Memory**: Reduce batch size or image size in training scripts
- **Poor convergence**: Check learning rate and verify data quality
- **Low mAP**: Verify label format (YOLO segmentation) and class mapping

### Data Issues
- **Path not found**: Use absolute paths in all yaml configuration files
- **Label format error**: Ensure YOLO segmentation format with normalized coordinates [0, 1]
- **Missing images**: Check that image-label filename pairs match (without extension)

---

## 📧 Contact

For questions or issues, please contact:
- **Email**: kmk0119804@yonsei.ac.kr
- **Institution**: Yonsei University

---

## 🙏 Acknowledgments

- Segment Anything Model (SAM) by Meta AI Research
- Ultralytics YOLOv8 framework

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

**Version:** 1.0  
**Last Updated:** 2025
