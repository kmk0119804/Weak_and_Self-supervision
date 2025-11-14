# Dataset

Construction Site Safety Dataset for Domain-Adaptive Instance Segmentation

---

## 📥 Download

**[Download Dataset Here!](https://forms.gle/2wuEjjv9Hn5McHmj7)** (5.43GB, zip)

*Fill out a brief form to access the download link.*

---

## 📊 Dataset Information

- **Size:** 5.43GB (compressed)
- **Images:** 3,000+
- **Classes:** 4 classes (worker, hardhat, strap, hook)
- **Annotation:** LabelMe JSON format (polygon annotations)
---

## 🗂️ Dataset Structure

```
dataset/
├── Source domain/
│   ├── images/
│   └── json/              # LabelMe JSON annotations
│
├── Target domain1(YUD-COSA dataset)/
│   ├── images/
│   └── json/
│
├── Target domain2/
│   ├── images/
│   └── json/
│
├── Target domain3/
│   ├── images/
│   └── json/
│
├── Target domain4/
│   ├── images/
│   └── json/
│
├── Target domain5/
│   ├── images/
│   └── json/
│
├── Target domain6/
│   ├── images/
│   └── json/
│
└── Target domain7/
    ├── images/
    └── json/
```

**Note:** YOLO format labels will be generated through the SAM pipeline (see main README).

---

## 🚀 Quick Start

### 1. Download & Extract

```bash
# After downloading dataset.zip
unzip dataset.zip -d Weak_and_Self-supervision/
```

### 2. Process Annotations

Follow the SAM pipeline in `segment-anything/preparing_data/` to convert JSON annotations to YOLO segmentation format.

---

## 🔒 Terms of Use

**Allowed:**
- ✅ Academic research and education
- ✅ Non-commercial projects
- ✅ Publications with citation

**Prohibited:**
- ❌ Commercial use without permission
- ❌ Redistribution or re-hosting
- ❌ Privacy violations

**Citation:**
```bibtex
@article{manguy2024domain,
  title={Domain-Adaptive Instance Segmentation for Far-Field Object Monitoring},
  author={Manguy and Collaborators},
  year={2024}
}
```

---

## 📧 Contact

Questions? Email: kmk0119804@yonsei.ac.kr

---

## ⚠️ Privacy Notice

All sensitive information has been removed or anonymized. Faces are blurred, and locations are anonymized.


## 📢 Important Notice

**We monitor all download requests.** If we detect improper or incomplete 
form responses (e.g., fake information, spam), we reserve the right to:
- Suspend public access to the dataset
- Require individual approval for future requests
- Restrict access to verified researchers only

Please provide genuine information to help us maintain open access 
for the research community.