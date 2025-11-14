# Getting Started

## 1. Download Dataset

**[📥 Download Dataset](https://forms.gle/YOUR_FORM_LINK)** (5.43GB)

```bash
unzip dataset.zip
```

## 2. Generate SAM Labels

```bash
cd segment-anything/preparing_data

# Run all steps
python step1_json_to_detection.py
python step2_detection_to_voc.py
cd .. && python generate_sam_masks.py && cd preparing_data
python step3_binary_to_json.py
python step4_combine_json.py
python step5_json_to_yolo.py
```

**Note:** Edit paths in each script before running.

## 3. Train Baseline

```bash
cd ../../yolo-V8/baseline
# Edit data.yaml with your paths
python train.py
```

## 4. Train Weak Supervision

```bash
cd ../weak-supervision
# Edit weak_data.yaml with your paths
python weak_train.py
```

## 5. Self-Training

```bash
cd ../self-training

# Generate pseudo labels
python self_inference.py
python self_yolo_to_yolo.py

# Train student model
python self_train.py

# Evaluate
python self_inference.py
```

**Note:** See individual README files in each folder for detailed instructions.

## Expected Results

| Stage | mAP@50 | Training Time |
|:------|:-------|:--------------|
| Baseline | ~42% | 2-4 hours |
| Weak Supervision | ~52% | 8-12 hours |
| Self-Training | ~58% | 12-16 hours |

---

**Questions?** Open an issue or contact: kmk0119804@yonsei.ac.kr