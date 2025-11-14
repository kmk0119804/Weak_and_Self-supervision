"""
Generate SAM Masks from Pascal VOC Bounding Boxes

Uses Segment Anything Model (SAM) to generate segmentation masks from bounding boxes.
Automatically processes all classes in the dataset.

Usage:
    python generate_sam_masks.py
"""

import os
import ast
from tqdm import tqdm
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

# SAM model settings
SAM_CHECKPOINT = "path/to/sam_vit_h_4b8939.pth"  # Download from SAM repo
MODEL_TYPE = "vit_h"  # Options: vit_h, vit_l, vit_b
DEVICE = "cuda"  # or "cpu"

# Dataset jobs to process
# Each job processes one split (train/val) with all classes automatically
JOBS = [
    {
        "split": "train",
        "pascal_voc_root": "path/to/pascal_voc/train",  # Contains class folders
        "image_folder": "path/to/images/train",
        "output_root": "path/to/sam_output/train",
        "classes": ["worker", "hardhat", "strap", "hook"],
    },
    {
        "split": "val",
        "pascal_voc_root": "path/to/pascal_voc/val",
        "image_folder": "path/to/images/val",
        "output_root": "path/to/sam_output/val",
        "classes": ["worker", "hardhat", "strap", "hook"],
    },
]

# ============================================================================
# Utility Functions
# ============================================================================

IMG_EXTS = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", 
            ".bmp", ".BMP", ".tif", ".TIF", ".tiff", ".TIFF"]


def read_text_files_in_folder(folder_path: str):
    """Read all txt files in folder and return dict {filename: content}"""
    file_contents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_contents[filename] = f.read()
    return file_contents


def find_image_path(image_folder: str, base_name: str) -> Optional[str]:
    """Find image path with automatic extension detection"""
    for ext in IMG_EXTS:
        img_path = os.path.join(image_folder, base_name + ext)
        if os.path.exists(img_path):
            return img_path
    return None


def init_sam():
    """Initialize SAM model and predictor"""
    print(f"Loading SAM model: {MODEL_TYPE}")
    print(f"Checkpoint: {SAM_CHECKPOINT}")
    print(f"Device: {DEVICE}\n")
    
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    
    return predictor


def process_one_class(predictor, label_folder: Path, image_folder: Path, 
                     out_vis_dir: Path, out_mask_dir: Path, class_name: str):
    """Process one class folder"""
    
    if not label_folder.exists():
        print(f"[Warning] Label folder not found: {label_folder}")
        return 0, 0
    
    # Create output directories
    out_vis_dir.mkdir(parents=True, exist_ok=True)   # Individual objects
    out_mask_dir.mkdir(parents=True, exist_ok=True)  # Combined masks
    
    # Read all label files
    file_contents = read_text_files_in_folder(str(label_folder))
    
    if not file_contents:
        print(f"[Info] No label files in: {label_folder}")
        return 0, 0
    
    success_count = 0
    error_count = 0
    
    pbar = tqdm(file_contents.items(), desc=f"  {class_name}", leave=False)
    
    for filename, content in pbar:
        try:
            # Parse bounding boxes: "[[xmin,ymin,xmax,ymax], ...]"
            boxes_list = list(ast.literal_eval(content))
            
            if not boxes_list:
                continue
            
            # Find corresponding image
            base_name = os.path.splitext(filename)[0]
            img_path = find_image_path(str(image_folder), base_name)
            
            if img_path is None:
                error_count += 1
                continue
            
            # Load image
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                error_count += 1
                continue
            
            H, W = image_bgr.shape[:2]
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Set image for SAM
            predictor.set_image(image_rgb)
            
            # Prepare bounding boxes (Nx4, xyxy format)
            input_boxes = torch.tensor(boxes_list, dtype=torch.float32, 
                                      device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(
                input_boxes, image_rgb.shape[:2]
            )
            
            # Generate masks
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            
            # Initialize combined mask
            combined_mask = np.zeros((H, W), dtype=np.uint8)
            img_stem = Path(img_path).stem
            
            # Save individual objects and combine
            for idx, mask in enumerate(masks, start=1):
                # Convert mask to binary (0 or 255)
                m = mask.detach().cpu().numpy().squeeze()
                m_bin = (m > 0.5).astype(np.uint8) * 255
                
                # Save individual object: filename_idx.png
                obj_path = out_vis_dir / f"{img_stem}_{idx}.png"
                Image.fromarray(m_bin, mode='L').save(str(obj_path))
                
                # Update combined mask
                combined_mask = np.maximum(combined_mask, m_bin)
            
            # Save combined mask
            sum_path = out_mask_dir / f"{img_stem}.png"
            Image.fromarray(combined_mask, mode='L').save(str(sum_path))
            
            success_count += 1
            
        except Exception as e:
            error_count += 1
            continue
    
    return success_count, error_count


def process_one_job(predictor, job_config):
    """Process one job (one split with all classes)"""
    
    split = job_config["split"]
    pascal_voc_root = Path(job_config["pascal_voc_root"])
    image_folder = Path(job_config["image_folder"])
    output_root = Path(job_config["output_root"])
    classes = job_config["classes"]
    
    print("="*70)
    print(f"Processing Split: {split}")
    print("="*70)
    print(f"Pascal VOC: {pascal_voc_root}")
    print(f"Images:     {image_folder}")
    print(f"Output:     {output_root}")
    print(f"Classes:    {classes}")
    print("="*70 + "\n")
    
    total_success = 0
    total_errors = 0
    
    # Process each class
    for class_name in classes:
        label_folder = pascal_voc_root / class_name
        out_vis_dir = output_root / class_name / "binary"
        out_mask_dir = output_root / class_name / "binary_sum"
        
        print(f"Processing class: {class_name}")
        
        success, errors = process_one_class(
            predictor=predictor,
            label_folder=label_folder,
            image_folder=image_folder,
            out_vis_dir=out_vis_dir,
            out_mask_dir=out_mask_dir,
            class_name=class_name
        )
        
        total_success += success
        total_errors += errors
        
        print(f"  ✅ Success: {success}, ❌ Errors: {errors}\n")
    
    print("="*70)
    print(f"Split '{split}' Complete!")
    print("="*70)
    print(f"Total processed: {total_success}")
    print(f"Total errors:    {total_errors}")
    print("="*70 + "\n")
    
    # Memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    """Main execution function"""
    
    print("="*70)
    print("SAM Mask Generation from Pascal VOC Bounding Boxes")
    print("="*70 + "\n")
    
    # Initialize SAM
    predictor = init_sam()
    
    # Process all jobs
    for job in JOBS:
        process_one_job(predictor, job)
    
    print("="*70)
    print("🎉 All jobs complete!")
    print("="*70)


if __name__ == "__main__":
    main()