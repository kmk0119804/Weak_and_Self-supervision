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

# ================== Common Settings ==================
sam_checkpoint = "path/to/sam_vit_h_4b8939.pth" # Download from SAM repo
model_type = "vit_h"
device = "cuda"  # or "cpu"
# ==============================================

# List of jobs to process (add as many as needed)
# out_vis_dir  -> Individual object binary folder (filename_1.png, filename_2.png, ...)
# out_mask_dir -> Combined binary folder (filename.png)
JOBS = [
    # ===== target domain train =====
    {
        "label_folder": "path/to/pascal_voc",
        "image_folder": "path/to/images",
        "out_vis_dir":  "path/to/sam_output/binary",
        "out_mask_dir": "path/to/sam_output/binary_sum",
    },

    # ===== target domain val =====
    {
        "label_folder": "path/to/pascal_voc",
        "image_folder": "path/to/images",
        "out_vis_dir":  "path/to/sam_output/binary",
        "out_mask_dir": "path/to/sam_output/binary_sum",
    },
]

# ================== Utility Functions ==================
def read_text_files_in_folder(folder_path: str):
    file_contents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            # Encoding issue prevention
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_contents[filename] = f.read()
    return file_contents

def find_image_path(image_folder: str, base_name: str) -> Optional[str]:
    """Find image path corresponding to label filename (base_name) in jpg/png/etc."""
    exts = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
    for ext in exts:
        p = os.path.join(image_folder, base_name + ext)
        if os.path.exists(p):
            return p
    return None

def init_sam():
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def process_one_job(predictor, label_folder, image_folder, out_vis_dir, out_mask_dir):
    # Create folders
    Path(out_vis_dir).mkdir(parents=True, exist_ok=True)   # Individual object binary
    Path(out_mask_dir).mkdir(parents=True, exist_ok=True)  # Combined binary

    file_contents = read_text_files_in_folder(label_folder)
    for filename, content in tqdm(file_contents.items(), desc=f'Processing [{Path(label_folder).name}]'):
        try:
            # "[[xmin,ymin,xmax,ymax], ...]" -> list
            boxes_list = list(ast.literal_eval(content))
            if not boxes_list:
                # Skip empty labels (uncomment to log)
                # print(f"[Info] Empty label: {filename}")
                continue

            # Find image with automatic extension detection based on label filename (without extension)
            base_name = os.path.splitext(filename)[0]
            img_path = find_image_path(image_folder, base_name)
            if img_path is None:
                print(f"[Warning] No corresponding image (jpg/png not found): {base_name}")
                continue

            # Load image
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                print(f"[Warning] Image load failed (corrupted/permission/encoding): {img_path}")
                continue

            H, W = image_bgr.shape[:2]
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            predictor.set_image(image)

            # Nx4 (xyxy)
            input_boxes = torch.tensor(boxes_list, dtype=torch.float32, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            # Combined mask
            combined_mask = np.zeros((H, W), dtype=np.uint8)

            # Save filename (without extension)
            img_stem = Path(img_path).stem

            # --- Save individual objects & combine ---
            for idx, mask in enumerate(masks, start=1):
                m = mask.detach().cpu().numpy().squeeze()
                m_bin = (m > 0.5).astype(np.uint8) * 255  # 0/255

                # Save individual object: filename_idx.png
                obj_path = Path(out_vis_dir) / f"{img_stem}_{idx}.png"
                Image.fromarray(m_bin, mode='L').save(str(obj_path))

                # Update combined mask
                combined_mask = np.maximum(combined_mask, m_bin)

            # Save combined binary
            sum_path = Path(out_mask_dir) / f"{img_stem}.png"
            Image.fromarray(combined_mask, mode='L').save(str(sum_path))

        except Exception as e:
            print(f"Error processing file {filename}: {e}")
            continue

    # Memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    predictor = init_sam()
    for job in JOBS:
        process_one_job(
            predictor,
            job["label_folder"],
            job["image_folder"],
            job["out_vis_dir"],   # Individual object binary
            job["out_mask_dir"],  # Combined binary
        )

if __name__ == "__main__":
    main()