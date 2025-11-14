import os
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import cv2
from tqdm import tqdm

# ============================================================================
# CONFIGURATION - Edit these paths
# ============================================================================
SAM_OUTPUT_ROOT = Path("path/to/sam_output")   # Output from generate_sam_masks.py
IMAGE_ROOT = Path("path/to/images")             # Original images
OUTPUT_ROOT = Path("path/to/labels_json")       # JSON output

SUBSETS: List[str] = ["train", "val"]
CLASSES: List[str] = ["worker", "hardhat", "strap", "hook"]

# ============================================================================

def imread_unicode_gray(p: Path) -> np.ndarray | None:
    """Safely read grayscale image (handles Unicode paths on Windows)"""
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return None

def find_original_image(image_dir: Path, stem: str) -> Path:
    """Find original image with automatic extension detection"""
    for ext in (".jpg", ".png", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"):
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Original image not found: {image_dir}\\{stem}.*")

def group_binary_by_image(binary_dir: Path) -> Dict[str, List[Path]]:
    """
    Group binary PNGs by original image.
    File names like 'image001_1.png', 'image001_2.png' -> grouped as 'image001'
    If no underscore, use filename as-is.
    """
    groups: Dict[str, List[Path]] = {}
    for p in binary_dir.glob("*.png"):
        name = p.stem
        if "_" in name:
            orig = name.rsplit("_", 1)[0]
        else:
            orig = name
        groups.setdefault(orig, []).append(p)
    return groups

def contours_to_polygons(bin_path: Path) -> List[List[List[float]]]:
    """
    Extract polygon contours from binary mask image.
    - No noise filtering
    - No simplification
    - Coordinates stored as float
    """
    img = imread_unicode_gray(bin_path)
    if img is None:
        # imread failed
        return []

    # Binarize (any non-zero value -> 255)
    mask = (img > 0).astype(np.uint8) * 255

    # Extract contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    polys: List[List[List[float]]] = []
    for cnt in contours:
        if cnt.ndim != 3 or cnt.shape[1] != 1 or cnt.shape[2] != 2:
            continue
        pts = cnt.reshape(-1, 2)  # (N, 2)
        # LabelMe: list of [x,y]
        polys.append(pts.astype(float).tolist())
    return polys

def convert_group_to_labelme_json(
    original_stem: str,
    original_dir: Path,
    binary_paths: List[Path],
    label_name: str,
    out_dir: Path,
) -> dict:
    """
    Convert multiple binary PNGs (for same image) to single LabelMe JSON.
    """
    # Find original image
    img_path = find_original_image(original_dir, original_stem)

    shapes = []
    for bin_png in binary_paths:
        polys = contours_to_polygons(bin_png)
        # Contours may not exist (completely black/white image)
        for points in polys:
            shapes.append({
                "label": label_name,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })

    # imageData can be large -> omit to save space (LabelMe can still open with imagePath)
    data = {
        "version": "5.1.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_path.name,
        "imageData": None
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{original_stem}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data

# ========== Main Loop ==========

def main():
    grand_total_groups = 0
    grand_total_jsons  = 0
    problems: List[str] = []  # Warning/problem paths

    for subset in SUBSETS:
        original_dir = IMAGE_ROOT / subset

        for cls in CLASSES:
            binary_dir = SAM_OUTPUT_ROOT / subset / cls / "binary"
            out_dir = OUTPUT_ROOT / subset / cls

            if not binary_dir.exists():
                print(f"[Skip] Binary folder not found: {binary_dir}")
                continue
            if not original_dir.exists():
                print(f"[Skip] Image folder not found: {original_dir}")
                continue

            groups = group_binary_by_image(binary_dir)
            group_items = sorted(groups.items(), key=lambda kv: kv[0])

            desc = f"{subset}/{cls}"
            pbar = tqdm(group_items, desc=desc, unit="img", leave=True)

            made_here = 0
            for stem, bin_list in pbar:
                try:
                    convert_group_to_labelme_json(
                        original_stem=stem,
                        original_dir=original_dir,
                        binary_paths=bin_list,
                        label_name=cls,
                        out_dir=out_dir,
                    )
                    made_here += 1
                except FileNotFoundError as e:
                    problems.append(f"[Image not found] {desc}/{stem} -> {e}")
                except Exception as e:
                    problems.append(f"[Conversion failed] {desc}/{stem} -> {e}")

            pbar.close()

            grand_total_groups += len(group_items)
            grand_total_jsons  += made_here

    print("\n==== Summary ====")
    print(f"Total image groups: {grand_total_groups}")
    print(f"JSON files created: {grand_total_jsons}")
    if problems:
        print("\n[⚠️ Problems]")
        for msg in problems:
            print(msg)
    else:
        print("\nCompleted successfully ✅")

if __name__ == "__main__":
    main()