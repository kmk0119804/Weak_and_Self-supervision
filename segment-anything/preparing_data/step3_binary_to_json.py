"""
Step 3: Convert SAM Binary Masks to LabelMe JSON (Separated by Class)

Converts binary mask images to LabelMe JSON format with polygon annotations.
Outputs are separated by class.

Usage:
    python step3_binary_to_json.py
"""

import os
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import cv2
from tqdm import tqdm

# ============================================================================
# DEFAULT CONFIGURATION (can be overridden when used as module)
# ============================================================================
DEFAULT_SAM_OUTPUT_ROOT = Path("path/to/sam_output")
DEFAULT_IMAGE_ROOT = Path("path/to/images")
DEFAULT_OUTPUT_ROOT = Path("path/to/labels_json")
DEFAULT_SUBSETS = ["train", "val"]
DEFAULT_CLASSES = ["worker", "hardhat", "strap", "hook"]

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
    for ext in (".jpg", ".png", ".jpeg", ".bmp", ".webp", ".tif", ".tiff",
                ".JPG", ".PNG", ".JPEG", ".BMP", ".WEBP", ".TIF", ".TIFF"):
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Original image not found: {image_dir}/{stem}.*")


def group_binary_by_image(binary_dir: Path) -> Dict[str, List[Path]]:
    """
    Group binary PNGs by original image.
    File names like 'image001_1.png', 'image001_2.png' -> grouped as 'image001'
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
    """Extract polygon contours from binary mask image"""
    img = imread_unicode_gray(bin_path)
    if img is None:
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
        polys.append(pts.astype(float).tolist())
    return polys


def convert_group_to_labelme_json(
    original_stem: str,
    original_dir: Path,
    binary_paths: List[Path],
    label_name: str,
    out_dir: Path,
):
    """Convert multiple binary PNGs (for same image) to single LabelMe JSON"""
    # Find original image
    img_path = find_original_image(original_dir, original_stem)

    shapes = []
    for bin_png in binary_paths:
        polys = contours_to_polygons(bin_png)
        for points in polys:
            shapes.append({
                "label": label_name,
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            })

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


def convert_binary_to_json(sam_output_root: Path, image_root: Path, output_root: Path,
                           subsets: List[str], classes: List[str], verbose: bool = True):
    """
    Main conversion function (can be called from other scripts)
    
    Args:
        sam_output_root: SAM output root (contains subset/class/binary/)
        image_root: Original images root
        output_root: Output JSON root
        subsets: List of subsets (e.g., ["train", "val"])
        classes: List of classes (e.g., ["worker", "hardhat"])
        verbose: Print progress messages
    
    Returns:
        dict: Conversion statistics
    """
    
    if verbose:
        print("="*70)
        print("Step 3: Converting SAM Binary to LabelMe JSON")
        print("="*70)
        print(f"SAM output: {sam_output_root}")
        print(f"Images:     {image_root}")
        print(f"Output:     {output_root}")
        print("="*70 + "\n")
    
    grand_total_groups = 0
    grand_total_jsons = 0
    problems: List[str] = []

    for subset in subsets:
        original_dir = image_root / subset

        for cls in classes:
            binary_dir = sam_output_root / subset / cls / "binary"
            out_dir = output_root / subset / cls

            if not binary_dir.exists():
                if verbose:
                    print(f"[Skip] Binary folder not found: {binary_dir}")
                continue
            if not original_dir.exists():
                if verbose:
                    print(f"[Skip] Image folder not found: {original_dir}")
                continue

            groups = group_binary_by_image(binary_dir)
            group_items = sorted(groups.items(), key=lambda kv: kv[0])

            desc = f"{subset}/{cls}"
            if verbose:
                pbar = tqdm(group_items, desc=desc, unit="img", leave=False)
            else:
                pbar = group_items

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
                    problems.append(f"[Image not found] {desc}/{stem}")
                except Exception as e:
                    problems.append(f"[Conversion failed] {desc}/{stem}: {e}")

            grand_total_groups += len(group_items)
            grand_total_jsons += made_here

    if verbose:
        print("\n" + "="*70)
        print("Step 3 Complete!")
        print("="*70)
        print(f"Total image groups: {grand_total_groups}")
        print(f"JSON files created: {grand_total_jsons}")
        if problems:
            print(f"\n[⚠️ Problems: {len(problems)}]")
            for msg in problems[:10]:
                print(f"  {msg}")
            if len(problems) > 10:
                print(f"  ... and {len(problems) - 10} more")
        else:
            print("\nCompleted successfully ✅")
        print("="*70 + "\n")
    
    return {
        "total_groups": grand_total_groups,
        "json_created": grand_total_jsons,
        "problems": len(problems),
        "output_root": output_root
    }


def main():
    """Main function for standalone execution"""
    convert_binary_to_json(
        sam_output_root=DEFAULT_SAM_OUTPUT_ROOT,
        image_root=DEFAULT_IMAGE_ROOT,
        output_root=DEFAULT_OUTPUT_ROOT,
        subsets=DEFAULT_SUBSETS,
        classes=DEFAULT_CLASSES,
        verbose=True
    )


if __name__ == "__main__":
    main()