"""
Step 1: Convert LabelMe JSON (Polygon) to COCO Detection Format

Converts polygon annotations to bounding box detection format.
Image dimensions are automatically read from:
  1. JSON file (imageWidth/imageHeight fields)
  2. Actual image file

Usage:
    python step1_json_to_detection.py
"""

from pathlib import Path
import json
from typing import List, Tuple, Dict
from PIL import Image

# ============================================================================
# CONFIGURATION - Edit these paths
# ============================================================================
JSON_FOLDER = Path("path/to/labelme/json/folder")     # Input: LabelMe JSON files
IMAGE_FOLDER = Path("path/to/images/folder")          # Input: Corresponding images (REQUIRED)
OUTPUT_JSON = Path("path/to/output/annotations.json") # Output: COCO format JSON

# Class names (fixed for construction site)
CLASS_ORDER = ["worker", "hardhat", "strap", "hook"] # class names will be changed as needed

# ============================================================================

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", 
            ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP"]


def get_image_size(json_path: Path, data: dict, image_folder: Path) -> Tuple[int, int]:
    """
    Get image dimensions from JSON or actual image file
    
    Priority:
      1. JSON file (imageWidth/imageHeight)
      2. Actual image file
    
    Returns:
      (width, height)
    
    Raises:
      FileNotFoundError: If image dimensions cannot be determined
    """
    # 1) Try JSON metadata first
    w = data.get("imageWidth")
    h = data.get("imageHeight")
    if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
        return w, h
    
    # 2) Try to find and read actual image
    candidates = []
    
    # From JSON imagePath field
    img_path_field = data.get("imagePath")
    if isinstance(img_path_field, str) and img_path_field:
        candidates.append(Path(img_path_field).name)
    
    # From JSON filename (with various extensions)
    stem = json_path.stem
    candidates += [stem + ext for ext in IMG_EXTS]
    
    # Search for image file
    for name in candidates:
        img_path = image_folder / name
        if img_path.exists():
            try:
                with Image.open(img_path) as im:
                    return im.size  # (width, height)
            except Exception as e:
                print(f"[Warning] Failed to read image {img_path}: {e}")
                continue
    
    # Failed to determine image size
    raise FileNotFoundError(
        f"Cannot determine image size for: {json_path.name}\n"
        f"  - No imageWidth/imageHeight in JSON\n"
        f"  - No matching image found in {image_folder}\n"
        f"  - Searched for: {', '.join(candidates[:5])}"
    )


def bbox_from_points(points) -> Tuple[float, float, float, float]:
    """Calculate bounding box from polygon points"""
    xs = [p[0] for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
    ys = [p[1] for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
    
    if not xs or not ys:
        return None
    
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    if xmax <= xmin or ymax <= ymin:
        return None
    
    return xmin, ymin, xmax, ymax


def rectangle_to_xyxy(points) -> Tuple[float, float, float, float]:
    """Convert rectangle (2 points) to bbox"""
    if len(points) < 2:
        return None
    
    (x1, y1), (x2, y2) = points[0], points[1]
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)
    
    if xmax <= xmin or ymax <= ymin:
        return None
    
    return xmin, ymin, xmax, ymax


def to_coco_xywh(xmin, ymin, xmax, ymax) -> List[float]:
    """Convert xyxy to COCO xywh format"""
    return [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]


def build_categories(class_order: List[str]) -> List[dict]:
    """Build COCO categories"""
    return [{"id": i+1, "name": name, "supercategory": "none"} 
            for i, name in enumerate(class_order)]


def convert_labelme_to_coco(json_folder: Path, image_folder: Path, output_json: Path):
    """Convert LabelMe JSON files to COCO detection format"""
    
    # Validate inputs
    if not json_folder.exists():
        raise FileNotFoundError(f"JSON folder not found: {json_folder}")
    
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    
    # Setup category mapping
    name2catid = {name: idx for idx, name in enumerate(CLASS_ORDER, start=1)}
    
    images = []
    annotations = []
    ann_id = 1
    img_id = 1
    
    # Get all JSON files
    json_files = sorted([p for p in json_folder.iterdir() if p.suffix.lower() == ".json"])
    
    if not json_files:
        print("[Warning] No JSON files found")
        return
    
    print(f"Processing {len(json_files)} JSON files...")
    print(f"Image folder: {image_folder}")
    print()
    
    skipped = 0
    
    for jp in json_files:
        try:
            with jp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Error] Failed to read {jp.name}: {e}")
            skipped += 1
            continue
        
        # Get image filename
        file_name = data.get("imagePath")
        if isinstance(file_name, str) and file_name.strip():
            file_name = Path(file_name).name
        else:
            file_name = jp.stem
        
        # Get image dimensions
        try:
            w, h = get_image_size(jp, data, image_folder)
        except FileNotFoundError as e:
            print(f"[Error] {e}")
            skipped += 1
            continue
        
        images.append({
            "id": img_id,
            "file_name": file_name,
            "width": int(w),
            "height": int(h),
        })
        
        # Process annotations
        shapes = data.get("shapes", [])
        for shp in shapes:
            if not isinstance(shp, dict):
                continue
            
            label = shp.get("label", "").strip()
            if not label or label not in name2catid:
                continue
            
            cat_id = name2catid[label]
            shape_type = shp.get("shape_type", "polygon")
            points = shp.get("points", [])
            
            # Calculate bbox
            if shape_type == "rectangle":
                xyxy = rectangle_to_xyxy(points)
            else:
                xyxy = bbox_from_points(points)
            
            if xyxy is None:
                continue
            
            xmin, ymin, xmax, ymax = xyxy
            xywh = to_coco_xywh(xmin, ymin, xmax, ymax)
            area = float(xywh[2] * xywh[3])
            
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": xywh,
                "area": area,
                "iscrowd": 0,
                "segmentation": []
            })
            ann_id += 1
        
        img_id += 1
    
    # Create COCO format
    coco = {
        "info": {"description": "LabelMe to COCO Detection", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": build_categories(CLASS_ORDER),
    }
    
    # Save output
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*70)
    print("Conversion Complete!")
    print("="*70)
    print(f"Images:         {len(images)}")
    print(f"Annotations:    {len(annotations)}")
    print(f"Categories:     {len(coco['categories'])}")
    if skipped > 0:
        print(f"Skipped:        {skipped}")
    print(f"Output:         {output_json}")
    print("="*70 + "\n")


if __name__ == "__main__":
    convert_labelme_to_coco(JSON_FOLDER, IMAGE_FOLDER, OUTPUT_JSON)