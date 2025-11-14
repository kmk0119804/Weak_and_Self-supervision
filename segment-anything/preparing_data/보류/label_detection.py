"""
Step 1: Convert LabelMe JSON (Polygon) to YOLO Detection TXT (Separated by Class)

Converts polygon/segmentation annotations to YOLO detection format (bounding boxes).
Outputs are separated by class into different folders.

Usage:
    python step1_json_to_detection.py
"""

import json
from pathlib import Path
from collections import defaultdict
from PIL import Image

# ============================================================================
# CONFIGURATION - Edit these paths
# ============================================================================
JSON_FOLDER = Path("path/to/labelme/json")     # Input: LabelMe JSON files
IMAGE_FOLDER = Path("path/to/images")          # Input: Corresponding images
OUTPUT_DIR = Path("path/to/output/labels")     # Output: Class folders with txt files

# Class names (in order, 0-indexed)
CLASS_ORDER = ["worker", "hardhat", "strap", "hook"]

# ============================================================================

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP"]


def get_image_size(json_path: Path, data: dict, image_folder: Path):
    """
    Get image dimensions from JSON or actual image file.
    
    Priority:
      1. JSON metadata (imageWidth/imageHeight)
      2. Actual image file
    
    Returns: (width, height)
    """
    # 1) Try JSON metadata first
    w = data.get("imageWidth")
    h = data.get("imageHeight")
    if isinstance(w, int) and isinstance(h, int) and w > 0 and h > 0:
        return w, h

    # 2) Search for image file
    candidates = []
    
    # From JSON imagePath field
    img_path_field = data.get("imagePath")
    if isinstance(img_path_field, str) and img_path_field:
        candidates.append(Path(img_path_field).name)
    
    # From JSON filename with various extensions
    stem = json_path.stem
    for ext in IMG_EXTS:
        candidates.append(stem + ext)
    
    # Try to find and open image
    for name in candidates:
        img_path = image_folder / name
        if img_path.exists():
            try:
                with Image.open(img_path) as img:
                    return img.size  # (width, height)
            except Exception:
                continue
    
    raise FileNotFoundError(f"Image not found for: {json_path.name}")


def bbox_from_points(points):
    """Extract bounding box from polygon points"""
    xs = [p[0] for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
    ys = [p[1] for p in points if isinstance(p, (list, tuple)) and len(p) == 2]
    
    if not xs or not ys:
        return None
    
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    if xmax <= xmin or ymax <= ymin:
        return None
    
    return xmin, ymin, xmax, ymax


def bbox_from_rectangle(points):
    """Extract bounding box from rectangle (2 points)"""
    if len(points) < 2:
        return None
    
    (x1, y1), (x2, y2) = points[0], points[1]
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)
    
    if xmax <= xmin or ymax <= ymin:
        return None
    
    return xmin, ymin, xmax, ymax


def to_yolo_format(xmin, ymin, xmax, ymax, img_w, img_h):
    """
    Convert absolute bbox to YOLO format (normalized).
    Returns: (cx, cy, w, h) all in [0, 1]
    """
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    
    if w <= 0 or h <= 0:
        return None
    
    # Normalize
    cx_norm = max(0.0, min(1.0, cx / img_w))
    cy_norm = max(0.0, min(1.0, cy / img_h))
    w_norm = max(0.0, min(1.0, w / img_w))
    h_norm = max(0.0, min(1.0, h / img_h))
    
    if w_norm == 0 or h_norm == 0:
        return None
    
    return cx_norm, cy_norm, w_norm, h_norm


def convert_json_to_yolo_txt(json_path: Path, image_folder: Path, output_dir: Path, 
                              class_to_id: dict, id_to_name: dict):
    """Convert one JSON file to YOLO detection txt (separated by class)"""
    
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Get image dimensions
    img_w, img_h = get_image_size(json_path, data, image_folder)
    
    # Separate lines by class
    class_lines = defaultdict(list)
    shapes = data.get("shapes", [])
    
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        
        # Get label
        label = shape.get("label", "").strip()
        if not label or label not in class_to_id:
            continue
        
        class_id = class_to_id[label]
        class_name = id_to_name[class_id]
        
        # Extract bbox based on shape type
        shape_type = shape.get("shape_type", "polygon")
        points = shape.get("points", [])
        
        if shape_type == "rectangle":
            bbox = bbox_from_rectangle(points)
        else:
            bbox = bbox_from_points(points)
        
        if bbox is None:
            continue
        
        # Convert to YOLO format
        yolo_bbox = to_yolo_format(*bbox, img_w, img_h)
        if yolo_bbox is None:
            continue
        
        cx, cy, w, h = yolo_bbox
        line = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        class_lines[class_name].append(line)
    
    # Save to class-specific folders
    for class_name, lines in class_lines.items():
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        output_txt = class_dir / (json_path.stem + ".txt")
        with output_txt.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))


def main():
    """Main conversion function"""
    
    if not JSON_FOLDER.exists():
        raise FileNotFoundError(f"JSON folder not found: {JSON_FOLDER}")
    
    # Build class mapping
    class_to_id = {name: idx for idx, name in enumerate(CLASS_ORDER)}
    id_to_name = {idx: name for name, idx in class_to_id.items()}
    
    # Get all JSON files
    json_files = sorted([p for p in JSON_FOLDER.iterdir() if p.suffix.lower() == ".json"])
    
    if not json_files:
        print("[Warning] No JSON files found")
        return
    
    print("="*70)
    print("Converting LabelMe JSON to YOLO Detection TXT (Class Separated)")
    print("="*70)
    print(f"Input:   {JSON_FOLDER}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Classes: {CLASS_ORDER}")
    print(f"Files:   {len(json_files)}")
    print("="*70 + "\n")
    
    # Convert each JSON file
    success = 0
    failed = 0
    
    for json_file in json_files:
        try:
            convert_json_to_yolo_txt(json_file, IMAGE_FOLDER, OUTPUT_DIR, 
                                     class_to_id, id_to_name)
            success += 1
        except Exception as e:
            print(f"[Error] Failed to convert {json_file.name}: {e}")
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("Conversion Complete!")
    print("="*70)
    print(f"Total files:      {len(json_files)}")
    print(f"Successful:       {success}")
    print(f"Failed:           {failed}")
    print(f"Class mapping:    {class_to_id}")
    print(f"Output structure: {OUTPUT_DIR}/<class_name>/*.txt")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()