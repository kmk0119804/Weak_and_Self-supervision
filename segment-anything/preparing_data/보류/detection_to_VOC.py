"""
Step 2: Convert YOLO Detection Labels to Pascal VOC Format

Converts YOLO txt labels (per class) to Pascal VOC bbox format (JSON).

Usage:
    python step2_detection_to_voc.py
"""

import json
from pathlib import Path

# ============================================================================
# CONFIGURATION - Edit these paths
# ============================================================================
INPUT_ROOT = Path("path/to/yolo/labels")    # Input: worker/, hardhat/, strap/, hook/ folders
OUTPUT_ROOT = Path("path/to/pascal_voc")    # Output: Pascal VOC format folders
IMAGE_WIDTH = 1920                          # Image width
IMAGE_HEIGHT = 1080                         # Image height

# Class mapping (fixed)
CLASS_MAP = {
    0: "worker",
    1: "hardhat",
    2: "strap",
    3: "hook",
}
VALID_CLASS_NAMES = set(CLASS_MAP.values())

# ============================================================================


def clamp(v, lo, hi):
    """Clamp value between min and max"""
    return max(lo, min(hi, v))


def parse_class_name(token: str):
    """Convert class ID to name"""
    try:
        cls_id = int(token)
        return CLASS_MAP.get(cls_id, str(cls_id))
    except ValueError:
        return token


def yolo_line_to_xyxy(line: str, img_w: int, img_h: int):
    """
    Convert YOLO format to Pascal VOC bbox
    
    Supports:
    - BBox: 'cls cx cy w h' or 'cls cx cy w h conf'
    - Segmentation: 'cls x1 y1 x2 y2 ... xN yN'
    
    Returns: (cls_name, xmin, ymin, xmax, ymax)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Invalid line format: {line}")
    
    cls_name = parse_class_name(parts[0])
    
    # Detection bbox format
    if len(parts) in (5, 6):
        cx, cy, w, h = map(float, parts[1:5])
        abs_cx = cx * img_w
        abs_cy = cy * img_h
        abs_w = w * img_w
        abs_h = h * img_h
        xmin = abs_cx - abs_w / 2.0
        ymin = abs_cy - abs_h / 2.0
        xmax = abs_cx + abs_w / 2.0
        ymax = abs_cy + abs_h / 2.0
    
    # Polygon format - calculate bbox
    else:
        coords = list(map(float, parts[1:]))
        if len(coords) % 2 != 0:
            # Try skipping confidence value
            float(parts[1])
            coords = list(map(float, parts[2:]))
            if len(coords) % 2 != 0:
                raise ValueError(f"Invalid polygon coordinates: {line}")
        
        xs = coords[0::2]
        ys = coords[1::2]
        abs_xs = [x * img_w for x in xs]
        abs_ys = [y * img_h for y in ys]
        xmin, xmax = min(abs_xs), max(abs_xs)
        ymin, ymax = min(abs_ys), max(abs_ys)
    
    # Clamp and convert to int
    xmin = int(round(clamp(xmin, 0, img_w - 1)))
    ymin = int(round(clamp(ymin, 0, img_h - 1)))
    xmax = int(round(clamp(xmax, 0, img_w - 1)))
    ymax = int(round(clamp(ymax, 0, img_h - 1)))
    
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    
    return cls_name, xmin, ymin, xmax, ymax


def convert_class_folder(input_dir: Path, output_dir: Path, 
                        img_w: int, img_h: int, target_class: str):
    """Convert one class folder"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    txt_files = [p for p in input_dir.iterdir() if p.suffix.lower() == ".txt"]
    if not txt_files:
        return 0, 0
    
    converted, failed = 0, 0
    
    for txt_file in sorted(txt_files):
        try:
            with txt_file.open("r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            
            boxes = []
            for ln in lines:
                cls_name, xmin, ymin, xmax, ymax = yolo_line_to_xyxy(ln, img_w, img_h)
                if cls_name == target_class:
                    boxes.append([xmin, ymin, xmax, ymax])
            
            # Save as single-line JSON
            out_path = output_dir / txt_file.name
            with out_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(boxes, ensure_ascii=False))
            
            converted += 1
        
        except Exception as e:
            print(f"[Warning] Failed to convert {txt_file.name}: {e}")
            failed += 1
    
    return converted, failed


def convert_all_classes(input_root: Path, output_root: Path, img_w: int, img_h: int):
    """Convert all class folders"""
    
    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    total_classes = 0
    total_files = 0
    total_conv = 0
    total_fail = 0
    
    # Process each class folder
    for class_dir in sorted([d for d in input_root.iterdir() if d.is_dir()]):
        cls_name = class_dir.name
        
        if cls_name not in VALID_CLASS_NAMES:
            print(f"[Info] Skipping unknown class: {cls_name}")
            continue
        
        txt_files = [p for p in class_dir.iterdir() if p.suffix.lower() == ".txt"]
        if not txt_files:
            continue
        
        total_classes += 1
        total_files += len(txt_files)
        
        print(f"Processing: {cls_name} ({len(txt_files)} files)...")
        
        out_dir = output_root / cls_name
        conv, fail = convert_class_folder(class_dir, out_dir, img_w, img_h, cls_name)
        
        total_conv += conv
        total_fail += fail
    
    # Print summary
    print("\n" + "="*70)
    print("Conversion Complete!")
    print("="*70)
    print(f"Classes:        {total_classes}")
    print(f"Input files:    {total_files}")
    print(f"Converted:      {total_conv}")
    print(f"Failed:         {total_fail}")
    print(f"Output folder:  {output_root}")
    print("="*70 + "\n")


if __name__ == "__main__":
    convert_all_classes(INPUT_ROOT, OUTPUT_ROOT, IMAGE_WIDTH, IMAGE_HEIGHT)