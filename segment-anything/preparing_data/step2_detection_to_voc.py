"""
Step 2: Convert YOLO Detection Labels to Pascal VOC Format

Converts YOLO txt labels (per class) to Pascal VOC bbox format (JSON).
Image dimensions are automatically detected from actual image files.

Usage:
    python step2_detection_to_voc.py
"""

import json
from pathlib import Path
from PIL import Image

# ============================================================================
# DEFAULT CONFIGURATION (can be overridden when used as module)
# ============================================================================
DEFAULT_INPUT_ROOT = Path("path/to/yolo/labels")
DEFAULT_OUTPUT_ROOT = Path("path/to/pascal_voc")
DEFAULT_IMAGE_FOLDER = Path("path/to/images")

# Class mapping (fixed)
CLASS_MAP = {
    0: "worker",
    1: "hardhat",
    2: "strap",
    3: "hook",
}
VALID_CLASS_NAMES = set(CLASS_MAP.values())

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP"]

# ============================================================================


def find_image_and_get_size(txt_file: Path, image_folder: Path):
    """Find corresponding image and get its dimensions"""
    stem = txt_file.stem
    
    for ext in IMG_EXTS:
        img_path = image_folder / (stem + ext)
        if img_path.exists():
            try:
                with Image.open(img_path) as img:
                    return img.size  # (width, height)
            except Exception:
                continue
    
    raise FileNotFoundError(f"Image not found for: {txt_file.name}")


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
    
    Input: 'cls cx cy w h' (normalized 0~1)
    Output: (cls_name, xmin, ymin, xmax, ymax) (absolute pixels)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Invalid line format: {line}")
    
    cls_name = parse_class_name(parts[0])
    
    # Detection bbox format: cls cx cy w h
    cx, cy, w, h = map(float, parts[1:5])
    
    # Convert to absolute coordinates
    abs_cx = cx * img_w
    abs_cy = cy * img_h
    abs_w = w * img_w
    abs_h = h * img_h
    
    # Calculate bbox
    xmin = abs_cx - abs_w / 2.0
    ymin = abs_cy - abs_h / 2.0
    xmax = abs_cx + abs_w / 2.0
    ymax = abs_cy + abs_h / 2.0
    
    # Clamp and convert to int
    xmin = int(round(clamp(xmin, 0, img_w - 1)))
    ymin = int(round(clamp(ymin, 0, img_h - 1)))
    xmax = int(round(clamp(xmax, 0, img_w - 1)))
    ymax = int(round(clamp(ymax, 0, img_h - 1)))
    
    # Ensure valid bbox
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    
    return cls_name, xmin, ymin, xmax, ymax


def convert_class_folder(input_dir: Path, output_dir: Path, 
                        image_folder: Path, target_class: str, verbose: bool = True):
    """Convert one class folder"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    txt_files = [p for p in input_dir.iterdir() if p.suffix.lower() == ".txt"]
    if not txt_files:
        return 0, 0
    
    converted, failed = 0, 0
    
    for txt_file in sorted(txt_files):
        try:
            # Get image dimensions automatically
            img_w, img_h = find_image_and_get_size(txt_file, image_folder)
            
            # Read YOLO labels
            with txt_file.open("r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            
            # Convert to Pascal VOC format
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
        
        except FileNotFoundError as e:
            if verbose:
                print(f"[Warning] {e}")
            failed += 1
        except Exception as e:
            if verbose:
                print(f"[Warning] Failed to convert {txt_file.name}: {e}")
            failed += 1
    
    return converted, failed


def convert_detection_to_voc(input_root: Path, output_root: Path, image_folder: Path, 
                             verbose: bool = True):
    """
    Main conversion function (can be called from other scripts)
    
    Args:
        input_root: Path to YOLO detection labels (class folders)
        output_root: Path to output directory
        image_folder: Path to original images
        verbose: Print progress messages
    
    Returns:
        dict: Conversion statistics
    """
    
    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")
    
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {image_folder}")
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("="*70)
        print("Step 2: Converting YOLO Detection to Pascal VOC Format")
        print("="*70)
        print(f"Input:   {input_root}")
        print(f"Output:  {output_root}")
        print(f"Images:  {image_folder}")
        print("="*70 + "\n")
    
    total_classes = 0
    total_files = 0
    total_conv = 0
    total_fail = 0
    
    # Process each class folder
    for class_dir in sorted([d for d in input_root.iterdir() if d.is_dir()]):
        cls_name = class_dir.name
        
        if cls_name not in VALID_CLASS_NAMES:
            if verbose:
                print(f"[Info] Skipping unknown class: {cls_name}")
            continue
        
        txt_files = [p for p in class_dir.iterdir() if p.suffix.lower() == ".txt"]
        if not txt_files:
            continue
        
        total_classes += 1
        total_files += len(txt_files)
        
        if verbose:
            print(f"Processing: {cls_name} ({len(txt_files)} files)...")
        
        out_dir = output_root / cls_name
        conv, fail = convert_class_folder(class_dir, out_dir, image_folder, cls_name, verbose)
        
        total_conv += conv
        total_fail += fail
    
    if verbose:
        print("\n" + "="*70)
        print("Step 2 Complete!")
        print("="*70)
        print(f"Classes:        {total_classes}")
        print(f"Input files:    {total_files}")
        print(f"Converted:      {total_conv}")
        print(f"Failed:         {total_fail}")
        print(f"Output folder:  {output_root}")
        print("="*70 + "\n")
    
    return {
        "classes": total_classes,
        "total": total_files,
        "converted": total_conv,
        "failed": total_fail,
        "output_dir": output_root
    }


def main():
    """Main function for standalone execution"""
    convert_detection_to_voc(
        input_root=DEFAULT_INPUT_ROOT,
        output_root=DEFAULT_OUTPUT_ROOT,
        image_folder=DEFAULT_IMAGE_FOLDER,
        verbose=True
    )


if __name__ == "__main__":
    main()