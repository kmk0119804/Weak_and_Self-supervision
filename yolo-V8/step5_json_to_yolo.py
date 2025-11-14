"""
Step 5: Convert LabelMe JSON to YOLO Segmentation Format

Converts LabelMe JSON files to YOLO segmentation txt format.
Image dimensions are read from JSON files (mandatory).

Usage:
    python step5_json_to_yolo_seg.py
"""

import json
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm

# ============================================================================
# DEFAULT CONFIGURATION (can be overridden when used as module)
# ============================================================================
DEFAULT_JSON_DIR = Path("path/to/labels_combined")
DEFAULT_SUBSETS = ["train", "val"]
DEFAULT_CLASSES = ["worker", "hardhat", "strap", "hook"]

# ============================================================================


def convert_labelme_json_segmentation(json_dir: Path, classes: List[str], verbose: bool = True):
    """
    Convert LabelMe JSON to YOLO segmentation format.
    
    Args:
        json_dir: Directory containing JSON files
        classes: List of class names (order matters for class IDs)
        verbose: Print progress messages
    
    Returns:
        dict: Conversion statistics
    """
    
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")
    
    # Find all JSON files
    json_files = sorted(json_dir.glob("**/*.json"))
    
    if not json_files:
        if verbose:
            print(f"[Warning] No JSON files found in: {json_dir}")
        return {"total": 0, "success": 0, "failed": 0, "errors": []}
    
    success = 0
    failed = 0
    errors: List[str] = []
    
    for json_file in tqdm(json_files, desc=f"Converting {json_dir.name}", disable=not verbose):
        try:
            # Load JSON
            with json_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get image dimensions (MANDATORY)
            try:
                h = data['imageHeight']
                w = data['imageWidth']
                
                if not isinstance(h, (int, float)) or not isinstance(w, (int, float)):
                    raise ValueError(f"Invalid image dimensions: width={w}, height={h}")
                
                if h <= 0 or w <= 0:
                    raise ValueError(f"Invalid image dimensions: width={w}, height={h}")
                
            except KeyError as e:
                raise ValueError(f"Missing required field in JSON: {e}")
            
            # Process shapes
            labels = []
            segments = []
            
            for shape in data.get('shapes', []):
                label = shape.get('label', '').strip()
                
                # Skip if not in class list
                if label not in classes:
                    continue
                
                points = shape.get('points', [])
                if not points:
                    continue
                
                # Convert to numpy array and normalize (0~1)
                try:
                    polygons = np.array(points, dtype=np.float64)
                    polygons = polygons / np.array([w, h])
                    
                    # Validate normalized coordinates
                    if np.any(polygons < 0) or np.any(polygons > 1):
                        errors.append(f"{json_file.name}: Coordinates out of bounds (0~1)")
                        continue
                    
                    labels.append(label)
                    segments.append(polygons)
                    
                except Exception as e:
                    errors.append(f"{json_file.name}: Failed to process polygon - {e}")
                    continue
            
            # Write to txt file
            output_txt = json_file.with_suffix('.txt')
            
            with output_txt.open('w', encoding='utf-8') as f:
                for idx, polygon in enumerate(segments):
                    class_id = classes.index(labels[idx])
                    
                    # Write: class_id x1 y1 x2 y2 x3 y3 ...
                    f.write(f"{class_id}")
                    
                    for point in polygon:
                        f.write(f" {point[0]:.6f} {point[1]:.6f}")
                    
                    f.write('\n')
            
            success += 1
            
        except Exception as e:
            failed += 1
            errors.append(f"{json_file.name}: {str(e)}")
            continue
    
    return {
        "total": len(json_files),
        "success": success,
        "failed": failed,
        "errors": errors
    }


def convert_json_to_yolo_seg(json_root: Path, subsets: List[str], classes: List[str], 
                             verbose: bool = True):
    """
    Main conversion function (can be called from other scripts)
    
    Args:
        json_root: Root directory containing subset folders
        subsets: List of subsets to process
        classes: List of class names
        verbose: Print progress messages
    
    Returns:
        dict: Conversion statistics
    """
    
    if verbose:
        print("="*70)
        print("Step 5: Converting LabelMe JSON to YOLO Segmentation")
        print("="*70)
        print(f"Input:   {json_root}")
        print(f"Classes: {classes}")
        print("="*70 + "\n")
    
    total_success = 0
    total_failed = 0
    all_errors: List[str] = []
    
    for subset in subsets:
        subset_dir = json_root / subset
        
        if not subset_dir.exists():
            if verbose:
                print(f"[Skip] Subset not found: {subset}")
            continue
        
        if verbose:
            print(f"📁 Processing subset: {subset}")
        
        result = convert_labelme_json_segmentation(
            json_dir=subset_dir,
            classes=classes,
            verbose=verbose
        )
        
        total_success += result["success"]
        total_failed += result["failed"]
        all_errors.extend([f"[{subset}] {e}" for e in result["errors"]])
    
    if verbose:
        print("\n" + "="*70)
        print("Step 5 Complete!")
        print("="*70)
        print(f"Total files:  {total_success + total_failed}")
        print(f"Successful:   {total_success}")
        print(f"Failed:       {total_failed}")
        
        if all_errors:
            print(f"\n[⚠️ Errors: {len(all_errors)}]")
            for error in all_errors[:10]:
                print(f"  {error}")
            if len(all_errors) > 10:
                print(f"  ... and {len(all_errors) - 10} more")
        else:
            print("\nCompleted successfully ✅")
        
        print("="*70 + "\n")
    
    return {
        "total": total_success + total_failed,
        "success": total_success,
        "failed": total_failed,
        "errors": all_errors
    }


def main():
    """Main function for standalone execution"""
    convert_json_to_yolo_seg(
        json_root=DEFAULT_JSON_DIR,
        subsets=DEFAULT_SUBSETS,
        classes=DEFAULT_CLASSES,
        verbose=True
    )


if __name__ == "__main__":
    main()