"""
Step 4: Combine Class-Separated JSON Files

Combines per-class JSON files into single JSON files per image.
All classes for the same image are merged into one JSON file.

Usage:
    python step4_combine_json.py
"""

from pathlib import Path
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from PIL import Image

# ============================================================================
# DEFAULT CONFIGURATION (can be overridden when used as module)
# ============================================================================
DEFAULT_INPUT_ROOT = Path("path/to/labels_json")
DEFAULT_OUTPUT_ROOT = Path("path/to/labels_combined")
DEFAULT_IMAGE_ROOT = Path("path/to/images")
DEFAULT_SUBSETS = ["train", "val"]
DEFAULT_CLASSES = ["worker", "hardhat", "strap", "hook"]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".WEBP", ".TIF", ".TIFF")

# ============================================================================


def load_json(p: Path) -> dict:
    """Load JSON file"""
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p: Path, data: dict) -> None:
    """Save JSON file"""
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def collect_by_filename(input_root: Path, subset: str, classes: List[str]) -> Dict[str, List[Path]]:
    """Group per-class JSON files by filename"""
    bucket: Dict[str, List[Path]] = defaultdict(list)
    for cls in classes:
        cls_dir = input_root / subset / cls
        if not cls_dir.exists():
            continue
        for p in cls_dir.glob("*.json"):
            bucket[p.name].append(p)
    return bucket


def try_pil_size(img_path: Path) -> Optional[Tuple[int, int]]:
    """Read image size using PIL (returns w,h). Returns None on failure"""
    try:
        with Image.open(img_path) as im:
            return im.size  # (w, h)
    except Exception:
        return None


def find_image_with_hint(image_dir: Path, stem: str, hint_names: List[str]) -> Optional[Tuple[int, int, str]]:
    """
    Find image and get dimensions.
    1) Try hint file names first (from JSON imagePath)
    2) Try stem + various extensions
    Returns (w, h, filename) on success
    """
    # 1) Try hints first
    for hint in hint_names:
        if not hint or not isinstance(hint, str):
            continue
        cand = image_dir / Path(hint).name
        if cand.exists():
            size = try_pil_size(cand)
            if size:
                w, h = size
                return int(w), int(h), cand.name

    # 2) Try stem + multiple extensions
    for ext in IMG_EXTS:
        cand = image_dir / f"{stem}{ext}"
        if cand.exists():
            size = try_pil_size(cand)
            if size:
                w, h = size
                return int(w), int(h), cand.name

    return None


def merge_jsons_for_image(json_paths: List[Path], image_dir: Path, stem: str) -> Tuple[dict, bool]:
    """
    Merge per-class JSONs for same image (stem).
    - Combine all shapes
    - Get version/flags from first JSON (OK if missing)
    - Get imageWidth/Height/Path from original image (None on failure)
    - imageData is always None (save space)
    Returns: (merged_json, found_image_flag)
    """
    merged_shapes: List[dict] = []
    version = None
    flags = None
    hint_names: List[str] = []

    for jp in json_paths:
        try:
            data = load_json(jp)
        except Exception:
            continue
        if version is None:
            version = data.get("version")
        if flags is None:
            flags = data.get("flags", {})
        sh = data.get("shapes", [])
        if isinstance(sh, list):
            merged_shapes.extend(sh)
        # Collect imagePath hints
        ip = data.get("imagePath")
        if isinstance(ip, str):
            hint_names.append(ip)

    meta = find_image_with_hint(image_dir, stem, hint_names)
    if meta:
        w, h, fname = meta
        found = True
    else:
        w = h = None
        fname = None
        found = False

    merged = {
        "version": version if version else "5.1.1",
        "flags": flags if isinstance(flags, dict) else {},
        "shapes": merged_shapes,
        "imagePath": fname,
        "imageData": None,
        "imageWidth": w,
        "imageHeight": h,
    }
    return merged, found


def combine_json_files(input_root: Path, output_root: Path, image_root: Path,
                       subsets: List[str], classes: List[str], verbose: bool = True):
    """
    Main combination function (can be called from other scripts)
    
    Args:
        input_root: Input JSON root (from step3)
        output_root: Output combined JSON root
        image_root: Original images root
        subsets: List of subsets
        classes: List of classes
        verbose: Print progress messages
    
    Returns:
        dict: Combination statistics
    """
    
    if verbose:
        print("="*70)
        print("Step 4: Combining Class-Separated JSON Files")
        print("="*70)
        print(f"Input:  {input_root}")
        print(f"Output: {output_root}")
        print(f"Images: {image_root}")
        print("="*70 + "\n")
    
    counters = {"merged": 0}
    missing: List[str] = []

    for subset in subsets:
        image_dir = image_root / subset
        output_subset_dir = output_root / subset

        groups = collect_by_filename(input_root, subset, classes)
        if not groups:
            if verbose:
                print(f"[{subset}] No JSON files found")
            continue

        if verbose:
            print(f"[{subset}] Merging {len(groups)} images...")

        for fname, paths in tqdm(groups.items(), desc=f"  {subset}", leave=False, 
                                disable=not verbose):
            stem = Path(fname).stem
            merged, found = merge_jsons_for_image(paths, image_dir=image_dir, stem=stem)
            if not found:
                missing.append(f"{subset}/{fname}")
            save_json(output_subset_dir / fname, merged)
            counters["merged"] += 1

    if verbose:
        print("\n" + "="*70)
        print("Step 4 Complete!")
        print("="*70)
        print(f"Merged JSON files: {counters['merged']}")
        if missing:
            print(f"\n[⚠️ Images not found: {len(missing)}]")
            print("  (Image dimensions set to null in JSON)")
            for item in missing[:10]:
                print(f"  {item}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
        else:
            print("\nAll images found ✅")
        print("="*70 + "\n")
    
    return {
        "merged": counters["merged"],
        "missing": len(missing),
        "output_root": output_root
    }


def main():
    """Main function for standalone execution"""
    combine_json_files(
        input_root=DEFAULT_INPUT_ROOT,
        output_root=DEFAULT_OUTPUT_ROOT,
        image_root=DEFAULT_IMAGE_ROOT,
        subsets=DEFAULT_SUBSETS,
        classes=DEFAULT_CLASSES,
        verbose=True
    )


if __name__ == "__main__":
    main()