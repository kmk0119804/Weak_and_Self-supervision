"""
Main Pipeline: LabelMe JSON → Pascal VOC Format

This script orchestrates the complete conversion pipeline:
  Step 1: JSON → YOLO Detection (temp)
  Step 2: YOLO Detection → Pascal VOC (final)
  Cleanup: Remove temporary files

Usage:
    python main.py
"""

import shutil
from pathlib import Path
import step1_json_to_detection as step1
import step2_detection_to_voc as step2

# ============================================================================
# CONFIGURATION - Edit these paths
# ============================================================================
JSON_FOLDER = Path("path/to/labelme/json")     # Input: LabelMe JSON files
IMAGE_FOLDER = Path("path/to/images")          # Input: Corresponding images
OUTPUT_DIR = Path("path/to/pascal_voc")        # Output: Pascal VOC format folders

# Class names (in order, 0-indexed)
CLASS_ORDER = ["worker", "hardhat", "strap", "hook"]

# Temporary folder (will be deleted after conversion)
TEMP_DIR = Path("temp_detection")

# ============================================================================


def main():
    """Main pipeline execution"""
    
    print("="*70)
    print("STARTING PIPELINE: JSON → Pascal VOC")
    print("="*70)
    print(f"Input JSON:   {JSON_FOLDER}")
    print(f"Images:       {IMAGE_FOLDER}")
    print(f"Output:       {OUTPUT_DIR}")
    print(f"Temp folder:  {TEMP_DIR}")
    print("="*70 + "\n")
    
    try:
        # ====================================================================
        # STEP 1: Convert JSON to YOLO Detection (temporary)
        # ====================================================================
        print("🔄 Starting Step 1...\n")
        
        step1_result = step1.convert_json_to_detection(
            json_folder=JSON_FOLDER,
            image_folder=IMAGE_FOLDER,
            output_dir=TEMP_DIR,
            class_order=CLASS_ORDER,
            verbose=True
        )
        
        if step1_result["success"] == 0:
            print("❌ Step 1 failed: No files converted")
            return
        
        # ====================================================================
        # STEP 2: Convert YOLO Detection to Pascal VOC
        # ====================================================================
        print("🔄 Starting Step 2...\n")
        
        step2_result = step2.convert_detection_to_voc(
            input_root=TEMP_DIR,
            output_root=OUTPUT_DIR,
            image_folder=IMAGE_FOLDER,
            verbose=True
        )
        
        # ====================================================================
        # CLEANUP: Remove temporary files
        # ====================================================================
        print("🧹 Cleaning up temporary files...")
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            print(f"✅ Removed temporary folder: {TEMP_DIR}\n")
        
        # ====================================================================
        # SUMMARY
        # ====================================================================
        print("="*70)
        print("🎉 PIPELINE COMPLETE!")
        print("="*70)
        print(f"Input files:       {step1_result['total']}")
        print(f"Step 1 converted:  {step1_result['success']}")
        print(f"Step 2 converted:  {step2_result['converted']}")
        print(f"Final output:      {OUTPUT_DIR}")
        print(f"Format:            Pascal VOC (JSON bbox)")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}\n")
        
        # Cleanup on error
        if TEMP_DIR.exists():
            print(f"🧹 Cleaning up temporary folder: {TEMP_DIR}")
            shutil.rmtree(TEMP_DIR)
        
        raise


if __name__ == "__main__":
    main()