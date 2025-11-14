"""
Main Pipeline: SAM Binary Masks → YOLO Segmentation Labels

This script orchestrates the complete conversion pipeline:
  Step 3: Binary masks → JSON (per class)
  Step 4: Per-class JSON → Combined JSON
  Step 5: Combined JSON → YOLO Segmentation txt

Usage:
    python main_sam_to_yolo.py
"""

from pathlib import Path
import shutil
import step3_binary_to_json as step3
import step4_combine_json as step4
import step5_json_to_yolo_seg as step5

# ============================================================================
# CONFIGURATION - Edit these paths
# ============================================================================
SAM_OUTPUT_ROOT = Path("path/to/sam_output")      # Input: SAM binary masks
IMAGE_ROOT = Path("path/to/images")               # Input: Original images
OUTPUT_ROOT = Path("path/to/labels")              # Output: YOLO labels (txt + json)

# Temporary folders
TEMP_JSON_ROOT = Path("temp_labels_json")         # Step 3 output (per-class JSON)
TEMP_COMBINED_ROOT = Path("temp_labels_combined") # Step 4 output (combined JSON)

# Processing configuration
SUBSETS = ["train", "val"]
CLASSES = ["worker", "hardhat", "strap", "hook"]

# Set to True to keep temporary files for debugging
KEEP_TEMP_FILES = False

# ============================================================================


def main():
    """Main pipeline execution"""
    
    print("="*70)
    print("STARTING PIPELINE: SAM Binary → YOLO Segmentation")
    print("="*70)
    print(f"SAM output:  {SAM_OUTPUT_ROOT}")
    print(f"Images:      {IMAGE_ROOT}")
    print(f"Output:      {OUTPUT_ROOT}")
    print(f"Classes:     {CLASSES}")
    print("="*70 + "\n")
    
    try:
        # ====================================================================
        # STEP 3: Convert Binary Masks to JSON (per class)
        # ====================================================================
        print("🔄 Starting Step 3: Binary → JSON (per class)...\n")
        
        step3_result = step3.convert_binary_to_json(
            sam_output_root=SAM_OUTPUT_ROOT,
            image_root=IMAGE_ROOT,
            output_root=TEMP_JSON_ROOT,
            subsets=SUBSETS,
            classes=CLASSES,
            verbose=True
        )
        
        if step3_result["json_created"] == 0:
            print("❌ Step 3 failed: No JSON files created")
            return
        
        # ====================================================================
        # STEP 4: Combine Per-Class JSON Files
        # ====================================================================
        print("🔄 Starting Step 4: Combine JSON files...\n")
        
        step4_result = step4.combine_json_files(
            input_root=TEMP_JSON_ROOT,
            output_root=TEMP_COMBINED_ROOT,
            image_root=IMAGE_ROOT,
            subsets=SUBSETS,
            classes=CLASSES,
            verbose=True
        )
        
        if step4_result["merged"] == 0:
            print("❌ Step 4 failed: No JSON files merged")
            return
        
        # ====================================================================
        # STEP 5: Convert Combined JSON to YOLO Segmentation
        # ====================================================================
        print("🔄 Starting Step 5: JSON → YOLO Segmentation...\n")
        
        step5_result = step5.convert_json_to_yolo_seg(
            json_root=TEMP_COMBINED_ROOT,
            subsets=SUBSETS,
            classes=CLASSES,
            verbose=True
        )
        
        if step5_result["success"] == 0:
            print("❌ Step 5 failed: No txt files created")
            return
        
        # ====================================================================
        # MOVE FINAL OUTPUT
        # ====================================================================
        print("📦 Moving final output to destination...\n")
        
        # Move combined JSON and txt files to final output
        for subset in SUBSETS:
            src_dir = TEMP_COMBINED_ROOT / subset
            dst_dir = OUTPUT_ROOT / subset
            
            if src_dir.exists():
                dst_dir.mkdir(parents=True, exist_ok=True)
                
                # Move all JSON and txt files
                for file_path in src_dir.glob("*"):
                    if file_path.suffix in [".json", ".txt"]:
                        dst_path = dst_dir / file_path.name
                        shutil.copy2(file_path, dst_path)
        
        print(f"✅ Final output saved to: {OUTPUT_ROOT}\n")
        
        # ====================================================================
        # CLEANUP: Remove temporary files
        # ====================================================================
        if not KEEP_TEMP_FILES:
            print("🧹 Cleaning up temporary files...")
            
            if TEMP_JSON_ROOT.exists():
                shutil.rmtree(TEMP_JSON_ROOT)
                print(f"  ✅ Removed: {TEMP_JSON_ROOT}")
            
            if TEMP_COMBINED_ROOT.exists():
                shutil.rmtree(TEMP_COMBINED_ROOT)
                print(f"  ✅ Removed: {TEMP_COMBINED_ROOT}")
            
            print()
        else:
            print(f"📁 Temporary files kept:")
            print(f"  - {TEMP_JSON_ROOT}")
            print(f"  - {TEMP_COMBINED_ROOT}\n")
        
        # ====================================================================
        # SUMMARY
        # ====================================================================
        print("="*70)
        print("🎉 PIPELINE COMPLETE!")
        print("="*70)
        print(f"Step 3 - Binary groups:  {step3_result['total_groups']}")
        print(f"Step 3 - JSON created:   {step3_result['json_created']}")
        print(f"Step 4 - JSON combined:  {step4_result['merged']}")
        print(f"Step 5 - TXT created:    {step5_result['success']}")
        print(f"")
        print(f"Final output:            {OUTPUT_ROOT}")
        print(f"  ├── train/")
        print(f"  │   ├── img001.json    (LabelMe format)")
        print(f"  │   └── img001.txt     (YOLO segmentation)")
        print(f"  └── val/")
        print(f"      ├── img001.json")
        print(f"      └── img001.txt")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}\n")
        
        # Cleanup on error
        if not KEEP_TEMP_FILES:
            print("🧹 Cleaning up temporary folders...")
            if TEMP_JSON_ROOT.exists():
                shutil.rmtree(TEMP_JSON_ROOT)
            if TEMP_COMBINED_ROOT.exists():
                shutil.rmtree(TEMP_COMBINED_ROOT)
        
        raise


if __name__ == "__main__":
    main()