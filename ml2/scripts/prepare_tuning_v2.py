import os
import shutil
from pathlib import Path
import yaml

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def process_dataset(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, prefix):
    if not src_img_dir.exists() or not src_lbl_dir.exists():
        return
        
    for img_path in src_img_dir.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue
            
        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
            
        # Destination paths
        dst_img_path = dst_img_dir / f"{prefix}{img_path.name}"
        dst_lbl_path = dst_lbl_dir / f"{prefix}{img_path.stem}.txt"
        
        # Copy image
        shutil.copy2(img_path, dst_img_path)
        
        # Process and copy label (convert all classes to 0)
        with open(lbl_path, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            parts[0] = "0" # Force class 0
            new_lines.append(" ".join(parts) + "\n")
            
        with open(dst_lbl_path, "w") as f:
            f.writelines(new_lines)

def main():
    base_dir = Path("/Users/ntcnstudio01/Documents/binhpd2/Final/final_ml2/ml2/data")
    tuning_v2_dir = base_dir / "tuning_v2"
    
    # Define source directories
    doc_dir = base_dir / "yolo_doc"
    spine_dir = base_dir / "tunning" / "Yolo Segmentation" / "split"
    
    splits = ["train", "val", "test"]
    
    print(f"Creating blended dataset at: {tuning_v2_dir}")
    
    for split in splits:
        print(f"Processing split: {split}...")
        dst_img = tuning_v2_dir / "images" / split
        dst_lbl = tuning_v2_dir / "labels" / split
        ensure_dir(dst_img)
        ensure_dir(dst_lbl)
        
        # Process flat documents (yolo_doc)
        src_img_doc = doc_dir / "images" / split
        src_lbl_doc = doc_dir / "labels" / split
        process_dataset(src_img_doc, src_lbl_doc, dst_img, dst_lbl, "doc_")
        
        # Process book spines (tuning)
        src_img_spine = spine_dir / "images" / split
        src_lbl_spine = spine_dir / "labels" / split
        process_dataset(src_img_spine, src_lbl_spine, dst_img, dst_lbl, "spine_")
        
        # Print count
        count = len(list(dst_img.glob("*")))
        print(f"  -> Total images in {split}: {count}")

    # Create dataset.yaml
    yaml_content = {
        "path": str(tuning_v2_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": {0: "page"}
    }
    
    yaml_path = tuning_v2_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
        
    print(f"Created {yaml_path}")
    print("Data blending complete!")

if __name__ == "__main__":
    main()
