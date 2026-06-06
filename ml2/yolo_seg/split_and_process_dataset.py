"""Split and process YOLO segmentation dataset.

Splits the train dataset under ml2/data/tunning/Yolo Segmentation into:
- train (80%)
- val (15%)
- test (5%)

At the same time, maps class 0 to:
- class 0: left_page (smaller average X)
- class 1: right_page (larger average X)
"""

import os
import random
import shutil
from pathlib import Path

def get_avg_x(line: str) -> float:
    parts = line.strip().split()
    if not parts:
        return 0.0
    # The first element is class ID, subsequent elements are x1, y1, x2, y2...
    xs = [float(parts[i]) for i in range(1, len(parts), 2)]
    return sum(xs) / len(xs) if xs else 0.0

def process_label_line(line: str, new_class_id: int) -> str:
    parts = line.strip().split()
    if not parts:
        return ""
    parts[0] = str(new_class_id)
    return " ".join(parts) + "\n"

def main():
    random.seed(42)
    
    base_dir = Path("ml2/data/tunning/Yolo Segmentation")
    train_img_dir = base_dir / "train" / "images"
    train_lbl_dir = base_dir / "train" / "labels"
    
    output_dir = base_dir / "split"
    
    # Clean up output directory if it exists
    if output_dir.exists():
        print(f"Cleaning existing directory: {output_dir}")
        shutil.rmtree(output_dir)
        
    # Find all images
    img_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    all_images = [p for p in train_img_dir.iterdir() if p.suffix in img_extensions]
    
    pairs = []
    for img_path in all_images:
        # Check corresponding label file
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
            
    print(f"Found {len(pairs)} image-label pairs.")
    
    # Shuffle pairs
    random.shuffle(pairs)
    
    # Split: 80% train, 15% val, 5% test
    n = len(pairs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.15)
    
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:]
    }
    
    # Create directories and copy/process files
    for split_name, split_pairs in splits.items():
        print(f"Processing {split_name} split ({len(split_pairs)} pairs)...")
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        
        for img_path, lbl_path in split_pairs:
            # Copy image
            shutil.copy2(img_path, img_out / img_path.name)
            
            # Read and process label
            with open(lbl_path, "r") as f:
                lines = [line for line in f.read().splitlines() if line.strip()]
                
            if len(lines) != 2:
                print(f"Warning: {lbl_path.name} has {len(lines)} annotations, expected 2. Copying as class 0.")
                new_lines = []
                for line in lines:
                    new_lines.append(process_label_line(line, 0))
            else:
                # Determine which polygon is left and which is right
                avg_x0 = get_avg_x(lines[0])
                avg_x1 = get_avg_x(lines[1])
                
                if avg_x0 <= avg_x1:
                    new_lines = [
                        process_label_line(lines[0], 0),  # left_page
                        process_label_line(lines[1], 1)   # right_page
                    ]
                else:
                    new_lines = [
                        process_label_line(lines[0], 1),  # right_page
                        process_label_line(lines[1], 0)   # left_page
                    ]
                    
            # Write new label
            with open(lbl_out / lbl_path.name, "w") as f_out:
                f_out.writelines(new_lines)
                
    # Write data.yaml
    yaml_content = f"""path: {output_dir.resolve()}
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: left_page
  1: right_page
"""
    with open(output_dir / "data.yaml", "w") as f_yaml:
        f_yaml.write(yaml_content)
        
    print(f"Successfully split dataset! Config saved to: {output_dir / 'data.yaml'}")

if __name__ == "__main__":
    main()
