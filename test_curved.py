import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def main():
    # Paths
    img_dir = Path("image_testing/curved")
    out_dir = Path("image_testing/results_curved")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Models
    model_train_path = "exported_models/yolo11n_seg_doc.pt"
    model_tuned_path = "exported_models/yolo11n_seg_spine_exclusion_best.pt"
    
    print(f"Loading YOLO Train from {model_train_path}")
    model_train = YOLO(model_train_path)
    print(f"Loading YOLO Tuned v1 from {model_tuned_path}")
    model_tuned = YOLO(model_tuned_path)
    
    img_paths = sorted([p for p in img_dir.glob("*.jpg")])
    if not img_paths:
        print("No images found in", img_dir)
        return
        
    print(f"Found {len(img_paths)} images.")
    
    for img_p in img_paths:
        print(f"Processing {img_p.name}...")
        img = cv2.imread(str(img_p))
        if img is None:
            continue
            
        # Inference
        res_train = model_train.predict(img, verbose=False)
        res_tuned = model_tuned.predict(img, verbose=False)
        
        # Plot
        img_train = res_train[0].plot()
        img_tuned = res_tuned[0].plot()
        
        # Resize all to have the same height for concatenation (e.g. height 800)
        target_h = 800
        def resize_to_h(image, th):
            h, w = image.shape[:2]
            tw = int(w * (th / h))
            return cv2.resize(image, (tw, th))
            
        img_orig_r = resize_to_h(img, target_h)
        img_train_r = resize_to_h(img_train, target_h)
        img_tuned_r = resize_to_h(img_tuned, target_h)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_orig_r, "Original", (10, 50), font, 1.5, (0, 0, 255), 3)
        cv2.putText(img_train_r, "YOLO Train (1-class)", (10, 50), font, 1.5, (0, 0, 255), 3)
        cv2.putText(img_tuned_r, "YOLO Tuned v1 (2-class)", (10, 50), font, 1.5, (0, 0, 255), 3)
        
        # Concatenate horizontally
        concat_img = np.concatenate((img_orig_r, img_train_r, img_tuned_r), axis=1)
        
        # Save
        out_p = out_dir / f"compare_{img_p.name}"
        cv2.imwrite(str(out_p), concat_img)
        print(f"Saved {out_p}")

if __name__ == "__main__":
    main()
