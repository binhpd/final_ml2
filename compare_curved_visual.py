import cv2
import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def overlay_mask(img, masks, color=(0, 255, 0), alpha=0.5):
    """Overlay yolo masks on image."""
    result = img.copy()
    if masks is None or len(masks) == 0:
        return result
        
    for mask in masks:
        # mask is (H, W) boolean or float
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        colored_mask = np.zeros_like(img, dtype=np.uint8)
        colored_mask[mask > 0.5] = color
        
        # Blending
        mask_indices = mask > 0.5
        result[mask_indices] = cv2.addWeighted(img[mask_indices], 1 - alpha, colored_mask[mask_indices], alpha, 0)
        
        # Draw contour
        contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)
        
    return result

def main():
    img_dir = Path("image_testing/curved")
    out_dir = Path("image_testing/curved_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_trained = YOLO("exported_models/yolo11n_seg_doc.pt")
    model_v2 = YOLO("exported_models/yolo_tuning_v2.pt")
    
    img_paths = sorted(img_dir.glob("*.jpg"))
    
    print(f"Testing {len(img_paths)} images from {img_dir}...")
    
    for p in img_paths:
        print(f"Processing {p.name}...")
        img = cv2.imread(str(p))
        if img is None:
            continue
            
        # 1. YOLO Trained
        res_trained = model_trained.predict(img, device="mps", verbose=False)[0]
        masks_trained = res_trained.masks.data.cpu().numpy() if res_trained.masks else None
        vis_trained = overlay_mask(img, masks_trained, color=(0, 0, 255)) # Red for old model
        
        # 2. YOLO Tuning V2
        res_v2 = model_v2.predict(img, device="mps", verbose=False)[0]
        masks_v2 = res_v2.masks.data.cpu().numpy() if res_v2.masks else None
        vis_v2 = overlay_mask(img, masks_v2, color=(0, 255, 0)) # Green for new model
        
        # Put text
        cv2.putText(vis_trained, "yolo_trained (Old)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
        cv2.putText(vis_v2, "yolo_tuning_v2 (New)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
        
        # Concatenate horizontally
        combined = np.hstack((vis_trained, vis_v2))
        
        # Resize for saving if it's too large (some images are 8MB, huge res)
        h, w = combined.shape[:2]
        max_w = 2000
        if w > max_w:
            scale = max_w / w
            combined = cv2.resize(combined, (int(w*scale), int(h*scale)))
            
        out_p = out_dir / f"compare_{p.name}"
        cv2.imwrite(str(out_p), combined)
        print(f"  -> Saved {out_p}")
        
    print("Done! Visual comparisons saved.")

if __name__ == "__main__":
    main()
