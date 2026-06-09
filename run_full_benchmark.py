"""
Script chạy toàn bộ benchmark:
1. Chạy compare_5_models.py trên MPS và CPU.
2. Đánh giá định lượng model yolo_tuned (spine exclusion) trên tập test 2 lớp (17 ảnh).
"""
import os
import sys
import subprocess
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import csv

# Đảm bảo import được các module từ ml2
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml2.u2net.eval import metrics_pair
from compare_5_models import get_params_count

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error executing command: {res.stderr}")
    else:
        print(res.stdout)
    return res.returncode == 0

def load_yolo_polygon_masks(label_path, h, w):
    mask_left = np.zeros((h, w), dtype=np.uint8)
    mask_right = np.zeros((h, w), dtype=np.uint8)
    if not os.path.exists(label_path):
        return mask_left, mask_right
    
    with open(label_path, "r") as f:
        lines = f.read().splitlines()
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls_id = int(parts[0])
        coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
        # Convert to pixel space
        coords[:, 0] *= w
        coords[:, 1] *= h
        coords = coords.astype(np.int32)
        if cls_id == 0:
            cv2.fillPoly(mask_left, [coords], 255)
        elif cls_id == 1:
            cv2.fillPoly(mask_right, [coords], 255)
    return mask_left, mask_right

def predict_yolo_multiclass(model, img, device):
    h, w = img.shape[:2]
    r = model.predict(img, device=device, verbose=False)
    pred_left = np.zeros((h, w), dtype=np.uint8)
    pred_right = np.zeros((h, w), dtype=np.uint8)
    
    if not r or r[0].masks is None or len(r[0].masks.data) == 0:
        return pred_left, pred_right
    
    masks = r[0].masks.data.cpu().numpy()
    cls = r[0].boxes.cls.cpu().numpy()
    
    left_masks = []
    right_masks = []
    for mask_i, cls_i in zip(masks, cls):
        cls_i = int(cls_i)
        if cls_i == 0:
            left_masks.append(mask_i)
        elif cls_i == 1:
            right_masks.append(mask_i)
            
    if left_masks:
        union_left = (np.sum(left_masks, axis=0) > 0.5).astype(np.uint8) * 255
        pred_left = cv2.resize(union_left, (w, h), interpolation=cv2.INTER_LINEAR)
        pred_left = (pred_left > 127).astype(np.uint8) * 255
    if right_masks:
        union_right = (np.sum(right_masks, axis=0) > 0.5).astype(np.uint8) * 255
        pred_right = cv2.resize(union_right, (w, h), interpolation=cv2.INTER_LINEAR)
        pred_right = (pred_right > 127).astype(np.uint8) * 255
        
    return pred_left, pred_right

def predict_yolo_v2_1class(model, img, device):
    h, w = img.shape[:2]
    r = model.predict(img, device=device, verbose=False)
    pred_left = np.zeros((h, w), dtype=np.uint8)
    pred_right = np.zeros((h, w), dtype=np.uint8)
    
    if not r or r[0].masks is None or len(r[0].masks.data) == 0:
        return pred_left, pred_right
    
    masks = r[0].masks.data.cpu().numpy()
    masks_resized = []
    for mask in masks:
        m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        masks_resized.append((m > 0.5).astype(np.uint8) * 255)
        
    if len(masks_resized) == 1:
        # Only 1 mask found, assume it's left page for benchmark
        return masks_resized[0], pred_right
        
    # Sort masks by their center X
    centers = []
    for m in masks_resized:
        M = cv2.moments(m)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        centers.append(cx)
        
    sorted_indices = np.argsort(centers)
    left_m = masks_resized[sorted_indices[0]]
    right_m = masks_resized[sorted_indices[-1]]
    return left_m, right_m

def benchmark_spine_exclusion(device="mps"):
    print("\n" + "="*50)
    print(f"📊 BENCHMARK SPICE EXCLUSION ON {device.upper()}")
    print("="*50)
    
    models_to_test = [
        ("yolo_tuned", "exported_models/yolo11n_seg_spine_exclusion_best.pt", predict_yolo_multiclass),
        ("yolo_tuning_v2", "exported_models/yolo_tuning_v2.pt", predict_yolo_v2_1class)
    ]
    
    test_img_dir = Path("ml2/data/tunning/Yolo Segmentation/split/images/test")
    test_label_dir = Path("ml2/data/tunning/Yolo Segmentation/split/labels/test")
    
    if not test_img_dir.exists():
        print(f"❌ Error: Test image directory not found at {test_img_dir}")
        return None
        
    img_paths = sorted([p for p in test_img_dir.glob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    print(f"Found {len(img_paths)} test images for Spine Exclusion evaluation.\n")
    
    from ultralytics import YOLO
    
    out_path = Path("ml2/results/spine_exclusion_benchmark.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        csv_w = csv.writer(f)
        csv_w.writerow(["model", "class", "iou", "dice", "mae", "bf", "latency_ms"])
        
        for model_name, weights, pred_func in models_to_test:
            if not os.path.exists(weights):
                print(f"❌ Error: Weights not found at {weights}")
                continue
                
            print(f">>> Benchmarking {model_name}...")
            model = YOLO(weights)
            # Warmup
            model(np.zeros((640, 640, 3), np.uint8), device=device, verbose=False)
            
            mets_left, mets_right, mets_avg = [], [], []
            times = []
            
            for img_p in img_paths:
                img = cv2.imread(str(img_p))
                h, w = img.shape[:2]
                
                # Ground Truth
                label_p = test_label_dir / f"{img_p.stem}.txt"
                gt_left, gt_right = load_yolo_polygon_masks(str(label_p), h, w)
                
                # Inference
                t_start = time.perf_counter()
                pred_left, pred_right = pred_func(model, img, device)
                times.append((time.perf_counter() - t_start) * 1000)
                
                # Calculate metrics
                m_left = metrics_pair(pred_left / 255.0, gt_left / 255.0)
                m_right = metrics_pair(pred_right / 255.0, gt_right / 255.0)
                
                m_avg = {
                    "iou": (m_left["iou"] + m_right["iou"]) / 2,
                    "dice": (m_left["dice"] + m_right["dice"]) / 2,
                    "mae": (m_left["mae"] + m_right["mae"]) / 2,
                    "bf": (m_left["bf"] + m_right["bf"]) / 2,
                }
                
                mets_left.append(m_left)
                mets_right.append(m_right)
                mets_avg.append(m_avg)
                
            def agg_list(lst):
                return {
                    "iou": float(np.mean([m["iou"] for m in lst])),
                    "dice": float(np.mean([m["dice"] for m in lst])),
                    "mae": float(np.mean([m["mae"] for m in lst])),
                    "bf": float(np.mean([m["bf"] for m in lst])),
                }
                
            left_res = agg_list(mets_left)
            right_res = agg_list(mets_right)
            avg_res = agg_list(mets_avg)
            lat_ms = float(np.median(times))
            
            print(f"[{model_name}] Average - IoU: {avg_res['iou']:.4f}, Dice: {avg_res['dice']:.4f}, lat: {lat_ms:.2f}ms")
            
            csv_w.writerow([model_name, "left_page", left_res["iou"], left_res["dice"], left_res["mae"], left_res["bf"], lat_ms])
            csv_w.writerow([model_name, "right_page", right_res["iou"], right_res["dice"], right_res["mae"], right_res["bf"], lat_ms])
            csv_w.writerow([model_name, "average", avg_res["iou"], avg_res["dice"], avg_res["mae"], avg_res["bf"], lat_ms])
    
    print(f"\nSaved spine exclusion results to {out_path}")

def main():
    # 1. Chạy so sánh 5 model trên MPS (Chỉ dùng ảnh thật: SmartDoc + Kaggle_real)
    print("🚀 Running 5-model benchmark on MPS (No Doc3D)...")
    run_cmd(f"{sys.executable} compare_5_models.py --roots ml2/data/smartdoc ml2/data/kaggle_real --device mps --limit 200 --out ml2/results/compare_5models_mps.csv")
    
    # 2. Chạy so sánh 5 model trên CPU (Chỉ dùng ảnh thật: SmartDoc + Kaggle_real)
    print("🚀 Running 5-model benchmark on CPU (No Doc3D)...")
    run_cmd(f"{sys.executable} compare_5_models.py --roots ml2/data/smartdoc ml2/data/kaggle_real --device cpu --limit 200 --out ml2/results/compare_5models_cpu.csv")
    
    # 3. Chạy benchmark spine exclusion (2-class)
    benchmark_spine_exclusion(device="mps")
    
    print("\n✅ All benchmarks completed successfully!")

if __name__ == "__main__":
    main()
