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

def benchmark_spine_exclusion(device="mps"):
    print("\n" + "="*50)
    print(f"📊 BENCHMARK SPICE EXCLUSION (2-CLASS) ON {device.upper()}")
    print("="*50)
    
    weights = "exported_models/yolo11n_seg_spine_exclusion_best.pt"
    if not os.path.exists(weights):
        print(f"❌ Error: Weights not found at {weights}")
        return None
        
    from ultralytics import YOLO
    model = YOLO(weights)
    
    # Warmup
    model(np.zeros((640, 640, 3), np.uint8), device=device, verbose=False)
    
    test_img_dir = Path("ml2/data/tunning/Yolo Segmentation/split/images/test")
    test_label_dir = Path("ml2/data/tunning/Yolo Segmentation/split/labels/test")
    
    if not test_img_dir.exists():
        print(f"❌ Error: Test image directory not found at {test_img_dir}")
        return None
        
    img_paths = sorted([p for p in test_img_dir.glob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    print(f"Found {len(img_paths)} test images for Spine Exclusion evaluation.")
    
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
        pred_left, pred_right = predict_yolo_multiclass(model, img, device)
        times.append((time.perf_counter() - t_start) * 1000)
        
        # Calculate metrics (0..1 scale for metrics_pair)
        m_left = metrics_pair(pred_left / 255.0, gt_left / 255.0)
        m_right = metrics_pair(pred_right / 255.0, gt_right / 255.0)
        
        # Avg metrics
        m_avg = {
            "iou": (m_left["iou"] + m_right["iou"]) / 2,
            "dice": (m_left["dice"] + m_right["dice"]) / 2,
            "mae": (m_left["mae"] + m_right["mae"]) / 2,
            "bf": (m_left["bf"] + m_right["bf"]) / 2,
        }
        
        mets_left.append(m_left)
        mets_right.append(m_right)
        mets_avg.append(m_avg)
        
    # Aggregate
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
    
    print("\n--- results ---")
    print(f"Left Page  - IoU: {left_res['iou']:.4f}, Dice: {left_res['dice']:.4f}, BF: {left_res['bf']:.4f}")
    print(f"Right Page - IoU: {right_res['iou']:.4f}, Dice: {right_res['dice']:.4f}, BF: {right_res['bf']:.4f}")
    print(f"Average    - IoU: {avg_res['iou']:.4f}, Dice: {avg_res['dice']:.4f}, BF: {avg_res['bf']:.4f}")
    print(f"Latency ({device.upper()}): {lat_ms:.2f} ms")
    
    # Save output
    out_path = Path("ml2/results/spine_exclusion_benchmark.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "iou", "dice", "mae", "bf", "latency_ms"])
        w.writerow(["left_page", left_res["iou"], left_res["dice"], left_res["mae"], left_res["bf"], lat_ms])
        w.writerow(["right_page", right_res["iou"], right_res["dice"], right_res["mae"], right_res["bf"], lat_ms])
        w.writerow(["average", avg_res["iou"], avg_res["dice"], avg_res["mae"], avg_res["bf"], lat_ms])
    print(f"Saved spine exclusion results to {out_path}")
    
    return {
        "left": left_res,
        "right": right_res,
        "avg": avg_res,
        "lat_ms": lat_ms
    }

def main():
    # 1. Chạy so sánh 5 model trên MPS
    print("🚀 Running 5-model benchmark on MPS...")
    run_cmd(f"{sys.executable} compare_5_models.py --device mps --limit 200 --out ml2/results/compare_5models_mps.csv")
    
    # 2. Chạy so sánh 5 model trên CPU
    print("🚀 Running 5-model benchmark on CPU...")
    run_cmd(f"{sys.executable} compare_5_models.py --device cpu --limit 200 --out ml2/results/compare_5models_cpu.csv")
    
    # 3. Chạy benchmark spine exclusion (2-class)
    benchmark_spine_exclusion(device="mps")
    
    print("\n✅ All benchmarks completed successfully!")

if __name__ == "__main__":
    main()
