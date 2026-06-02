import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml2.pipeline_integration.u2net_wrapper import U2NetDetector

def blend_with_white_bg(img, alpha_mask):
    """Ghép ảnh BGR lên nền trắng dựa vào alpha mask (0-255)."""
    mask_f = alpha_mask.astype(float) / 255.0
    if len(mask_f.shape) == 2:
        mask_f = mask_f[:, :, np.newaxis]
    white_bg = np.ones_like(img, dtype=np.uint8) * 255
    blended = (img.astype(float) * mask_f + white_bg.astype(float) * (1.0 - mask_f)).astype(np.uint8)
    return blended

def draw_label(img, text):
    """Vẽ nhãn văn bản lên góc trên của ảnh."""
    out = img.copy()
    cv2.rectangle(out, (5, 5), (min(img.shape[1] - 5, 380), 45), (0, 0, 0), -1)
    cv2.putText(out, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Đường dẫn đến ảnh hoặc thư mục ảnh")
    parser.add_argument("--yolo_ckpt", type=str, default="ml2/checkpoints/yolo11n_seg_doc.pt")
    parser.add_argument("--u2net_ckpt", type=str, default="ml2/checkpoints/u2netp_doc.pth")
    parser.add_argument("--u2net_full_ckpt", type=str, default="ml2/checkpoints/u2net_full.pth")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    # Collect images
    image_paths = []
    if os.path.isdir(args.image):
        for f in os.listdir(args.image):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('compare_'):
                image_paths.append(os.path.join(args.image, f))
    else:
        image_paths.append(args.image)
        
    if not image_paths:
        print(f"❌ Không tìm thấy ảnh test trong {args.image}")
        return

    # Load models once
    print("\n--- Khởi tạo Models ---")
    yolo_model = YOLO(args.yolo_ckpt)
    _ = yolo_model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    
    if not os.path.exists(args.u2net_ckpt):
        fallback_ckpts = [
            "ml2/checkpoints/u2netp_doc_final.pth",
            "ml2/checkpoints/u2netp_main_best.pth",
            "ml2/checkpoints/u2netp_main_final.pth"
        ]
        for fb in fallback_ckpts:
            if os.path.exists(fb):
                args.u2net_ckpt = fb
                break
    u2net_model = U2NetDetector(ckpt=args.u2net_ckpt, device=args.device, is_lite=True)
    _ = u2net_model.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    u2net_full_model = None
    if os.path.exists(args.u2net_full_ckpt):
        try:
            print("🚀 Đang tải mô hình U2Net Full...")
            u2net_full_model = U2NetDetector(ckpt=args.u2net_full_ckpt, device=args.device, is_lite=False)
            _ = u2net_full_model.detect(np.zeros((640, 640, 3), dtype=np.uint8))
            print("✅ Đã tải thành công U2Net Full.")
        except Exception as e:
            print(f"⚠️ Không thể tải U2Net Full (có thể do lỗi trọng số): {e}")
    else:
        print(f"⚠️ Không tìm thấy file {args.u2net_full_ckpt}. Bỏ qua U2Net Full.")

    print(f"Bắt đầu xử lý {len(image_paths)} ảnh...")
    
    for i, image_path in enumerate(sorted(image_paths)):
        print(f"\n[{i+1}/{len(image_paths)}] 📂 Đang đọc ảnh: {image_path}")
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print("❌ Lỗi: Không thể đọc ảnh!")
            continue
        
        h, w, c = img_bgr.shape
        print(f"📏 Kích thước ảnh gốc: {w}x{h} px")

        # --- 1. RUN YOLO ---
        t_start = time.perf_counter()
        results = yolo_model(img_bgr, device=args.device, verbose=False)
        yolo_time = (time.perf_counter() - t_start) * 1000
        
        yolo_mask_img = np.zeros((h, w), dtype=np.uint8)
        if results and results[0].masks is not None:
            mask_data = results[0].masks.data[0].cpu().numpy()
            mask_data = (mask_data * 255).astype(np.uint8)
            yolo_mask_img = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
        yolo_doc = blend_with_white_bg(img_bgr, yolo_mask_img)
        print(f"✅ YOLO hoàn thành trong: {yolo_time:.2f} ms")

        # --- 2. RUN U2NET ---
        t_start = time.perf_counter()
        u2net_mask_img = u2net_model.detect(img_bgr)
        u2net_time = (time.perf_counter() - t_start) * 1000
        u2net_doc = blend_with_white_bg(img_bgr, u2net_mask_img)
        print(f"✅ U2Net hoàn thành trong: {u2net_time:.2f} ms")

        # --- 3. RUN U2NET FULL ---
        u2net_full_mask_img = np.zeros((h, w), dtype=np.uint8)
        u2net_full_doc = np.zeros_like(img_bgr)
        if u2net_full_model is not None:
            t_start = time.perf_counter()
            u2net_full_mask_img = u2net_full_model.detect(img_bgr)
            u2net_full_time = (time.perf_counter() - t_start) * 1000
            u2net_full_doc = blend_with_white_bg(img_bgr, u2net_full_mask_img)
            print(f"✅ U2Net Full hoàn thành trong: {u2net_full_time:.2f} ms")

        # --- 4. GHÉP VÀ LƯU ẢNH ---
        target_w, target_h = 360, 480
        
        img_res = cv2.resize(img_bgr, (target_w, target_h))
        panel_1 = draw_label(img_res, "1. Anh goc")
        
        yolo_mask_3ch = cv2.merge([yolo_mask_img]*3)
        panel_2 = draw_label(cv2.resize(yolo_mask_3ch, (target_w, target_h)), "2. YOLO Mask")
        panel_3 = draw_label(cv2.resize(yolo_doc, (target_w, target_h)), "3. YOLO Cutout")
        
        u2net_mask_3ch = cv2.merge([u2net_mask_img]*3)
        panel_4 = draw_label(cv2.resize(u2net_mask_3ch, (target_w, target_h)), "4. U2Net Lite Mask")
        panel_5 = draw_label(cv2.resize(u2net_doc, (target_w, target_h)), "5. U2Net Lite Cutout")

        panels = [panel_1, panel_2, panel_3, panel_4, panel_5]

        if u2net_full_model is not None:
            u2net_full_mask_3ch = cv2.merge([u2net_full_mask_img]*3)
            panel_6 = draw_label(cv2.resize(u2net_full_mask_3ch, (target_w, target_h)), "6. U2Net Full Mask")
            panel_7 = draw_label(cv2.resize(u2net_full_doc, (target_w, target_h)), "7. U2Net Full Cutout")
            panels.extend([panel_6, panel_7])
        
        concat_img = np.hstack(panels)
        
        base_name = os.path.basename(image_path)
        out_name = f"compare_{os.path.splitext(base_name)[0]}.png"
        out_path = os.path.join(os.path.dirname(image_path), out_name)
        cv2.imwrite(out_path, concat_img)
        print(f"🎁 Đã lưu ảnh so sánh trực quan tại: {out_path}")

if __name__ == "__main__":
    main()
