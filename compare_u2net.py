"""
Script so sánh hiệu năng và chất lượng giữa:
1. U²-Net cũ (sử dụng thư viện rembg - Model 176MB tổng quát)
2. U²-Netp tự train (Model 4.77MB chuyên biệt cho Tài liệu)

Cách dùng:
    python compare_u2net.py --image "đường_dẫn_ảnh.jpg"
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

# Đảm bảo import được các module từ ml2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    # Thêm background mờ cho text
    cv2.rectangle(out, (5, 5), (min(img.shape[1] - 5, 380), 45), (0, 0, 0), -1)
    cv2.putText(out, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out

def main():
    parser = argparse.ArgumentParser(description="So sánh U2-Net cũ (rembg) và U2-Netp tự train")
    parser.add_argument("--image", type=str, default=None, help="Đường dẫn đến ảnh cần test")
    parser.add_argument("--ckpt", type=str, default="ml2/checkpoints/u2netp_doc.pth", help="Checkpoint model tự train")
    parser.add_argument("--device", type=str, default="mps", help="Device để chạy model tự train (mps/cpu/cuda)")
    args = parser.parse_args()

    # 1. Tìm ảnh test mặc định nếu không truyền vào
    image_path = args.image
    if not image_path:
        possible_paths = [
            "Pipeline With ML/test_final.jpg",
            "Pipeline With ML/test_final2.jpg",
        ]
        for p in possible_paths:
            if os.path.exists(p):
                image_path = p
                break
        
    if not image_path or not os.path.exists(image_path):
        print(f"❌ Không tìm thấy ảnh test! Vui lòng truyền đường dẫn qua --image")
        return

    print(f"📂 Đang đọc ảnh test: {image_path}")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("❌ Lỗi: Không thể đọc ảnh!")
        return
    
    h, w, c = img_bgr.shape
    print(f"📏 Kích thước ảnh gốc: {w}x{h} px")

    # --- PHẦN 1: RUN REMBG (U²-NET CŨ) ---
    print("\n--- 1. Đang chạy U²-Net cũ qua thư viện `rembg` (Tổng quát, 176MB)...")
    rembg_available = False
    rembg_time = 0.0
    rembg_mask = None
    rembg_doc = None
    
    try:
        from rembg import remove
        rembg_available = True
        
        # Warmup
        _ = remove(np.zeros((100, 100, 3), dtype=np.uint8))
        
        # Benchmark thực tế
        t_start = time.perf_counter()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        subject_orig = remove(img_rgb)
        rembg_time = (time.perf_counter() - t_start) * 1000  # ms
        
        # Lấy mask & tài liệu đã đục nền
        rembg_mask = subject_orig[:, :, 3]
        rembg_doc = blend_with_white_bg(img_bgr, rembg_mask)
        print(f"✅ U²-Net cũ (rembg) hoàn thành trong: {rembg_time:.2f} ms")
        
    except ImportError:
        print("⚠️ Cảnh báo: Chưa cài đặt thư viện 'rembg'. Vui lòng chạy 'pip install rembg' để so sánh.")
        print("👉 Sẽ bỏ qua phần so sánh trực quan của rembg.")

    # --- PHẦN 2: RUN CUSTOM U²-NETP (TỰ TRAIN) ---
    print(f"\n--- 2. Đang chạy U²-Netp tự train (Chuyên biệt tài liệu, 4.77MB) trên '{args.device}'...")
    custom_available = False
    custom_time = 0.0
    custom_mask = None
    custom_doc = None
    
    try:
        from ml2.pipeline_integration.u2net_wrapper import U2NetDetector
        
        # Kiểm tra file checkpoint có tồn tại không
        if not os.path.exists(args.ckpt):
            # Thử tìm file fallback khác
            fallback_ckpts = [
                "ml2/checkpoints/u2netp_doc_final.pth",
                "ml2/checkpoints/u2netp_main_best.pth",
                "ml2/checkpoints/u2netp_main_final.pth"
            ]
            for fb in fallback_ckpts:
                if os.path.exists(fb):
                    args.ckpt = fb
                    break
        
        if not os.path.exists(args.ckpt):
            raise FileNotFoundError(f"Không tìm thấy checkpoint tự train tại: {args.ckpt}")
            
        detector = U2NetDetector(ckpt=args.ckpt, device=args.device)
        custom_available = True
        
        # Warmup
        _ = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        
        # Benchmark thực tế
        t_start = time.perf_counter()
        custom_mask = detector.detect(img_bgr)
        custom_time = (time.perf_counter() - t_start) * 1000  # ms
        
        custom_doc = blend_with_white_bg(img_bgr, custom_mask)
        print(f"✅ U²-Netp tự train hoàn thành trong: {custom_time:.2f} ms")
        
    except Exception as e:
        print(f"❌ Lỗi khi chạy mô hình tự train: {e}")

    # --- PHẦN 3: BÁO CÁO & XUẤT ẢNH SO SÁNH ---
    print("\n" + "="*70)
    print("📋 BẢNG SO SÁNH HIỆU NĂNG INFRENCE")
    print("="*70)
    print(f"{'Mô hình':<25} | {'Kích thước file':<18} | {'Thời gian chạy (ms)':<20}")
    print("-"*70)
    
    if rembg_available:
        print(f"{'U²-Net cũ (rembg)':<25} | {'~176 MB':<18} | {f'{rembg_time:.2f} ms':<20}")
    else:
        print(f"{'U²-Net cũ (rembg)':<25} | {'~176 MB':<18} | {'N/A (Chưa cài rembg)':<20}")
        
    if custom_available:
        print(f"{'U²-Netp tự train':<25} | {'4.77 MB (Lite)':<18} | {f'{custom_time:.2f} ms':<20}")
        if rembg_available and custom_time > 0:
            speedup = rembg_time / custom_time
            print(f"\n🚀 Tốc độ: Model tự train nhanh gấp ~{speedup:.2f} lần so với model cũ!")
    else:
        print(f"{'U²-Netp tự train':<25} | {'4.77 MB (Lite)':<18} | {'N/A (Lỗi load checkpoint)':<20}")
    print("="*70)

    # Ghép ảnh so sánh bằng OpenCV
    panels = []
    
    # 1. Ảnh gốc
    img_resized = cv2.resize(img_bgr, (360, 480))
    panels.append(draw_label(img_resized, "1. Anh goc"))
    
    # 2. U2-Net cũ
    if rembg_available and rembg_mask is not None:
        mask_3ch = cv2.merge([rembg_mask, rembg_mask, rembg_mask])
        mask_res = cv2.resize(mask_3ch, (360, 480))
        doc_res = cv2.resize(rembg_doc, (360, 480))
        panels.append(draw_label(mask_res, "2. Mask rembg (U2-Net)"))
        panels.append(draw_label(doc_res, "3. Doc rembg (176MB)"))
        
    # 3. U2-Netp tự train
    if custom_available and custom_mask is not None:
        mask_3ch = cv2.merge([custom_mask, custom_mask, custom_mask])
        mask_res = cv2.resize(mask_3ch, (360, 480))
        doc_res = cv2.resize(custom_doc, (360, 480))
        panels.append(draw_label(mask_res, "4. Mask tu train"))
        panels.append(draw_label(doc_res, "5. Doc tu train (4.7MB)"))

    if len(panels) > 1:
        concat_img = np.hstack(panels)
        output_report = "u2net_comparison_result.png"
        cv2.imwrite(output_report, concat_img)
        print(f"\n🎁 Đã lưu ảnh so sánh trực quan chất lượng tại: {output_report}")
        print("👉 Bạn hãy mở file này ra xem để đánh giá chi tiết chất lượng tách viền tờ giấy!")
        print("💡 Ảnh gồm các cột: Ảnh gốc | Mask rembg | Doc rembg | Mask tự train | Doc tự train")

if __name__ == "__main__":
    main()
