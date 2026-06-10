"""Test script to crop document pages from images using the trained U2-Net model.

Extracts page contours from U2-Net binary mask predictions, supports smoothing, erosion,
pixel-level cutout, and perspective warp for both single images and folders.
"""

import argparse
import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Ensure root path is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def order_points(pts: np.ndarray) -> np.ndarray:
    """Sort coordinates to: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    # top-left has the smallest sum, bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # top-right has the smallest difference, bottom-left has the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def crop_page(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Perform perspective warp to flatten the document page."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute the width of the new image
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    # Compute the height of the new image
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))
    return warped

def get_four_corners(contour: np.ndarray) -> np.ndarray | None:
    """Approximate a contour to exactly 4 corner points."""
    peri = cv2.arcLength(contour, True)
    for factor in np.linspace(0.01, 0.1, 100):
        approx = cv2.approxPolyDP(contour, factor * peri, True)
        if len(approx) == 4:
            return approx.squeeze(1)
            
    # Fallback: Minimum bounding box corners
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.intp(box)

def blend_with_white_bg(img: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
    """Overlay BGR image onto a white background based on binary mask."""
    mask_f = alpha_mask.astype(float) / 255.0
    if len(mask_f.shape) == 2:
        mask_f = mask_f[:, :, np.newaxis]
    white_bg = np.ones_like(img, dtype=np.uint8) * 255
    blended = (img.astype(float) * mask_f + white_bg.astype(float) * (1.0 - mask_f)).astype(np.uint8)
    return blended

def shrink_corners(pts: np.ndarray, factor: float) -> np.ndarray:
    """Shrink 4 corners towards their center by a percentage factor."""
    centroid = pts.mean(axis=0)
    shrunk = centroid + (1.0 - factor) * (pts - centroid)
    return shrunk.astype(pts.dtype)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input test image or directory")
    ap.add_argument("--ckpt", default="exported_models/u2netp_doc_final.pth", 
                    help="Path to trained U2-Net weight file (.pth)")
    ap.add_argument("--out-dir", default="ml2/results/u2net_crop", help="Output directory to save crops")
    ap.add_argument("--erode-iter", type=int, default=3, help="Number of erosion iterations")
    ap.add_argument("--erode-kernel", type=int, default=5, help="Size of erosion kernel")
    ap.add_argument("--shrink-factor", type=float, default=0.03, help="Fraction to shrink corners inward")
    ap.add_argument("--warp", action="store_true", help="Perform 4-corner perspective warp instead of cutout")
    ap.add_argument("--smooth-kernel", type=int, default=15, help="Gaussian blur kernel to smooth edges")
    ap.add_argument("--device", default="mps", help="Inference device (mps/cpu/cuda)")
    args = ap.parse_args()
    
    use_cutout = not args.warp
    
    # Check if custom model exists
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[ERROR] U2-Net checkpoint {ckpt_path} not found!")
        sys.exit(1)
        
    print(f"Initializing U2-Net detector from: {ckpt_path} on {args.device}...")
    from ml2.pipeline_integration.u2net_wrapper import U2NetDetector
    detector = U2NetDetector(ckpt=str(ckpt_path), device=args.device)
    
    img_input = args.image
    img_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    img_path = Path(img_input)
    
    if not img_path.exists():
        print(f"[ERROR] Image path or directory does not exist: {img_path}")
        sys.exit(1)
        
    if img_path.is_dir():
        image_files = [p for p in img_path.iterdir() if p.suffix in img_extensions]
        print(f"Processing directory: {img_path} ({len(image_files)} images found)")
    else:
        image_files = [img_path]
        
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for single_img_path in image_files:
        print(f"\nProcessing image: {single_img_path.name}")
        img = cv2.imread(str(single_img_path))
        if img is None:
            print(f"[WARNING] Could not read image: {single_img_path}")
            continue
            
        h, w = img.shape[:2]
        
        # Run prediction using U2-Net wrapper
        binary_mask = detector.detect(img)
        
        # Find contours of predicted mask to identify distinct document pages/parts
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to remove small noise regions (minimum 2% of total image area)
        min_area = 0.02 * (w * h)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if not valid_contours:
            print(f"[WARNING] No significant document regions detected in {single_img_path.name}")
            cv2.imwrite(str(out_dir / f"orig_{single_img_path.name}"), img)
            continue
            
        print(f"Detected {len(valid_contours)} document components.")
        
        # Sort contours horizontally (left to right)
        contour_data = []
        for cnt in valid_contours:
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            contour_data.append({'contour': cnt, 'cX': cX})
        contour_data.sort(key=lambda x: x['cX'])
        
        for i, c_data in enumerate(contour_data):
            cnt = c_data['contour']
            
            # Dynamic naming based on count
            if len(contour_data) == 2:
                logical_name = "left_page" if i == 0 else "right_page"
            elif len(contour_data) == 1:
                logical_name = "page"
            else:
                logical_name = f"page_part{i+1}"
                
            # Create a clean mask specifically for this contour component
            single_mask = np.zeros_like(binary_mask)
            cv2.drawContours(single_mask, [cnt], -1, 255, -1)
            
            # Smooth edges using Gaussian Blur and re-thresholding
            if args.smooth_kernel > 0:
                ksize = args.smooth_kernel if args.smooth_kernel % 2 == 1 else args.smooth_kernel + 1
                single_mask = cv2.GaussianBlur(single_mask, (ksize, ksize), 0)
                _, single_mask = cv2.threshold(single_mask, 127, 255, cv2.THRESH_BINARY)
                
            # Apply Erosion to shrink mask slightly inward
            if args.erode_iter > 0:
                kernel = np.ones((args.erode_kernel, args.erode_kernel), np.uint8)
                single_mask = cv2.erode(single_mask, kernel, iterations=args.erode_iter)
                
            if use_cutout:
                print(f" -> Creating pixel-level cutout for '{logical_name}'...")
                cutout_img = blend_with_white_bg(img, single_mask)
                
                # Crop to the bounding box of the processed mask
                x_b, y_b, w_b, h_b = cv2.boundingRect(single_mask)
                if w_b > 0 and h_b > 0:
                    cutout_img = cutout_img[y_b:y_b+h_b, x_b:x_b+w_b]
                
                # Draw detailed contour for visualization
                contours_sub, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                img_contour = img.copy()
                if contours_sub:
                    largest_cnt = max(contours_sub, key=cv2.contourArea)
                    cv2.drawContours(img_contour, [largest_cnt], -1, (0, 255, 0), 3)
                    
                out_name = f"cutout_{logical_name}_{single_img_path.stem}.jpg"
                cv2.imwrite(str(out_dir / out_name), cutout_img)
                print(f"[SAVED] Pixel cutout saved to: {out_dir / out_name}")
                
                if contours_sub:
                    img_contour_cropped = img_contour[y_b:y_b+h_b, x_b:x_b+w_b]
                    out_contour_name = f"visualized_cutout_{logical_name}_{single_img_path.stem}.jpg"
                    cv2.imwrite(str(out_dir / out_contour_name), img_contour_cropped)
                    print(f"[SAVED] Detailed contour saved to: {out_dir / out_contour_name}")
                continue
                
            # Warp (perspective transform) path
            corners = get_four_corners(cnt)
            if corners is not None and len(corners) == 4:
                if args.shrink_factor > 0.0:
                    corners = shrink_corners(corners, args.shrink_factor)
                
                print(f" -> Found page '{logical_name}', warping...")
                warped = crop_page(img, corners)
                
                # Draw corners for visualization
                img_vis = img.copy()
                for corner in corners:
                    cv2.circle(img_vis, tuple(corner), 10, (0, 0, 255), -1)
                cv2.polylines(img_vis, [corners], True, (0, 255, 0), 3)
                
                out_name = f"cropped_{logical_name}_{single_img_path.stem}.jpg"
                cv2.imwrite(str(out_dir / out_name), warped)
                print(f"[SAVED] Cropped output saved to: {out_dir / out_name}")
                
                cv2.imwrite(str(out_dir / f"visualized_{logical_name}_{single_img_path.name}"), img_vis)
            else:
                print(f"[WARNING] Could not approximate 4 corners for '{logical_name}'")

if __name__ == "__main__":
    main()
