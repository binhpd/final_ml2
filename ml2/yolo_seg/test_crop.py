"""Test script to crop document pages from images using the trained YOLOv11n-seg model.

Extracts left_page (class 0) and right_page (class 1) masks, computes 4 corners,
and performs perspective warp.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

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
    # Try approximating with decreasing epsilon until we get 4 points
    peri = cv2.arcLength(contour, True)
    for factor in np.linspace(0.01, 0.1, 100):
        approx = cv2.approxPolyDP(contour, factor * peri, True)
        if len(approx) == 4:
            return approx.squeeze(1)
            
    # Fallback: Minimum bounding box corners
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)

def blend_with_white_bg(img: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
    """Overlay BGR image onto a white background based on binary mask."""
    mask_f = alpha_mask.astype(float) / 255.0
    if len(mask_f.shape) == 2:
        mask_f = mask_f[:, :, np.newaxis]
    white_bg = np.ones_like(img, dtype=np.uint8) * 255
    blended = (img.astype(float) * mask_f + white_bg.astype(float) * (1.0 - mask_f)).astype(np.uint8)
    return blended

def shrink_corners(pts: np.ndarray, factor: float) -> np.ndarray:
    """Shrink 4 corners towards their center by a percentage factor (e.g. 0.02 = 2% inward)."""
    centroid = pts.mean(axis=0)
    shrunk = centroid + (1.0 - factor) * (pts - centroid)
    return shrunk.astype(pts.dtype)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input test image")
    ap.add_argument("--model", default="ml2/checkpoints/yolo11n_seg_spine_exclusion_best.pt", 
                    help="Path to trained YOLOv11-seg weight file (.pt)")
    ap.add_argument("--out-dir", default="ml2/results/test_crop", help="Output directory to save crops")
    ap.add_argument("--show", action="store_true", help="Display the cropped and visualized images interactively")
    ap.add_argument("--no-save", action="store_true", help="Disable saving the output images to disk")
    ap.add_argument("--erode-iter", type=int, default=3, help="Number of erosion iterations to shrink the mask inward (for tight cropping)")
    ap.add_argument("--erode-kernel", type=int, default=5, help="Size of the kernel matrix for erosion")
    ap.add_argument("--shrink-factor", type=float, default=0.03, help="Fraction to shrink corners inward (e.g., 0.03 = 3% inward)")
    ap.add_argument("--warp", action="store_true", help="Perform 4-corner perspective warp (straight lines) instead of pixel cutout")
    ap.add_argument("--smooth-kernel", type=int, default=15, help="Size of Gaussian blur kernel to smooth jagged mask edges (set to 0 to disable)")
    args = ap.parse_args()
    
    # By default, use cutout (curved pixel mask) unless warp is explicitly requested
    use_cutout = not args.warp
    
    # Ensure dependencies are available
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics package not found. Please activate the virtual environment.")
        sys.exit(1)
        
    img_input = args.image
    is_url = img_input.startswith(("http://", "https://"))
    
    # Ensure model path exists
    model_path = Path(args.model)
    if not model_path.exists():
        fallback_path = Path("ml2/checkpoints/yolo11n_seg_spine_exclusion_best.pt")
        if fallback_path.exists():
            model_path = fallback_path
        else:
            print(f"[WARNING] Specified model weight {model_path} not found. Trying default smoke weight.")
            
    print(f"Loading YOLOv11-seg model from: {model_path}")
    model = YOLO(str(model_path))
    
    # Identify image paths or download url to process
    image_files = []
    url_image_data = None
    
    if is_url:
        import urllib.request
        print(f"Downloading image from URL: {img_input}")
        try:
            req = urllib.request.Request(img_input, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
                url_image_data = cv2.imdecode(arr, -1)
            # Create a mock path for the filename
            image_files = [Path("downloaded_url_image.jpg")]
        except Exception as e:
            print(f"[ERROR] Failed to download image from URL: {e}")
            sys.exit(1)
    else:
        img_path = Path(img_input)
        if not img_path.exists():
            print(f"[ERROR] Image/Directory path does not exist: {img_path}")
            sys.exit(1)
        img_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        if img_path.is_dir():
            image_files = [p for p in img_path.iterdir() if p.suffix in img_extensions]
            print(f"Processing directory: {img_path} ({len(image_files)} images found)")
        else:
            image_files = [img_path]
        
    out_dir = Path(args.out_dir)
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)
        
    for single_img_path in image_files:
        print(f"\nProcessing image: {single_img_path.name}")
        # Read image
        if is_url and url_image_data is not None:
            img = url_image_data
        else:
            img = cv2.imread(str(single_img_path))
            
        if img is None:
            print(f"[WARNING] Could not read image: {single_img_path}")
            continue
        h, w = img.shape[:2]
        # Run prediction
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        results = model.predict(img, imgsz=640, device=device)
        result = results[0]
        
        if result.masks is None:
            print(f"[WARNING] No segmentations detected in {single_img_path.name}")
            if not args.no_save:
                cv2.imwrite(str(out_dir / f"orig_{single_img_path.name}"), img)
            continue
            
        print(f"Detected {len(result.masks)} segmentations.")
        
        # Create a single copy of the original image to draw all detections
        img_visualization = img.copy()
        has_any_detection = False
        
        # Sort masks by X centroid if there are multiple masks (e.g. 2 pages)
        mask_data_list = []
        for idx, (mask_obj, box_obj) in enumerate(zip(result.masks, result.boxes)):
            conf = box_obj.conf[0].item()
            mask_arr = mask_obj.data[0].cpu().numpy()
            mask_resized = cv2.resize(mask_arr, (w, h))
            binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
            
            # Compute centroid X
            M = cv2.moments(binary_mask)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
            else:
                cX = 0
            
            mask_data_list.append({
                'binary_mask': binary_mask,
                'conf': conf,
                'cX': cX,
                'idx': idx
            })
            
        # Sort by X coordinate (Left to Right)
        mask_data_list.sort(key=lambda x: x['cX'])
        
        # Assign logical names based on sorting
        for i, m_data in enumerate(mask_data_list):
            binary_mask = m_data['binary_mask']
            conf = m_data['conf']
            
            # Dynamic naming
            if len(mask_data_list) == 2:
                logical_name = "left_page" if i == 0 else "right_page"
            elif len(mask_data_list) == 1:
                logical_name = "page"
            else:
                logical_name = f"page_part{i+1}"
            
            # Smooth jagged mask edges using Gaussian Blur and Re-thresholding
            if args.smooth_kernel > 0:
                ksize = args.smooth_kernel if args.smooth_kernel % 2 == 1 else args.smooth_kernel + 1
                binary_mask = cv2.GaussianBlur(binary_mask, (ksize, ksize), 0)
                _, binary_mask = cv2.threshold(binary_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Apply Erosion if requested to shrink mask inward for tight cropping
            if args.erode_iter > 0:
                kernel = np.ones((args.erode_kernel, args.erode_kernel), np.uint8)
                binary_mask = cv2.erode(binary_mask, kernel, iterations=args.erode_iter)
            
            if use_cutout:
                print(f" -> Creating pixel-level cutout for '{logical_name}' (conf: {conf:.2f})...")
                cutout_img = blend_with_white_bg(img, binary_mask)
                
                # Crop tightly to the bounding box of the mask
                x_b, y_b, w_b, h_b = cv2.boundingRect(binary_mask)
                if w_b > 0 and h_b > 0:
                    cutout_img = cutout_img[y_b:y_b+h_b, x_b:x_b+w_b]
                
                # Draw detailed contour on the global visualization image
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_cnt = max(contours, key=cv2.contourArea)
                    cv2.drawContours(img_visualization, [largest_cnt], -1, (0, 255, 0), 3) # Green line
                    # Write label on the visualization image
                    M = cv2.moments(largest_cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = 100, 100
                    cv2.putText(img_visualization, f"{logical_name} ({conf:.2f})", (cX - 50, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    has_any_detection = True
                
                if args.show:
                    # Resize window if cutout/contour is too large
                    display_w = 800
                    display_h = int(h_b * (800 / w_b)) if w_b > 0 else 800
                    cutout_small = cv2.resize(cutout_img, (display_w, display_h))
                    cv2.imshow(f"Pixel Cutout - {logical_name}", cutout_small)
                    print("Press any key on image window to continue...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                if not args.no_save:
                    out_name = f"cutout_{logical_name}_{single_img_path.stem}.jpg"
                    cv2.imwrite(str(out_dir / out_name), cutout_img)
                    print(f"[SAVED] Pixel cutout saved to: {out_dir / out_name}")
                continue

            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            largest_cnt = max(contours, key=cv2.contourArea)
            
            # Approximate 4 corners
            corners = get_four_corners(largest_cnt)
            if corners is not None and len(corners) == 4:
                # Apply shrink towards the center if requested
                if args.shrink_factor > 0.0:
                    corners = shrink_corners(corners, args.shrink_factor)
                
                print(f" -> Found page '{logical_name}' (conf: {conf:.2f}), warping...")
                warped = crop_page(img, corners)
                
                # Draw corners and polygon on global visualization image
                for corner in corners:
                    cv2.circle(img_visualization, tuple(corner), 10, (0, 0, 255), -1)
                cv2.polylines(img_visualization, [corners], True, (0, 255, 0), 3)
                
                M = cv2.moments(largest_cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 100, 100
                cv2.putText(img_visualization, f"{logical_name} ({conf:.2f})", (cX - 50, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                has_any_detection = True
                
                if args.show:
                    # Resize windows if images are too large for screen
                    display_w = 800
                    display_h = int(h * (800 / w))
                    img_vis_small = cv2.resize(img_visualization, (display_w, display_h))
                    cv2.imshow(f"Visualized Corners - {logical_name}", img_vis_small)
                    
                    # Warped image display
                    warped_h, warped_w = warped.shape[:2]
                    display_warped_w = 600
                    display_warped_h = int(warped_h * (600 / warped_w))
                    warped_small = cv2.resize(warped, (display_warped_w, display_warped_h))
                    cv2.imshow(f"Warped Page - {logical_name}", warped_small)
                    
                    print("Press any key on image window to continue...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                
                if not args.no_save:
                    # Save cropped output
                    out_name = f"cropped_{logical_name}_{single_img_path.stem}.jpg"
                    cv2.imwrite(str(out_dir / out_name), warped)
                    print(f"[SAVED] Cropped output saved to: {out_dir / out_name}")
            else:
                print(f"[WARNING] Could not approximate 4 corners for detected '{logical_name}'")
        
        # After processing all segments for the image, save the single, unified visualization
        if not args.no_save and has_any_detection:
            out_vis_name = f"visualized_{single_img_path.name}"
            cv2.imwrite(str(out_dir / out_vis_name), img_visualization)
            print(f"[SAVED] Overall detection visualization saved to: {out_dir / out_vis_name}")

if __name__ == "__main__":
    main()
