# Tóm Tắt Nhiệm Vụ & Hướng Đề Xuất Đồ Án Cuối Kỳ
## Môn học: Xử lý Ảnh và Video — Chủ đề: Khôi phục ảnh chụp tài liệu

Tài liệu này tổng hợp lại toàn bộ yêu cầu hiện tại của bạn cùng các hướng đề xuất đi sâu vào **Học sâu (Deep Learning - DL)** để làm bài tập cuối kỳ/đồ án môn học.

---

## 1. Tổng Kết Yêu Cầu Của Bạn
*   **Chủ đề dự án:** Khôi phục ảnh chụp tài liệu từ điện thoại di động (Document Image Restoration).
*   **Yêu cầu cốt lõi:**
    1.  Nghiên cứu kiến trúc mã nguồn và tài liệu hiện có trong dự án của nhóm.
    2.  Phân tích, bóc tách chi tiết các bước sử dụng xử lý ảnh truyền thống (CV) và Trí tuệ nhân tạo (ML/DL).
    3.  Đề xuất các bài toán **Xây mới (From Scratch)** và **Tối ưu hóa (Tuning)** chuyên sâu về Deep Learning để làm đồ án cuối kỳ.

---

## 2. Bản Đồ Phân Tách Công Nghệ Trong Pipeline Hiện Tại

Hệ thống của nhóm đang áp dụng mô hình **Cascading Hybrid Pipeline** (Lai ghép nhiều tầng) rất thông minh:

| Bước Xử Lý | Giải Pháp Xử Lý Ảnh Truyền Thống (CV) | Giải Pháp Trí Tuệ Nhân Tạo (ML/DL) | Đánh Giá Thực Tế |
| :--- | :--- | :--- | :--- |
| **Step 1: Định Vị & Bóc Nền** | Grayscale, Gaussian Blur, Canny Edge, `approxPolyDP`, Hough Lines | **YOLOv8-Seg**, **DocAligner** (mạng dự đoán heatmap góc), **U²-Net** (Rembg - bóc nền) | **ML/DL đóng vai trò then chốt:** Vượt qua mọi nhược điểm của Canny khi nền phức tạp hoặc ảnh bị che khuất góc. |
| **Step 2: Nắn Phẳng Hình Học** | Perspective Transform, Coons Patch (nội suy spline viền cong) | **page-dewarp** (Tối ưu hóa toán học), **UVDoc** (Dự đoán lưới Neural Grid 2D/3D) | **UVDoc (DL) là đỉnh cao:** Ước lượng độ cong vật lý 3D của trang sách và "ủi phẳng" hoàn hảo bằng nội suy bilinear. |
| **Step 3: Tăng Cường Bề Mặt** | Vá lóa (Inpaint), Khử nhòe rung (Unsharp Masking), Khử bóng 3 kênh màu RGB độc lập, CLAHE & Percentile Histogram | *Chưa áp dụng Deep Learning cho bước này.* | **OpenCV truyền thống chiếm ưu thế:** Xử lý pixel thời gian thực cực tốt, loại bỏ bóng râm và làm rõ nét chữ siêu nhanh. |

---

## 3. Ba Hướng Đề Xuất Đồ Án Chuyên Sâu (Deep Learning)

Bạn nên chọn **1 trong 3 hướng đi** dưới đây để làm đồ án cuối kỳ tùy thuộc vào mục tiêu và thế mạnh của nhóm:

### 🚀 HƯỚNG 1: Phân Vùng & Định Vị Tài Liệu Chuyên Dụng (Step 1 — Document Segmentation)
> *Phù hợp nếu bạn muốn giải pháp an toàn, dễ làm, có kết quả nhanh để tập trung tối ưu hóa hiệu năng.*

*   **Nhánh Xây Mới (From Scratch) — Điểm học thuật cao:**
    *   Tự thiết kế cấu trúc mạng **MobileNetV3-UNet** hoặc **SegNet** bằng PyTorch.
    *   Tự viết pipeline tiền xử lý và tăng cường dữ liệu (Albumentations) như tạo bóng giả lập, xoay nghiêng, thêm nhiễu hạt.
    *   Huấn luyện mô hình từ đầu trên tập dữ liệu mở (MIDV-500, SmartDoc) kết hợp hàm mất mát: $Loss = \text{BCEWithLogitsLoss} + \text{DiceLoss}$.
*   **Nhánh Tối Ưu (Tuning) — Điểm thực tiễn cao:**
    *   Fine-tune mô hình YOLOv8-Seg hoặc DocAligner trên tập dữ liệu hóa đơn/sách tiếng Việt thực tế tự chụp.
    *   Thử nghiệm benchmark sự cân bằng giữa **Độ chính xác (mAP)** và **Tốc độ (Inference Time)** ở các kích thước ảnh đầu vào khác nhau.
    *   Thực hiện **Lượng tử hóa mô hình (INT8 Quantization)** sang ONNX để chạy mượt mà trên CPU hoặc điện thoại di động.

---

### 🧠 HƯỚNG 2: Xử Lý Cong Vênh & Nắn Phẳng Bằng Học Sâu (Step 2 — Geometric Dewarping)
> *Hàm lượng học thuật cực cao, giải quyết bài toán khó nhất trong khôi phục tài liệu 3D, rất dễ đạt điểm 10 tuyệt đối.*

*   **Nhánh Xây Mới (From Scratch) — Sáng tạo giải thuật:**
    *   Xây dựng một mạng CNN gọn nhẹ (ví dụ ResNet18 backbone) chỉ dự đoán **Lưới biến dạng thô (Coarse Grid $8 \times 8$)** đại diện cho 64 điểm kiểm soát trên trang giấy cong.
    *   Sử dụng lớp toán học **Thin Plate Splines (TPS)** hoặc **B-Spline** để nội suy mịn các pixel còn lại. Phương pháp này giúp mô hình siêu nhẹ nhưng vẫn đạt độ chính xác cao.
*   **Nhánh Tối Ưu (Tuning) — Hoàn thiện công nghệ:**
    *   Nghiên cứu và khắc phục lỗi của lớp nội suy `grid_sample` trên chip M-series Apple Silicon cho mô hình **UVDoc** hiện tại của nhóm.
    *   Đo đạc chỉ số chất lượng ảnh phục dựng bằng các độ đo khoa học: **MS-SSIM** và **LD (Local Distortion)** so với ảnh gốc phẳng tuyệt đối (ground truth).

---

### 🎨 HƯỚNG 3: Khử Bóng Râm & Tăng Cường Ảnh Bằng AI (Step 3 — Generative Enhancement)
> *Tạo đột phá mới hoàn toàn cho dự án, mang lại kết quả demo thị giác (Wow-effect) ấn tượng nhất khi thuyết trình slide.*

*   **Nhánh Xây Mới (From Scratch) — Trải nghiệm mô hình sinh:**
    *   Xây dựng mô hình khử bóng đổ chuyên sâu bằng mạng sinh đối nghịch **Pix2Pix GAN** hoặc mạng **Attention U-Net** đơn giản.
    *   Mô hình nhận vào ảnh bị sấp bóng, lóa sáng và tự động tái tạo ra ảnh quét sạch sẽ, nền trắng phau nhưng nét chữ vẫn sắc sảo.
*   **Nhánh Tối Ưu (Tuning) — Giải pháp lai Hybrid:**
    *   Sử dụng một mô hình CNN cực nhỏ để dự đoán bản đồ bóng đổ (Shadow Mask), sau đó đưa vào thuật toán chia sáng RGB truyền thống để tối ưu hóa tốc độ xử lý thời gian thực mà vẫn giữ được độ rực rỡ của con dấu đỏ/chữ ký xanh.

---

## 4. Báo Cáo Chi Tiết
Toàn bộ phân tích sâu sắc về kiến trúc, cơ chế hoạt động của U²-Net, DocAligner, UVDoc cùng sơ đồ khối chi tiết đã được lưu trữ tại file:
👉 **[docs/document_restoration_ml_analysis.md](file:///Users/binhpham/.gemini/antigravity/brain/7629e224-9e46-445d-b239-77b3013024b4/document_restoration_ml_analysis.md)**

*Tài liệu này được tạo ra để đồng hành cùng bạn trong suốt quá trình bảo vệ đồ án cuối kỳ.*
