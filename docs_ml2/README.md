# 📚 docs_ml2 — Tài liệu đồ án Phân đoạn Tài liệu & Loại bỏ Gáy Sách

> **Trạng thái:** ✅ Hoàn thiện Giai đoạn Nâng cao | M4 Max 48GB
> **Đồ án cuối kỳ ML2 — Nhóm 6**

---

## 🎯 Chủ đề Đồ Án
**Xây dựng mô hình AI để phân đoạn và cắt trang tài liệu từ ảnh chụp điện thoại.**

Dự án được chia làm 2 giai đoạn:
- **Giai đoạn 1 (Cơ bản):** So sánh hiệu năng giữa hai họ mạng U²-Net (tách nền pixel-level) và YOLOv11-seg (instance segmentation) trong bài toán cắt trang giấy đơn cơ bản.
- **Giai đoạn 2 (Nâng cao):** Giải quyết vấn đề thực tế khi chụp sách có dính gáy sách. Mô hình YOLO được huấn luyện nâng cao với dữ liệu gán nhãn chi tiết (`left_page`, `right_page`) kết hợp kỹ thuật hậu xử lý (Cutout & Contour Smoothing) để cắt chính xác nội dung trang, loại bỏ phần gáy sách dư thừa và làm mịn đường cắt.

---

## 🗂️ Cấu trúc thư mục

```
docs_ml2/
│
├── 📍 README.md                     ← Bạn đang đọc (master index)
├── 📖 01_Problem_Statement.md       ← Phát biểu bài toán, mục tiêu 2 giai đoạn và Data Split.
├── 🔬 02_Research_Review.md         ← Lịch sử nghiên cứu, ưu/nhược điểm U2Net vs YOLO.
├── 📋 03_Project_Plan.md            ← Kế hoạch triển khai & Quản lý rủi ro.
├── ⚙️ 04_Technical_Spec_and_KPI.md  ← Thông số kỹ thuật, cấu trúc mã nguồn, hậu xử lý và KPI.
├── 🛠️ 05_Training_Guide.md          ← Hướng dẫn chạy code, chuẩn bị data và huấn luyện mô hình.
└── 📊 06_Evaluation_Results.md      ← Bảng đánh giá chất lượng (mIoU, Benchmark) và phân tích lỗi.
```

---

## ⚡ TL;DR — 30 giây tóm tắt

**Bạn đang xây dựng gì:** Một hệ thống cắt trang tài liệu cực mịn từ ảnh chụp điện thoại (Cutout), với khả năng tự động phân biệt và loại bỏ gáy sách khi chụp tài liệu sách/tạp chí hai trang.

| Giai đoạn | Mô hình được chọn | Chức năng chính |
|-----------|-------------------|-----------------|
| **1. Cơ bản** | `U²-Netp lite` & `YOLOv11n-seg` | Cắt trang đơn (so sánh baseline) |
| **2. Nâng cao** | `YOLOv11n-seg (Tuned)` | Nhận diện riêng biệt trang trái/phải, **loại bỏ gáy sách**. Hậu xử lý làm mịn đường viền. |

**KPI đạt được:** Cắt chính xác sát mép trang giấy, loại bỏ hiện tượng viền "gai gai" bằng thuật toán Gaussian Blur + Thresholding, hỗ trợ mask pixel-level trong suốt (Cutout).

---

## 🚀 Bước tiếp theo để đọc tài liệu
Nếu bạn muốn tìm hiểu sâu về cách nhóm đã xây dựng tính năng loại bỏ gáy sách, hãy đọc theo thứ tự sau:
1. Đọc **[01_Problem_Statement.md](01_Problem_Statement.md)** để hiểu tại sao bài toán cũ chưa đáp ứng được thực tế.
2. Đọc **[04_Technical_Spec_and_KPI.md](04_Technical_Spec_and_KPI.md)** để xem thuật toán xử lý hậu kỳ (smoothing).
3. Đọc **[05_Training_Guide.md](05_Training_Guide.md)** để biết lệnh chạy xuất ảnh ra với viền cong tự nhiên.
