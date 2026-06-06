# PHÁT BIỂU BÀI TOÁN & MỤC TIÊU

## 1. Bối cảnh & Bài Toán

### 1.1 Bài toán
Bài toán đặt ra là phát hiện, phân đoạn và cắt chính xác vùng tài liệu văn bản (Document Segmentation) ra khỏi bối cảnh nền phức tạp bằng mô hình học sâu.

### 1.2 Sự phát triển của dự án (2 Giai đoạn)

Dự án ban đầu được xây dựng để giải quyết bài toán cơ bản, sau đó nâng cấp lên bài toán nâng cao để giải quyết triệt để nhu cầu thực tế:

**Giai đoạn 1: Bài toán cắt trang giấy đơn cơ bản**
- **Mục tiêu:** Xây dựng và so sánh hiệu năng của 2 kiến trúc (U²-Netp và YOLOv11-seg) trong việc tách tài liệu.
- **Thực tế:** Cả 2 mô hình đều cắt tốt trang giấy rời. Tuy nhiên, khi người dùng chụp ảnh sách hoặc tạp chí mở (có 2 trang), mô hình thường nhận diện lẹm vào phần gáy sách (spine) hoặc gộp cả 2 trang thành 1, dẫn đến nội dung bị cong vênh, thừa viền đen, sai lệch khung hình.

**Giai đoạn 2: Nâng cao - Cắt chính xác trang giấy & Loại bỏ gáy sách (Trọng tâm hiện tại)**
- **Mục tiêu:** Nâng cấp giải pháp phân đoạn, giúp hệ thống không chỉ nhận diện được trang giấy mà còn **phân biệt được trang trái/phải**, từ đó **loại bỏ hoàn toàn phần gáy sách dư thừa**.
- **Giải pháp:** 
  - Gán nhãn lại tập dữ liệu thành các định dạng đa lớp (`left_page`, `right_page`).
  - Lựa chọn **YOLOv11-seg** làm mô hình chính (do hỗ trợ đa lớp tốt hơn U2Net).
  - Tích hợp kỹ thuật hậu xử lý (Post-processing) gồm: Cutout (xuất mask pixel-level trong suốt) và Gaussian Blur (làm mịn đường bao răng cưa).

---

## 2. Chiến lược Dữ liệu (Dataset Strategy)

Để mô hình có thể phân biệt và cắt bỏ gáy sách, chúng tôi đã phải xây dựng một chiến lược dữ liệu cực kỳ bài bản từ khâu thu thập, phân tích đặc trưng đến tái cấu trúc nhãn.

### 2.1 Đặc trưng các bộ dữ liệu được sử dụng
Nhóm không chỉ sử dụng một tập dữ liệu duy nhất mà kết hợp nhiều nguồn để đảm bảo tính tổng quát (Generalization):
1. **SmartDoc2-Images (24,887 ảnh):** 
   - *Mô tả:* Tập dữ liệu cốt lõi cung cấp bối cảnh thực tế rất đa dạng (văn bản A4, hóa đơn, tài liệu in). 
   - *Đặc điểm kỹ thuật:* Chứa nhiều góc chụp nghiêng (Perspective distortion) và các điều kiện ánh sáng trong phòng (bóng đèn neon, ánh sáng cửa sổ).
2. **kaggle_real (620 ảnh):** 
   - *Mô tả:* Tập ảnh thu thập trực tiếp từ camera điện thoại thông thường. 
   - *Đặc điểm kỹ thuật:* Chứa các nhiễu thực tế cực kỳ khó nhằn như chói sáng (glare), bóng đổ tay người chụp (shadows), nhiễu noise do chụp thiếu sáng.
3. **Doc3D (90,372 ảnh - OOD Test):** 
   - *Mô tả:* Tập dữ liệu được sinh bằng đồ họa máy tính (CGI).
   - *Đặc điểm kỹ thuật:* Mô phỏng chính xác sự biến dạng không gian vật lý 3D của tờ giấy: giấy bị nhăn nhúm, gấp nếp, cuộn tròn. Dùng để kiểm thử giới hạn (Out-Of-Distribution) của mô hình.

### 2.2 Quy trình Tái cấu trúc Nhãn (Relabeling Algorithm)
Trong dữ liệu gốc, mọi tờ giấy (dù là trang trái hay trang phải) đều được gán chung một nhãn là "document". Nếu đưa dữ liệu này vào huấn luyện, mô hình sẽ không hiểu được khái niệm "gáy sách ở giữa". 

Nhóm đã tự phát triển script `split_and_process_dataset.py` để tự động hóa việc chia tách nhãn với thuật toán như sau:
1. **Phân tích Tọa độ (Polygon Parsing):** Đọc tọa độ 4 góc của bounding box/polygon từ file nhãn gốc.
2. **Tính toán Trọng tâm (Centroid Calculation):** Thuật toán tính điểm trung tâm của mỗi tờ giấy.
3. **Xác định Gáy sách ảo (Virtual Spine Axis):** Nếu trong một ảnh phát hiện 2 tờ giấy, thuật toán sẽ vẽ một đường trục dọc (trục y) nằm ở giữa 2 trọng tâm đó (chính là gáy sách).
4. **Phân loại (Classification):** 
   - Tờ giấy có trọng tâm nằm bên trái trục gáy sách ảo được đổi tên nhãn thành `left_page`.
   - Tờ giấy có trọng tâm nằm bên phải được đổi tên thành `right_page`.

**Ý nghĩa thực tiễn:** Kỹ thuật này ép mạng nơ-ron YOLOv11-seg phải học được đặc trưng hình học bất đối xứng: "Trang bên trái thì mép bên phải của nó là gáy sách, và ngược lại". Kết quả là khi test thực tế, mô hình tự động "từ chối" không bao hàm phần gáy sách dính mực đen vào trong mặt nạ (mask) dự đoán.

## 3. Các tính năng đầu ra yêu cầu (KPI Đầu Ra)

Hệ thống sau khi nâng cấp phải đáp ứng được các tiêu chuẩn hình ảnh "chuẩn Pro":

1. **Cutout (Pixel-level Mask):** Ảnh đầu ra không phải là hình chữ nhật chứa nền, mà là đối tượng trang giấy được cắt uốn lượn chính xác theo đường cong thực tế, nền xung quanh là nền trong suốt/trắng.
2. **Smooth Contour:** Đường bao viền của trang giấy không được xuất hiện hiện tượng răng cưa ("gai gai" do aliasing của mạng phân giải thấp 160x160). Yêu cầu bắt buộc áp dụng Gaussian Blur kết hợp Thresholding.
3. **Spine Exclusion:** Cắt sát mép nội dung văn bản, không bao hàm vùng đóng gáy của sách.

---

## 4. Timeline & Tiến độ Thực tế (M4 Max 48GB)

| Giai đoạn | Trạng thái / Kết quả đạt được |
|---|---|
| **Pha 1: Train Baseline (U2Net vs YOLO)** | **Đã hoàn thành** (Train xong cả 2 mô hình) |
| **Pha 2: Chuẩn bị Dữ liệu Gáy Sách** | **Đã hoàn thành** (Viết script map `left_page`/`right_page`) |
| **Pha 2: Tuning YOLOv11** | **Đã hoàn thành** (Model: `yolo11n_seg_spine_exclusion_best.pt`) |
| **Pha 2: Hậu xử lý (Smoothing + Cutout)** | **Đã hoàn thành** (`test_crop.py` mặc định dùng `--cutout` và `--smooth-kernel 15`) |
| **Đánh giá & Benchmark** | **Đã hoàn thành** (Giải quyết triệt để lỗi "gai gai" và lẹm gáy) |

---

*Bài toán hiện tại hoàn toàn tập trung vào việc áp dụng mô hình phân đoạn kết hợp hậu xử lý thuật toán để tạo ra công cụ scan tài liệu di động chuyên nghiệp.*
