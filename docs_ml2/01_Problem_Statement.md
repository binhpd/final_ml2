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
  - Sử dụng chiến lược **Data Blending 1-Class**: Gộp chung ảnh tài liệu phẳng và ảnh sách cong, gán chung 1 nhãn (`page`) để khắc phục lỗi Catastrophic Forgetting.
  - Lựa chọn **YOLOv11-seg** làm mô hình chính (do hỗ trợ đa lớp và tốc độ siêu việt).
  - Tích hợp kỹ thuật hậu xử lý (Post-processing) phân tích trọng tâm tọa độ X để tách biệt trái/phải, Cutout và Gaussian Blur (làm mịn đường bao).

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

### 2.2 Quy trình Xử lý Nhãn và Trộn Dữ Liệu (Data Blending Algorithm)
Ban đầu, chúng tôi thử nghiệm việc đổi nhãn thành `left_page` và `right_page`. Tuy nhiên, điều này gây ra lỗi "Catastrophic Forgetting" (quên hoàn toàn tài liệu phẳng) vì mô hình chỉ mải mê tìm kiếm nếp gấp của gáy sách.

Nhóm đã chuyển sang chiến lược **Data Blending V2**:
1. **Gộp Nhãn (1-Class unification):** Mọi tờ giấy (trái, phải, phẳng, cong) đều được gán chung nhãn `0: page`.
2. **Sub-sampling & Trộn (Blending):** Sử dụng script `prepare_tuning_v2_fast.py` để trộn tập ảnh gáy sách (spine_dataset) với tập ảnh phẳng (SmartDoc/Kaggle). 
3. **Phân biệt bằng Hậu xử lý (Post-processing):** Mô hình chỉ làm nhiệm vụ xuất ra các vùng mask `page` mượt nhất. Thuật toán Hậu xử lý sẽ tự động tính toán trọng tâm X (Centroid) của các mask. Mask bên trái trục màn hình là trang trái, mask bên phải là trang phải.

**Ý nghĩa thực tiễn:** Kỹ thuật này ép mạng nơ-ron YOLOv11-seg phải học khái niệm tổng quát của "một trang giấy" trong mọi hoàn cảnh, giúp giải quyết triệt để lỗi quên dữ liệu phẳng, đồng thời vẫn giữ được khả năng bóc tách gáy sách một cách tự nhiên.

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
| **Pha 2: Chuẩn bị Dữ liệu Data Blending V2** | **Đã hoàn thành** (Trộn dữ liệu phẳng và cong 1-Class) |
| **Pha 2: Tuning YOLOv11 (Khắc phục Forgetting)** | **Đã hoàn thành** (Model: `yolo_tuning_v2.pt`) |
| **Pha 2: Hậu xử lý (Smoothing + Cutout + Sort X)** | **Đã hoàn thành** (`test_crop.py` phân tích trục X) |
| **Đánh giá & Benchmark (Bảng 6 Models)** | **Đã hoàn thành** (Giải quyết triệt để lỗi "gai gai", lẹm gáy và forgetting) |

---

*Bài toán hiện tại hoàn toàn tập trung vào việc áp dụng mô hình phân đoạn kết hợp hậu xử lý thuật toán để tạo ra công cụ scan tài liệu di động chuyên nghiệp.*
