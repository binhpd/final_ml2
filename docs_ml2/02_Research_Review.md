# NGHIÊN CỨU & KHẢO SÁT LÝ THUYẾT

> **Mục đích:** Tóm tắt các nghiên cứu nền tảng dẫn đến quyết định lựa chọn mô hình và chiến lược giải quyết bài toán loại bỏ gáy sách.

---

## 1. Phân tích Các Kiến trúc Cơ bản (Giai đoạn 1)

Trong giai đoạn đầu của đồ án, nhóm đã đánh giá và huấn luyện 2 kiến trúc mạng phổ biến nhất cho bài toán tách đối tượng (Object Segmentation):

### 1.1 U²-Net (Salient Object Detection)
- **Bản chất Kiến trúc (Architecture):** U²-Net (U-Square Net) là một mạng nơ-ron được thiết kế đặc biệt cho bài toán phát hiện đối tượng nổi bật (Salient Object Detection). Nó kế thừa triết lý của mạng U-Net truyền thống (Encoder-Decoder với Skip Connections) nhưng ở một mức độ sâu sắc hơn.
- **Khối RSU (ReSidual U-block):** Đây là "trái tim" của U²-Net. Trong khi U-Net truyền thống sử dụng các lớp Convolution đơn giản ở mỗi tầng, U²-Net nhúng nguyên một mạng U-Net nhỏ gọn vào trong từng khối block. Thiết kế lồng nhau này giúp mạng trích xuất đặc trưng đa tỷ lệ (multi-scale) ngay từ các lớp nông mà không cần giảm quá sâu độ phân giải (downsampling). Điều này tránh được việc mất mát thông tin ngữ cảnh cục bộ.
- **So sánh với U-Net truyền thống:** U-Net gốc thường bị giới hạn về receptive field ở các lớp đầu tiên, do đó khó phân biệt các ranh giới mảnh hoặc các vùng tương phản thấp. U²-Net, nhờ cấu trúc RSU, xử lý chi tiết viền mép (boundary) xuất sắc hơn nhiều, tạo ra mask phân giải cao.
- **Ưu điểm thực tế:** 
  - Khả năng tách nền và làm sắc nét viền (pixel-level) là tuyệt hảo. 
  - Dung lượng bản U²-Netp (lite) cực nhẹ (~4.7MB), rất phù hợp để nhúng thẳng vào các ứng dụng Mobile App/Edge Devices chạy realtime mà không cần GPU mạnh.
- **Nhược điểm (Lý do phải chuyển sang YOLO cho GĐ2):** U²-Net chỉ có khả năng tạo ra một mặt nạ nhị phân (Binary Mask) chung cho toàn bộ "vật thể nổi bật" trong khung hình. Nó hoàn toàn "mù" trong việc phân loại đối tượng thành các lớp khác nhau (Ví dụ: không phân biệt được đâu là trang trái, đâu là trang phải) và không tách biệt được các đối tượng chồng chéo (Instance Segmentation).

### 1.2 YOLOv11-seg (Instance Segmentation)
- **Bản chất Kiến trúc:** Thuộc họ mạng YOLO nổi tiếng về tốc độ, nhưng YOLO-seg không chỉ vẽ hộp giới hạn (Bounding Box) mà còn có một nhánh Mask Head để phân đoạn thực thể (Instance Segmentation). 
- **Backbone & Module C2PSA:** Điểm đột phá lớn nhất của YOLOv11 so với các phiên bản trước là sự kết hợp của kiến trúc CSPDarknet cải tiến với khối C2PSA (Cross-Stage Partial Network with Spatial Attention). C2PSA tăng cường mạnh mẽ khả năng tập trung vào đặc trưng không gian (Spatial Attention), giúp mô hình "chú ý" tốt hơn vào các đường nét hình học như lề giấy, mép viền, ngay cả trong điều kiện ánh sáng cực kém.
- **So sánh với YOLOv8:** Theo công bố của Ultralytics, YOLOv11-seg (bản nano) sử dụng ít hơn tới 22% số lượng tham số so với YOLOv8n-seg, nhưng chỉ số mAP lại cao hơn. Việc giảm thiểu tham số nhưng tăng độ sâu tính toán giúp YOLOv11 hiệu quả hơn hẳn khi chạy trên các thiết bị giới hạn tài nguyên như điện thoại di động.
- **Ưu điểm thực tế:** 
  - Tốc độ suy luận (Inference Speed) nhanh vô đối, đạt >100 FPS trên Mac Studio M4 Max.
  - Sức mạnh cốt lõi cho bài toán gáy sách: Khả năng phân tách thực thể độc lập (Instance) và hỗ trợ đa nhãn (Multi-class). Điều này cho phép YOLO xuất ra 2 mask tách biệt cho `left_page` và `right_page`, mặc định loại trừ hoàn toàn các khu vực nằm ngoài (bao gồm cả background và gáy sách ở giữa).
- **Nhược điểm:** Do cơ chế nhánh Mask của YOLO thường xuất ra mặt nạ ở độ phân giải thấp hơn nhiều so với ảnh gốc (thường là 160x160), khi phóng to (upscale) mask này lên kích thước ảnh thực tế, đường viền mép sách thường bị hiện tượng "răng cưa" (Aliasing effect). Do đó, YOLOv11-seg **bắt buộc phải đi kèm với một bộ Post-processing mạnh mẽ** (Gaussian Blur + Thresholding) thì mới dùng được trong sản phẩm thương mại.

---

## 2. Thử thách Thực tế: Bài toán Gáy Sách (Spine)

### 2.1 Hiện trạng
Khi người dùng chụp ảnh hóa đơn hoặc giấy A4 đơn lẻ đặt trên bàn, cả U²-Net và YOLO đều hoạt động hoàn hảo. Tuy nhiên, khi ứng dụng vào thực tế học tập, sinh viên thường chụp tài liệu từ **sách giáo khoa, vở ghi chép, tạp chí đang mở**.

Lúc này, bức ảnh chứa 2 trang giấy liền nhau, nối bởi gáy sách (spine). 

### 2.2 Vấn đề gặp phải
- Các ứng dụng hoặc mô hình đơn giản thường nhận diện nhầm cả 2 trang sách là một tài liệu duy nhất (Bounding box bao trùm cả cuốn sách). 
- Kết quả khi dùng thuật toán bẻ thẳng (Perspective Transform) lên một cuốn sách cong 2 trang sẽ khiến toàn bộ chữ bị méo mó nghiêm trọng, mất tỷ lệ thực tế.
- Phần gáy sách (thường có bóng đổ đen, hoặc keo dán gáy) bị dính vào mép trong của ảnh cắt, gây mất thẩm mỹ.

### 2.3 Giải pháp: Tuning YOLOv11 với Nhãn Cấu Trúc
Để giải quyết triệt để, nhóm quyết định từ bỏ phương pháp nhận diện 1 class chung (document) và nâng cấp thành bài toán phân đoạn đa lớp có tính không gian.

Thay vì dùng U²-Net (không hỗ trợ đa lớp tốt), nhóm quyết định sử dụng **YOLOv11-seg** làm mô hình chính yếu cho Giai đoạn 2.
- **Cấu trúc lại Nhãn (Relabeling):** Xây dựng bộ dataset mới trong đó đối tượng trang giấy được phân làm 2 lớp:
  - left_page (Trang giấy nằm bên trái gáy sách ảo).
  - right_page (Trang giấy nằm bên phải gáy sách ảo).
- **Loại trừ Gáy Sách (Spine Exclusion):** Khi mô hình học được đâu là mép bên phải của left_page và đâu là mép bên trái của right_page, khoảng trống ở giữa (chính là gáy sách) sẽ bị loại bỏ khỏi mặt nạ (mask) nhận diện một cách tự nhiên.

---

## 3. Các Nghiên cứu Hậu Xử Lý (Post-processing)

Như đã phân tích ở Mục 1, YOLOv11-seg gặp điểm yếu là biên mask hay bị gai gai (răng cưa). Để sản phẩm đạt chất lượng Production, chúng tôi đã nghiên cứu và triển khai các thuật toán xử lý ảnh truyền thống (Computer Vision) để khắc phục:

1. **Gaussian Blur (Làm mờ Gauss):**
   Mặt nạ nhị phân xuất ra từ YOLO được đưa qua một bộ lọc mờ Gaussian. Bộ lọc này có tác dụng nội suy (interpolate) các điểm ảnh ở mép răng cưa, tạo ra một sự chuyển tiếp gradient mềm mại từ đen (nền) sang trắng (tài liệu). 
   *Tham số cấu hình:* smooth-kernel (Kích thước ma trận lọc, ví dụ 15 x 15).

2. **Otsu / Fixed Thresholding (Tái nhị phân hóa):**
   Sau khi đã làm mờ, mặt nạ trở thành dạng grayscale (ảnh xám). Bằng cách áp dụng lại một ngưỡng (threshold) lên ảnh xám này, chúng ta thu được một đường bao (contour) nhị phân hoàn toàn mới, loại bỏ sạch hoàn toàn các răng cưa vuông vức ban đầu, tạo ra đường cong tự nhiên bám sát mép giấy thật.

3. **Cutout Generation:**
   Sử dụng mask đã làm mịn làm kênh Alpha (Độ trong suốt) ghép vào ảnh màu gốc. Kết quả thu được là một hình ảnh tài liệu uốn lượn tự nhiên, trên nền trắng/trong suốt, không còn bất kỳ chi tiết nền rác (background) hoặc gáy sách nào.
