# BÁO CÁO NGHIÊN CỨU KHOA HỌC
**Đề tài:** Xây Dựng Hệ Thống Tự Động Căn Chỉnh và Làm Rõ Nét Ảnh Chụp Tài Liệu Bằng Phương Pháp Lai (Hybrid Front-line Machine Learning & Computer Vision)

**Thực hiện:** Nhóm 6 
**Môn học:** Xử lý Ảnh và Video

---

## TÓM TẮT (ABSTRACT)
Việc số hóa tài liệu thông qua camera trên thiết bị di động ngày càng phổ biến nhưng phải đối mặt với nhiều rào cản vật lý khắc nghiệt như nhiễu ánh sáng, méo phối cảnh, giấy biến dạng lồi lõm và chất lượng camera kém. Các phương pháp xử lý ảnh tĩnh (Computer Vision) truyền thống bộc lộ nhiều điểm yếu khi xử lý trong môi trường không kiểm soát. Bài báo cáo này đề xuất một kiến trúc "Hybrid Pipeline" tiên tiến, trong đó sử dụng mô hình Học Sâu siêu nhẹ chuyên biệt **U²-Netp lite (1.19M params, 4.77MB)** được huấn luyện từ đầu (from scratch) làm lực lượng tiền trạm để phân vùng (Segmentation) tài liệu; kết hợp cùng mô hình **YOLOv11n-seg (2.9M params)** phục vụ phân đoạn đa nhiệm. Đánh giá thực nghiệm trên tập dữ liệu quy mô lớn (**25,507 ảnh** SmartDoc2-Images và kaggle_real) cho thấy phương pháp đề xuất đạt độ chính xác phân đoạn vượt trội với **mIoU 0.9902** và **Dice/F1-score 0.9951** trên tập kiểm thử 2,550 ảnh. Ở khâu cuối, hệ thống áp dụng kỹ thuật cân bằng ánh sáng độc lập trên 3 kênh màu (RGB Independent Shadow Normalization) và phơi sáng mềm (Soft Binarization), tạo ra các bản scan kỹ thuật số sắc bén, phông nền trắng đồng đều vô cực không thua kém máy quét chuyên dụng, với tốc độ xử lý đạt **73 FPS** trên phần cứng Apple Silicon (Metal Performance Shaders).

---

## 1. GIỚI THIỆU (INTRODUCTION)

Sự bùng nổ của các thiết bị di động thông minh đã biến camera điện thoại thành công cụ số hóa tài liệu tiện lợi nhất. Tuy nhiên, thay thế cơ chế chiếu sáng và mặt kính phẳng của máy quét quang học (Optical Scanner) bằng camera di động tạo ra vô số thách thức vật lý [1]. Dựa trên quan sát thực địa, nhóm nghiên cứu đã tổng hợp 4 nhóm rào cản chính:

1. **Giao thoa ánh sáng (Illumination Issues):** Bóng đổ không đều, đốm lóa mù chữ (Glare), nhiễu hạt ISO.
2. **Biến dạng hình học (Geometric Distortions):** Méo phối cảnh (Perspective), giấy bị quăn mép, hoặc biến dạng cuộn lồi lõm do nếp gấp gáy sách [2].
3. **Giới hạn quang học phần cứng:** Hiện tượng rung mờ tay (Motion Blur), nhiễu quang sai và nhiễu vi tần (Moire).
4. **Suy thoái vật lý bề mặt:** Giấy ố vàng, mực thấm mặt lưng (Bleed-through), nét mực phai mờ đứt đoạn.

Mục tiêu của nghiên cứu này là xây dựng một hệ thống phần mềm (Pipeline) có khả năng tự động cô lập tài liệu khỏi phông nền tự nhiên, phục hồi về phương diện hình học phẳng (2D) và tăng cường tương phản văn bản tới chất lượng của một bộ số hóa chuyên nghiệp.

---

## 2. TỔNG QUAN CÁC NGHIÊN CỨU LIÊN QUAN (RELATED WORKS)

### 2.1. Phương pháp thị giác máy tính truyền thống
Trong suốt nhiều thập kỷ, bài toán xén tài liệu (Document Localization) chủ yếu ưu tiên Trích xuất Cạnh (Edge Detection) như **Canny Algorithm (1986)** [3], hoặc xấp xỉ đường thẳng như **Hough Transform**. Tuy đạt tốc độ cao, các thuật toán này cực kỳ nhạy cảm với phông nền rườm rà (thảm cỏ, họa tiết bàn làm việc) và lập tức ném ra ngoại lệ (exception) khi giấy khuyết góc [4]. Để làm rõ nét chữ, các thuật toán binarization như **Otsu (1979)** [5] hay **Sauvola (2000)** [6] thường được ứng dụng. Otsu thất bại nặng nề trước gradient bóng râm, còn Sauvola gọt quá bén dẫn đến hiện tượng vỡ hạt mép chữ.

### 2.2. Kỷ nguyên Học sâu (Deep Learning) trong nhận diện Tài liệu
Sự xuất hiện của Mạng Neural Tích chập (CNN) đã tái định nghĩa bài toán phân rã tài liệu. Các mạng Phân đoạn thực thể (Instance/Semantic Segmentation) như **U-Net (Ronneberger et al., 2015)** [7] và mạng hướng đối tượng dư (Salient Object Detection) như **U²-Net (Qin et al., 2020)** [8] cho phép cô lập tờ giấy dựa trên đặc trưng ngữ nghĩa phức hợp (Semantic Features), bỏ qua hoàn toàn tiểu tiết nhiễu bẩn. Đối với bài toán là phẳng tài liệu, công trình **DocUNet (Ma et al., 2018)** [9] định hình lại khái niệm Nắn phẳng 3D bằng cách ánh xạ bề mặt lồi lõm vào một mặt phẳng tham số thay vì dùng Ma trận chiếu 3x3 thuần túy.

---

## 3. PHƯƠNG PHÁP ĐỀ XUẤT (PROPOSED HYBRID PIPELINE)

Hệ thống của nhóm đưa ra cấu trúc **Front-line Machine Learning**, chia làm 3 bước nối tiếp gắt gao.

### Bước 1: Trích xuất và Cô lập Vùng Tài Liệu (ML Detection)
Nhóm loại bỏ hoàn toàn các thuật toán trích xuất cạnh truyền thống dễ gãy vỡ (Canny, approxPolyDP) ở khâu tiền trạm và thay thế bằng 2 luồng Deep Learning chuyên biệt:
- **Trích xuất nền trực tiếp (U²-Netp lite):** Thiết kế và huấn luyện từ đầu (from scratch) kiến trúc **U²-Netp** siêu nhẹ (1.19 triệu tham số, kích thước 4.77 MB). Mô hình sử dụng các khối RSU (Residual U-block) lồng nhau giúp thu nhận đặc trưng đa tỉ lệ cực tốt. Thay thế cho thư viện `rembg` (176MB) cồng kềnh, mô hình được tối ưu hóa riêng cho tài liệu giấy trắng bằng **Combo Loss (BCE + soft Jaccard IoU + SSIM)** áp dụng cơ chế giám sát sâu (Deep Supervision) trên toàn bộ 7 đầu ra để tăng tốc hội tụ.
- **Phân đoạn đa nhiệm (YOLOv11n-seg):** Tích hợp song song mô hình **YOLOv11n-seg (2.9M params, 6MB)** fine-tune từ trọng số COCO để đồng thời phát hiện vùng biên (bounding box), đa giác phân đoạn và vẽ trực quan hóa kết quả đa đối tượng thời gian thực.

### Bước 2: Phục hồi Đẳng cấu (Geometric Dewarping) & Giám định IoU
Đối mặt với biến dạng góc lượn lõm:
- **Perspective Transform:** Ma trận chiếu hình $3\times3$ chuẩn hóa kéo mặt nghiêng về 1 mặt trực diện (`cv2.warpPerspective`).
- **Text-line Dewarping / UVDoc (AI):** Dựa trên luồng tư duy của **DocUNet** [9], hệ thống sử dụng mạng Neural (như UVDoc) phân tích bề mặt tài liệu, sinh ra lưới Warp dạng Spline Grid 3D để uốn đảo chiều các Pixel dềnh sóng. 
- **Chốt chặn Toán Học (Geometric IoU Anti-Pinch):** Để ngăn chặn rủi ro AI làm méo (pinch) các tài liệu vốn đã phẳng, thuật toán áp dụng đối chuẩn **Intersection over Union (IoU)**. Nếu viền cắt của mạng U²-Netp khớp >94% với đa giác 4 đỉnh lý tưởng, AI Dewarping sẽ bị chặn đứng để bảo tồn nội suy quang học nguyên thủy (Perspective Transform).

### Bước 3: Tăng cường và Làm mềm Tiêu chuẩn Kép (Endpoint CV Enhancement)
Sau khi ML đã lo cấu trúc dọn dẹp vật lý, Computer Vision can thiệp nhằm tinh chỉnh hóa quang học mức Cell-pixel:
1. **Khử rọi bóng Kênh Độc Lập (RGB-Independent Illumination Normalization):** Theo nghiên cứu của Gonzalez & Woods [10] về chiếu sáng không gian tĩnh, nhóm áp dụng toán tử hình thái học `MORPH_CLOSE` (kernel 21x21) để bào mòn nền chữ. Đột phá nằm ở chỗ: thay vì xử lý ảnh xám, thuật toán tách lọc độc lập 3 kênh Đỏ, Lục, Lam (RGB) để tạo ra 3 Background Maps ảo chứa dải Gradient bóng phân tán theo đúng Nhiệt độ màu thực tế. Thuật toán đem ma trận ảnh chia lại cho các Background Maps này, giúp trung hòa triệt để các tảng bóng tay, bóng điện thoại [1] mà không làm ám mờ (Color halos) mực xanh chữ ký.
2. **Khôi phục đốm Flash (Inpainting):** Áp dụng thuật toán nội suy Telea [11] (`cv2.inpaint`) điền đắp màu vá các lỗ rỗ do đèn Flash.
3. **Phơi sáng Mềm (Piecewise Linear Stretching / Soft Binarization):** 
   - Không áp dụng ngưỡng cắt đứt đoạn như Otsu. Phương pháp dùng kỹ thuật dãn biểu đồ tương phản tuyến tính mảnh (Piecewise Linear). 
   - $P_{out} = 0 \text{ nếu } P_{in} \le \text{Black Point}$
   - $P_{out} = 255 \text{ nếu } P_{in} \ge \text{White Point}$
   - Giữ lại quãng xám giữa để làm bước đệm chống nội suy răng cưa (Anti-aliasing). Giúp nét mực đen thẳm, giấy tẩy trắng tới vô cực, nhưng sườn chữ mềm mại uốn nắn nguyên trạng.

---

## 4. THỰC NGHIỆM VÀ ĐÁNH GIÁ (EXPERIMENTS & EVALUATION)

### 4.1. Thiết lập Thực nghiệm & Dữ liệu
Quá trình huấn luyện mô hình được triển khai trực tiếp trên phần cứng **Mac Studio M4 Max 48GB** sử dụng backend **Metal Performance Shaders (MPS)** của PyTorch. 
*   **Tập dữ liệu:** Quy mô **25,507 ảnh chụp tài liệu thực tế** (SmartDoc2-Images và kaggle_real) chứa đầy đủ các thử thách như nghiêng phối cảnh, lóa đèn flash, và tay che góc.
*   **Split Chiến lược:** Chia 70/20/10 theo `background_id`/`document_id` để triệt tiêu hoàn toàn rủi ro rò rỉ dữ liệu giữa các frame hình. Tập huấn luyện (Train) gồm 17,918 ảnh, tập kiểm định (Val) gồm 5,039 ảnh, và tập kiểm thử (Test) gồm 2,550 ảnh.
*   **Tham số huấn luyện:** Huấn luyện trong 80 Epoch với kích thước ảnh $320\times320$, Adam Optimizer, Cosine Annealing learning rate (base 1e-3), 5 epoch warmup. Tổng thời gian huấn luyện U²-Netp lite đạt **13 giờ 25 phút**.

### 4.2. So sánh Định lượng trên Tập kiểm thử (N = 2,550)
Kết quả kiểm thử mô hình **U²-Netp lite** vượt xa các chỉ tiêu đề ra của Plan B:

| Tập dữ liệu | Chỉ số IoU | Chỉ số Dice (F1) | MAE | Boundary F1 (BF) | N (Số mẫu) | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SmartDoc** | **0.9907** | **0.9953** | 0.0007 | **0.9177** | 2,488 | Gần như hoàn hảo |
| **kaggle_real** | **0.9716** | **0.9856** | 0.0147 | **0.4717** | 62 | Ảnh điện thoại thực tế khó |
| **TỔNG KIỂM THỬ** | **0.9902** | **0.9951** | **0.0010** | **0.9069** | **2,550** | **Vượt vượt trội mọi KPI** |

*   **So với Target Plan B:** Độ trùng khớp mIoU đạt **0.9902** (vượt +19.3% so với target $\ge 0.83$), Dice đạt **0.9951** (vượt +14.4% so với target $\ge 0.87$), sai số pixel MAE chỉ **0.0010** (nhỏ hơn 50 lần so với target $<0.05$).
*   **Đánh giá Tổng quát hóa ngoại miền (OOD):** Khi thử nghiệm trên tập dữ liệu **Doc3D** (4,520 ảnh giấy nhăn cong 3D phức tạp, mô hình *chưa từng được học*), mô hình đạt **mIoU 0.7302** và **Dice 0.8032**, chứng minh khả năng tự thích ứng cấu trúc hình học cực kỳ ấn tượng.

### 4.3. Đánh giá Hiệu năng & Tốc độ (Speed Benchmark)
Đo lường trực tiếp trên Mac Studio M4 Max cho thấy bước đột phá về hiệu năng:
*   Mô hình **U²-Netp lite đạt tốc độ 73.0 FPS** (latency trung bình **13.7 ms**) khi chạy trên GPU MPS, nhanh gấp **3.6 lần** so với mục tiêu realtime đặt ra ($\ge 20$ FPS).
*   **So với Baseline rembg cũ:** Kích thước mô hình **giảm 37 lần** (từ 176 MB xuống 4.77 MB), tốc độ chạy **nhanh gấp 9.1 lần** (từ 8 FPS lên 73 FPS) và độ chính xác mIoU **tăng +27%** (từ 0.78 lên 0.99).
*   Mô hình YOLOv11n-seg đã huấn luyện thành công đạt tốc độ 117 FPS trên MPS với file trọng số siêu nhỏ (5.98MB).

### 4.4. Đánh giá Định tính (Chất lượng hình ảnh đầu ra)
Khi kết hợp mô hình phân vùng Deep Learning ở Bước 1 và khâu hậu kỳ xử lý ảnh ở Bước 3, hệ thống đạt chất lượng scan cấp độ thương mại:
- **Khử bóng đổ triệt để:** Nhờ cơ chế RGB-Independent Normalization, toàn bộ bóng đổ thẫm của bàn tay chụp bị triệt tiêu hoàn toàn, phông nền giấy trắng tinh khiết đồng đều vô cực 100%.
- **Bảo toàn màu sắc:** Trái ngược với việc nhị phân hóa cứng làm mất màu mực, phương pháp lai bảo vệ hoàn chỉnh dấu mộc đỏ chói và chữ ký mực xanh nguyên bản. Viền nét chữ mềm mại, hoàn toàn không bị răng cưa nhờ kỹ thuật soft binarization.

---

## 5. KẾT LUẬN VÀ HƯỚNG MỞ RỘNG (CONCLUSION)

Bài báo cáo đã phác thảo và nghiệm chứng độ ưu việt của mô hình **Hybrid Front-line ML Document Scanner**. Việc thay thế hoàn toàn mạng dò bắt góc cổ điển và nhị phân hóa cục bộ bằng những mô hình Salient Segmentation AI (`U²-Net`, `DocAligner`) đã phá vỡ rào cản gãy vỡ mặt học hình học. Chốt chặn Enhancement từ `OpenCV` tối ưu lại thẩm mĩ văn bản đến độ hoàn hảo. 

**Hướng mở rộng trong tương lai:**
Giải pháp sắp tới nhắm đến bài toán **Document Layout Analysis**, bổ sung AI quét bóc những dải hình minh họa màu (Figure/Picture) riêng biệt ra khỏi trang giấy nhằm bảo vệ không cho chạy qua màng lọc phơi sáng trắng đen. Ngoài ra, việc tinh giản mạng nơ-ron thành tệp TFLite để xử lý tiết kiệm RAM (Edge Devices) ngay trên thiết bị Smartphone là khâu thiết yếu để thương mại hóa sản phẩm.

---

## TÀI LIỆU THAM KHẢO

1. B. Pham et al., "Phân tích và Khương cấu Hệ thống Máy quét thông minh," *Báo cáo Dự án Nhóm 6*, 2026.
2. S. Tian et al., "Physical challenges in mobile document capture," *ICDAR*, 2015.
3. J. Canny, "A Computational Approach to Edge Detection," *IEEE PAMI*, vol. 8, no. 6, pp. 679-698, 1986.
4. G. Bradski and A. Kaehler, *Learning OpenCV: Computer Vision with the OpenCV Library*. O'Reilly Media, 2008.
5. N. Otsu, "A Threshold Selection Method from Gray-Level Histograms," *IEEE TSMC*, vol. 9, no. 1, pp. 62-66, 1979.
6. J. Sauvola and M. Pietikäinen, "Adaptive document image binarization," *Pattern Recognition*, vol. 33, no. 2, pp. 225-236, 2000.
7. O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," *MICCAI*, 2015.
8. X. Qin et al., "U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection," *Pattern Recognition*, vol. 106, 107404, 2020.
9. K. Ma et al., "DocUNet: Document Image Unwarping via A Stacked U-Net," *CVPR*, 2018.
10. R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th Ed. Pearson, 2018.
11. A. Telea, "An Image Inpainting Technique Based on the Fast Marching Method," *Journal of Graphics Tools*, vol. 9, no. 1, pp. 23-34, 2004.
