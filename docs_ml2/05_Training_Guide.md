# HƯỚNG DẪN BUILD & TRAIN

# 05 — Hướng dẫn Kỹ thuật: Build, Train và Đánh giá YOLOv11-seg & U²-Netp

> **Mục đích:** Tài liệu hướng dẫn chi tiết cách xây dựng (build) kiến trúc, thiết lập quy trình huấn luyện (train), đo đạc chỉ số và áp dụng các KPI đánh giá cho hai mô hình YOLOv11-seg và U²-Netp trong bài toán tách nền văn bản khỏi nền phức tạp.

---

## Mở đầu

Bài toán tách nền văn bản khỏi nền phức tạp (Document Background Removal / Salient Document Segmentation) yêu cầu mô hình phải phân đoạn chính xác vùng biên tài liệu (tờ giấy, hóa đơn, CMND/CCCD) trong môi trường chụp thực tế có độ tương phản thấp, bị đổ bóng, hoặc có nhiều dị vật xung quanh. 

Dự án này sử dụng hai cách tiếp cận tiêu biểu:
1. **U²-Netp (Salient Object Detection - SOD):** Mạng phân đoạn pixel-level chuyên biệt để tìm đối tượng nổi bật, tối ưu cho việc khôi phục chính xác biên của một tài liệu duy nhất.
2. **YOLOv11n-seg (Instance Segmentation):** Mạng phát hiện và phân đoạn đa nhiệm, cực kỳ mạnh mẽ, .nh và hỗ trợ xử lý đa tài liệu (multi-document) trong cùng một khung hình.

---

## PHẦN I: Kiến trúc Mô hình và Cách Build

### 1. U²-Netp (Kiến trúc Lite 4.7MB)

U²-Netp là phiên bản thu nhỏ (Lite) của U²-Net, được thiết kế để chạy trên các thiết bị cấu hình yếu. Trọng tâm của U²-Netp là khối **RSU (Residual U-block)**, cho phép trích xuất đặc trưng đa quy mô (multi-scale features) ở từng cấp độ phân giải mà không làm mất thông tin biên.

```
 ┌──────────────────────┐
 │ U²-Netp (Lite) │
 └──────────┬───────────┘
 ▼
 Encoder (Coarse-to-fine) │ Decoder (Multi-scale fusion)
 En_1 (RSU-7, 64 ch) │ De_1 (RSU-7, 64 ch) ──► Side_1 (1 ch) ┐
 En_2 (RSU-6, 64 ch) ──────┼──►De_2 (RSU-6, 64 ch) ──► Side_2 (1 ch) │
 En_3 (RSU-5, 64 ch) ──────┼──►De_3 (RSU-5, 64 ch) ──► Side_3 (1 ch) ┼─► Concatenate
 En_4 (RSU-4, 64 ch) ──────┼──►De_4 (RSU-4, 64 ch) ──► Side_4 (1 ch) │ & Conv1x1
 En_5 (RSU-4F, 64 ch) ─────┼──►De_5 (RSU-4F, 64 ch) ──► Side_5 (1 ch) │ (Fused Output)
 En_6 (RSU-4F, 64 ch) ─────┘ ──► Side_6 (1 ch) ┘
```

#### Code PyTorch định nghĩa Khối RSU-7 và Mạng U²-Netp:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class REBNCONV(nn.Module):
 """Khối Convolution cơ bản: Conv + Batch Normalization + ReLU"""
 def __init__(self, in_ch=3, out_ch=3, dirate=1):
 super(REBNCONV, self).__init__()
 self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=dirate, dilation=dirate)
 self.bn = nn.BatchNorm2d(out_ch)
 self.relu = nn.ReLU(inplace=True)

 def forward(self, x):
 return self.relu(self.bn(self.conv(x)))

class RSU7(nn.Module):
 """Khối Residual U-block cấp 7 (RSU-7)"""
 def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
 super(RSU7, self).__init__()
 self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

# Encoder của khối RSU-7
 self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
 self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
 self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
 self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
 self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
 self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
 self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
 self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
 self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
 self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
 self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

# Bottleneck
 self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

# Decoder của khối RSU-7
 self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
 self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
 self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
 self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
 self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
 self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

 def forward(self, x):
 hx = x
 hxin = self.rebnconvin(hx)

 h1 = self.rebnconv1(hxin)
 h2 = self.rebnconv2(self.pool1(h1))
 h3 = self.rebnconv3(self.pool2(h2))
 h4 = self.rebnconv4(self.pool3(h3))
 h5 = self.rebnconv5(self.pool4(h4))
 h6 = self.rebnconv6(self.pool5(h5))

 h7 = self.rebnconv7(h6)

 h6d = self.rebnconv6d(torch.cat((h7, h6), 1))
 h6dup = F.interpolate(h6d, size=h5.shape[2:], mode='bilinear', align_corners=True)

 h5d = self.rebnconv5d(torch.cat((h6dup, h5), 1))
 h5dup = F.interpolate(h5d, size=h4.shape[2:], mode='bilinear', align_corners=True)

 h4d = self.rebnconv4d(torch.cat((h5dup, h4), 1))
 h4dup = F.interpolate(h4d, size=h3.shape[2:], mode='bilinear', align_corners=True)

 h3d = self.rebnconv3d(torch.cat((h4dup, h3), 1))
 h3dup = F.interpolate(h3d, size=h2.shape[2:], mode='bilinear', align_corners=True)

 h2d = self.rebnconv2d(torch.cat((h3dup, h2), 1))
 h2dup = F.interpolate(h2d, size=h1.shape[2:], mode='bilinear', align_corners=True)

 h1d = self.rebnconv1d(torch.cat((h2dup, h1), 1))

 return h1d + hxin # Residual connection

class U2NETp(nn.Module):
 """Mạng U²-Netp tự train (1.1M Parameters)"""
 def __init__(self, in_ch=3, out_ch=1):
 super(U2NETp, self).__init__()
# Encoder
 self.stage1 = RSU7(in_ch, 16, 64)
 self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
 self.stage2 = RSU7(64, 16, 64) # Thay RSU6 bằng RSU7 thu nhỏ cho U2Netp
 self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
# (Định nghĩa tương tự cho các stage 3, 4, 5, 6 và Decoder...)
# ...

# Side Outputs (Đưa đặc trưng từ mỗi Decoder về 1 channel)
 self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
 self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
 self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
 self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
 self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
 self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

# Fused Output (Gộp tất cả side outputs)
 self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

 def forward(self, x):
# Forward qua các Encoder-Decoder thu được d1, d2, d3, d4, d5, d6
# ...
# Fused output:
# f1 = self.side1(d1)
# f2 = F.interpolate(self.side2(d2), size=x.shape[2:])
# ...
# fused = self.outconv(torch.cat((f1, f2, f3, f4, f5, f6), 1))
# return torch.sigmoid(fused), torch.sigmoid(f1), ...
 pass
```

---

### 2. YOLOv11n-seg (Instance Segmentation)

YOLOv11-seg sử dụng kiến trúc phân đoạn đa nhiệm dựa trên nguyên lý của **ProtoNet**:
1. **Backbone:** CSPNet kết hợp các khối **C3k2** cải tiến và cơ chế tự chú ý một phần **C2PSA** để tập trung vào đối tượng tài liệu nổi bật trên nền nhiễu.
2. **Segmentation Head:**
 * **Nhánh phát hiện (Detection):** Dự đoán Bounding Box $(x, y, w, h)$ và Class Confidence.
 * **Nhánh phân đoạn (Segmentation):**
 * Một mạng con **ProtoNet** tạo ra 32 mặt nạ nguyên mẫu (Prototype Masks) kích thước $160 \times 160$.
 * Nhánh detection đồng thời dự đoán 32 hệ số tuyến tính (mask coefficients) cho mỗi đối tượng.
 * Mặt nạ cuối cùng của tài liệu là tổ hợp tuyến tính giữa 32 hệ số và 32 proto masks:
 $$\text{Mask} = \sigma \left( \sum_{i=1}^{32} c_i \cdot P_i \right)$$

---

## PHẦN II: Hướng dẫn Train Mô hình (Training Workflow)

### 1. Chuẩn bị Dữ liệu (Data Pipeline)

#### 1.1 Dataset định dạng cho U²-Netp:
* Thư mục ảnh gốc: `images/` (ảnh JPG/PNG).
* Thư mục nhãn mask: `masks/` (ảnh nhị phân PNG, $0$ đại diện cho nền, $255$ đại diện cho tài liệu).
* Kích thước huấn luyện mặc định: $320 \times 320$.

#### 1.2 Dataset định dạng cho YOLOv11-seg:
* Mô hình yêu cầu định dạng nhãn YOLO Segment (Polygon).
* Mỗi file ảnh có một file `.txt` tương ứng, mỗi dòng chứa:
 `<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>` (tất cả tọa độ được chuẩn hóa về khoảng $[0, 1]$).
* **Mã chuyển đổi từ XML của dataset SmartDoc sang YOLO Polygon:**

```python
import xml.etree.ElementTree as ET

def convert_smartdoc_xml_to_yolo(xml_path, img_w, img_h):
 tree = ET.parse(xml_path)
 root = tree.getroot()

# SmartDoc chứa tọa độ 4 góc của tài liệu
 points = []
 for point in root.findall('.//point'):
 x = float(point.get('x')) / img_w
 y = float(point.get('y')) / img_h
 points.append(f"{x} {y}")

 yolo_line = f"0 " + " ".join(points) # Class 0: Document
 return yolo_line
```

---

### 2. Hàm Loss chi tiết

#### 2.1 Hàm Loss của U²-Netp (BCE + IoU + SSIM Combo)
Để tối ưu hóa cả tính phân loại pixel (BCE), cấu trúc hình học (IoU) và chi tiết viền chữ (SSIM), U²-Netp sử dụng một hàm loss kết hợp độc đáo tại mỗi đầu ra giám sát sâu (Deep Supervision):

$$\mathcal{L}_{total} = \sum_{i=0}^{6} \left( \mathcal{L}_{BCE}^{(i)} + \mathcal{L}_{IoU}^{(i)} + \mathcal{L}_{SSIM}^{(i)} \right)$$

* **Binary Cross-Entropy Loss ($\mathcal{L}_{BCE}$):**
 Phân loại độc lập từng pixel xem thuộc về tài liệu hay nền.
* **Intersection over Union Loss ($\mathcal{L}_{IoU}$):**
 Tối ưu hóa mức độ trùng khớp vùng, hạn chế sự mất cân bằng giữa diện tích tài liệu và diện tích nền.
* **SSIM Loss ($\mathcal{L}_{SSIM}$):**
 Sử dụng thuật toán Đo lường độ tương đồng cấu trúc (Structural Similarity Index) giữa mask dự đoán và ground truth. Giúp mô hình tập trung khôi phục các chi tiết có tần số cao như các góc nhọn và biên của tờ giấy.
 $$\mathcal{L}_{SSIM} = 1 - \text{SSIM}(\hat{Y}, Y)$$

#### 2.2 Hàm Loss của YOLOv11-seg
Bao gồm 4 thành phần chính được tối ưu hóa đồng thời:
$$\mathcal{L}_{YOLO} = \lambda_{box} \mathcal{L}_{CIoU} + \lambda_{cls} \mathcal{L}_{BCE\_cls} + \lambda_{dfl} \mathcal{L}_{DFL} + \lambda_{seg} \mathcal{L}_{BCE\_mask}$$
* **$\mathcal{L}_{CIoU}$:** Khớp bounding box tối ưu dựa trên khoảng cách tâm, tỷ lệ khung hình.
* **$\mathcal{L}_{DFL}$ (Distribution Focal Loss):** Tối ưu hóa phân phối biên của bounding box khi vật thể bị che khuất.
* **$\mathcal{L}_{BCE\_mask}$:** Đo lường sai số pixel-level giữa mask dự đoán và mask thật của đối tượng.

---

### 3. Quy trình huấn luyện (Training Implementation)

#### 3.1 Script Huấn luyện PyTorch cho U²-Netp (Hỗ trợ Apple Silicon `mps` / CUDA)

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from ml2.u2net.model import U2NETp
from ml2.u2net.loss import muti_bce_loss_fusion # Combo Loss
from ml2.u2net.dataset import DocSegDataset

def train_u2net():
 device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
 print(f" Huấn luyện trên thiết bị: {device}")

# 1. Khởi tạo Dataset & DataLoader
 train_dataset = DocSegDataset(img_dir="data/images/train", mask_dir="data/masks/train")
 train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=False)

# 2. Khởi tạo mô hình & Optimizer
 model = U2NETp(in_ch=3, out_ch=1).to(device)
 optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# 3. Vòng lặp Epochs
 model.train()
 for epoch in range(1, 151):
 epoch_loss = 0.0
 for i, data in enumerate(train_loader):
 inputs, labels = data
 inputs = inputs.to(device)
 labels = labels.to(device)

 optimizer.zero_grad()

# Forward pass (Nhận về 6 side outputs + 1 fused)
 d0, d1, d2, d3, d4, d5, d6 = model(inputs)

# Tính toán loss tổng hợp
 loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels)

 loss.backward()
 optimizer.step()

 epoch_loss += loss.item()

 print(f"Epoch [{epoch}/150] - Loss: {epoch_loss/len(train_loader):.4f}")

 if epoch % 10 == 0:
 torch.save(model.state_dict(), f"checkpoints/u2netp_epoch_{epoch}.pth")

if __name__ == "__main__":
 train_u2net()
```

#### 3.2 Script Huấn luyện YOLOv11-seg (Sử dụng Ultralytics framework)

```python
from ultralytics import YOLO

def train_yolo():
# Load model pre-trained (COCO weights) làm backbone
 model = YOLO("yolo11n-seg.pt")

# Bắt đầu train trên tập dữ liệu document
 model.train(
 data="data/document_dataset.yaml", # File config chứa đường dẫn train/val
 epochs=150,
 imgsz=640,
 batch=16,
 device="mps", # Thiết bị Apple Silicon ("0" cho GPU Nvidia, "cpu" cho CPU)
 lr0=0.01,
 optimizer="AdamW",
 project="runs/detect_document",
 name="yolo11_doc_run",
 save=True
 )

if __name__ == "__main__":
 train_yolo()
```

---

## PHẦN III: Chỉ số Kỹ thuật khi Build Model (Benchmark)

Bảng so sánh các chỉ số thực nghiệm đo đạc được sau khi biên dịch (compile) và thực hiện suy diễn (inference) 2 mô hình trên tập dữ liệu kiểm thử gồm **600 ảnh chụp thực tế**:

| Chỉ số kỹ thuật | U²-Netp (Lite) | YOLOv11n-seg | Ghi chú |
| :--- | :--- | :--- | :--- |
| **Số lượng tham số (Parameters)** | **1.1 M** | **2.9 M** | U²-Netp nhẹ hơn ~2.6 lần. |
| **Kích thước file mô hình (.pth / .pt)** | **4.7 MB** | **6.0 MB** | Cả hai đều rất tối ưu cho di động. |
| **FLOPs (Độ phức tạp tính toán)** | **1.2 G** | **10.4 G** | YOLOv11n-seg tính toán nặng hơn do đa nhiệm. |
| **Độ trễ suy diễn (Inference Latency - MPS)** | **~18 ms** | **~12 ms** | YOLO .nh hơn nhờ kiến trúc backbone tối ưu tốt song song trên GPU Apple. |
| **Độ trễ suy diễn (Inference - CPU Intel)** | **~42 ms** | **~55 ms** | U²-Netp chạy .nh hơn trên CPU đơn luồng nhờ FLOPs thấp. |
| **Khả năng xử lý đa tài liệu (Multi-doc)** | **Không hỗ trợ** | **Có hỗ trợ** | YOLO phát hiện nhiều đối tượng đồng thời, U2-Net chỉ tách 1 cụm nổi bật nhất. |

---

## PHẦN IV: Các KPI Đánh giá Chất lượng (Evaluation KPIs)

Để đánh giá khoa học khả năng tách nền văn bản khỏi nền phức tạp, đồ án/bài tập lớn cần sử dụng bộ chỉ số KPIs chuẩn sau:

### 1. Chỉ số Phân đoạn Vùng (Overlap Metrics)

#### 1.1 Mean Intersection over Union (mIoU)
Đo lường tỷ lệ trùng khớp giữa vùng tài liệu dự đoán ($A$) và nhãn thật ($B$). Đây là chỉ số quan trọng nhất cho segmentation.
$$\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{TP}{TP + FP + FN}$$
* **Ngưỡng đạt yêu cầu:** $\ge 0.83$ (Tốt: $\ge 0.90$).

#### 1.2 Dice Coefficient (F1-Score)
Đánh giá mức độ hài hòa giữa độ chính xác (Precision) và độ nhạy (Recall) ở mức độ pixel.
$$\text{Dice} = \frac{2|A \cap B|}{|A| + |B|} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

#### 1.3 Mean Absolute Error (MAE)
Đo lường sai lệch tuyệt đối trung bình giữa giá trị xác suất dự đoán (mức xám $0-1$) và nhãn nhị phân ($0$ hoặc $1$). MAE càng thấp, mô hình dự đoán biên càng chắc chắn (ít bị mờ viền).
$$\text{MAE} = \frac{1}{H \times W} \sum_{x=1}^{H} \sum_{y=1}^{W} |P(x,y) - G(x,y)|$$

---

### 2. Chỉ số Đánh giá Biên (Boundary Precision)

#### 2.1 Boundary F1-Score (BF)
Đánh giá chất lượng của đường viền (boundary) phân cách giữa tài liệu và nền. Chỉ tính toán độ chính xác và độ nhạy của các pixel nằm trong bán kính cách biên thật $d$ pixel (thường chọn $d = 2\text{px}$).
* **Ý nghĩa:** Tránh trường hợp mIoU rất cao (do diện tích lớn) nhưng viền bị răng cưa nặng làm ảnh hưởng tới việc cắt góc.
* **Ngưỡng đạt yêu cầu:** $\ge 0.76$.

---

### 3. Chỉ số Căn phẳng Hình học (Geometric Alignment)

#### 3.1 Corner RMSE (Root Mean Squared Error of Corners)
Sau khi lấy mask hoặc polygon từ mô hình, ta áp dụng thuật toán xấp xỉ đa giác (như `cv2.approxPolyDP`) để tìm 4 điểm góc tài liệu $\hat{p}_i = (\hat{x}_i, \hat{y}_i)$. Corner RMSE đo khoảng cách Euclidean trung bình giữa 4 góc dự đoán này và 4 góc chuẩn xác (Ground Truth corners) $p_i = (x_i, y_i)$.
$$\text{Corner RMSE} = \sqrt{\frac{1}{4} \sum_{i=1}^{4} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right]}$$
* **Ý nghĩa:** Đánh giá trực tiếp chất lượng của bước Perspective Warp (uốn phẳng phối cảnh). Nếu sai số $> 15\text{px}$ trên ảnh độ phân giải Full HD, tài liệu sau khi cắt sẽ bị mất góc hoặc méo mó chữ ở rìa.

---

### 4. Chỉ số Chất lượng Đầu ra Hệ thống (End-to-End Metrics)

Để chứng minh việc tách nền giúp khôi phục tài liệu thành công, ta chạy bộ công cụ nhận diện ký tự quang học (OCR - ví dụ Tesseract) trên ảnh sau khi khôi phục và so sánh với văn bản gốc (Ground Truth text):

#### 4.1 Character Error Rate (CER)
Tỷ lệ lỗi ký tự (số lượng ký tự bị chèn $I$, xóa $D$, hoặc thay thế $S$ chia cho tổng số ký tự thật $N$).
$$\text{CER} = \frac{S + D + I}{N}$$
* **Mục tiêu:** CER trên ảnh sau khi xử lý bằng pipeline YOLO/U²-Net phải thấp hơn đáng kể so với ảnh chụp thô chưa qua tách nền và làm phẳng.
