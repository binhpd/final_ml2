# So Sánh 4 Benchmark KPI — U²-Netp vs YOLOv11n-seg

> Nguồn: `ml2/results/kpi_{accuracy,speed,robustness,e2e}.csv` (tập test 7008 ảnh, Mac Studio M4 Max).
> rembg = baseline zero-shot (0 ở các bảng = không chạy được/không có nhãn).

## Benchmark 1 — Accuracy (toàn tập test, N=7008)

| Model | IoU ↑ | Dice ↑ | MAE ↓ | Boundary-F1 ↑ |
|---|:---:|:---:|:---:|:---:|
| U²-Netp | 0.7179 | 0.7847 | **0.1295** | 0.1344 |
| **YOLOv11n-seg** | **0.8013** | **0.8355** | 0.1401 | **0.2606** |

→ YOLO thắng IoU/Dice/Boundary-F1; U²-Net chỉ nhỉnh hơn MAE (sai số pixel trung bình).

## Benchmark 2 — Speed (median trên 100 lần chạy)

| Model | Device | Median (ms) ↓ | p95 (ms) | FPS ↑ |
|---|---|:---:|:---:|:---:|
| U²-Netp | CPU | 465.5 | 493.1 | 2.1 |
| U²-Netp | MPS | 62.5 | 68.6 | 16.0 |
| YOLOv11n-seg | CPU | **19.0** | 20.1 | **52.5** |
| YOLOv11n-seg | MPS | **8.2** | 8.7 | **121.6** |

→ YOLO nhanh hơn **~24× trên CPU**, ~7.6× trên MPS. Quyết định cho mobile: chỉ YOLO đạt realtime CPU.

## Benchmark 3 — Robustness (theo từng dataset)

| Dataset | Model | IoU ↑ | Dice ↑ | Boundary-F1 |
|---|---|:---:|:---:|:---:|
| SmartDoc (giấy phẳng) | U²-Netp | **0.9639** | **0.9813** | 0.1107 |
| SmartDoc | YOLOv11n-seg | 0.9398 | 0.9690 | 0.1096 |
| Doc3D (giấy cong/OOD) | U²-Netp | 0.5825 | 0.6765 | 0.1475 |
| Doc3D | **YOLOv11n-seg** | **0.7250** | **0.7621** | **0.3437** |

→ Giấy phẳng: hai mô hình tương đương (U²-Net nhỉnh). Dữ liệu khó/biến dạng (Doc3D): **YOLO bền hơn rõ rệt** (+0.14 IoU). Đây là điểm quyết định cho ảnh chụp thực tế.

## Benchmark 4 — End-to-End (pipeline đầy đủ, N=620)

| Pipeline | Median (ms) ↓ | PSNR | SSIM | CER |
|---|:---:|:---:|:---:|:---:|
| U²-Netp | **87.9** | n/a | n/a | n/a |
| YOLOv11n-seg | 144.7 | n/a | n/a | n/a |

→ Pipeline E2E (gồm hậu xử lý): U²-Net nhanh hơn do mask pixel sẵn sàng; YOLO tốn thêm bước decode mask + smoothing.
⚠️ PSNR/SSIM/CER = 0 trong CSV → **chưa được tính** (cần chạy lại với ground-truth dewarp/OCR). Số liệu chất lượng E2E hiện không hợp lệ.

---

## Tổng kết điểm (4/4)

| Benchmark | Thắng | Khoảng cách |
|---|---|---|
| 1. Accuracy | YOLO | +0.083 IoU |
| 2. Speed | YOLO | ~24× (CPU) |
| 3. Robustness (Doc3D/OOD) | YOLO | +0.143 IoU |
| 3. Robustness (SmartDoc) | U²-Net | +0.024 IoU |
| 4. E2E latency | U²-Net | -57 ms |

**Kết luận:** YOLOv11n-seg vượt trội ở 3/4 benchmark (accuracy, speed, robustness OOD) — chính là lý do chọn cho app mobile on-device. U²-Net chỉ thắng ở giấy phẳng lý tưởng (SmartDoc) và độ trễ pipeline đơn lẻ.

> Lưu ý mâu thuẫn số liệu: `docs_ml2/06_Evaluation_Results.md` ghi U²-Net mIoU 0.99 / YOLO 0.94, nhưng CSV thực đo cho IoU 0.72 / 0.80. Bảng docs có thể lấy từ subset SmartDoc hoặc cấu hình khác — nên thống nhất theo CSV gốc (`kpi_*.csv`).
