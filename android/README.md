# Doc Scanner ML2 — Android (on-device document segmentation)

Ứng dụng Android test tính năng **cắt nền tài liệu on-device** bằng model YOLOv11n-seg
(TFLite). Mở camera → khung trang giấy được nhận diện realtime → bấm chụp để xuất ảnh
đã tách nền (background trong suốt).

## Tính năng
- **Camera live + overlay**: vẽ mask + khung (bbox kiểu scanner) của trang giấy theo thời gian thực.
- **Chọn model**: nút `1 trang` (model `document`) ↔ `Sách (bỏ gáy)` (model `left_page/right_page`).
- **Chụp & cắt nền**: dùng mask làm kênh alpha → xuất cutout nền trong suốt.
- 100% on-device, không cần mạng. Inference TFLite + XNNPACK (CPU đa luồng).

## Model
Hai file `.tflite` (FP32, imgsz 640) export từ checkpoint gốc bằng Ultralytics, nằm tại
`app/src/main/assets/`:
| Asset | Nguồn | Lớp |
|---|---|---|
| `yolo_doc.tflite` | `exported_models/yolo11n_seg_doc.pt` | `document` |
| `yolo_spine.tflite` | `exported_models/yolo11n_seg_spine_exclusion_best.pt` | `left_page`, `right_page` |

Layout I/O đã xác minh:
- input `images` `[1,640,640,3]` float32, RGB, chuẩn hoá 0..1
- output `[1, 4+nc+32, 8400]` (box xywh chuẩn hoá 0..1 + class sigmoid + 32 mask coeff)
- output `[1,160,160,32]` mask prototypes

Re-export khi cần:
```bash
source venv_ml2/bin/activate
yolo export model=exported_models/yolo11n_seg_doc.pt format=tflite imgsz=640
yolo export model=exported_models/yolo11n_seg_spine_exclusion_best.pt format=tflite imgsz=640
# rồi copy *_float32.tflite vào android/app/src/main/assets/ (yolo_doc.tflite / yolo_spine.tflite)
```

## Build & Run

### Cách 1 — Android Studio (khuyến nghị)
1. `File ▸ Open` → chọn thư mục `android/`.
2. Đợi Gradle sync (tự tải AGP 8.5.2, CameraX, TFLite).
3. Cắm điện thoại (bật USB debugging) → Run ▶.

### Cách 2 — CLI
```bash
cd android
# JDK 17 của Android Studio:
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
./gradlew :app:assembleDebug          # APK: app/build/outputs/apk/debug/app-debug.apk
./gradlew :app:installDebug           # cài thẳng vào máy đang cắm
```
`local.properties` đã trỏ `sdk.dir` tới `~/Library/Android/sdk`.

## Kiến trúc code
| File | Vai trò |
|---|---|
| `YoloSegModel.kt` | Nạp TFLite, tiền xử lý letterbox, decode (NMS + mask), trả `InferenceResult` |
| `OverlayView.kt` | Vẽ mask + khung lên preview, ánh xạ toạ độ center-crop khớp `PreviewView` |
| `CutoutUtils.kt` | Dùng mask làm alpha → cutout nền trong suốt |
| `MainActivity.kt` | CameraX (Preview + ImageAnalysis), toggle model, chụp |
| `Detection.kt` | Data class kết quả |

## Thông số chạy
- minSdk 24, targetSdk 34, Kotlin 1.9.24, AGP 8.5.2.
- Inference CPU/XNNPACK 4 luồng. Frame analysis dùng `STRATEGY_KEEP_ONLY_LATEST` (drop frame, không nghẽn).

## Hạn chế & hướng nâng cấp
- Khung hiển thị là **mask tô màu + bbox**, chưa phải tứ giác 4 góc bám mép giấy.
  Nâng cấp: trích contour từ mask → `minAreaRect`/`approxPolyDP` để vẽ tứ giác + perspective warp.
- Model FP32 ~11.6MB/cái. Có thể dùng bản `float16.tflite` (~5.8MB) hoặc INT8 (cần representative dataset) để giảm size/tăng tốc NPU.
- Tăng tốc thêm: bật GPU delegate (`tensorflow-lite-gpu`) hoặc NNAPI nếu thiết bị hỗ trợ.
- Cutout chạy ở độ phân giải khung analysis; muốn nét hơn dùng `ImageCapture` chụp full-res rồi mới infer.
