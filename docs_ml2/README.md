# 📚 docs_ml2 — Tài liệu đồ án U²-Net + YOLO-Seg Document Segmentation

> **Trạng thái:** ✅ Plan B đã chốt | M4 Max 48GB | 3 datasets online | Timeline 7 ngày
> **Đồ án cuối kỳ ML2 — Nhóm 6**

---

## 🗂️ Cấu trúc thư mục (3 tầng)

```
docs_ml2/
│
├── 📍 README.md             ← Bạn đang đọc (master index, 1 trang)
│
├── 📋 01_KeHoach.md         ← Plan B + Checklist (gộp Plan + Decisions + Tick)
├── 🔬 02_Spec_KyThuat.md    ← Kiến trúc + Loss + KPI + 41 file code spec
├── 📖 03_Research_Note.md   ← Tóm tắt 9 file research nền tảng
│
└── _archive/                ← Tài liệu cũ (giữ làm reference)
    ├── older_plans/         ← 5 file plan cũ (đã gộp vào 01_KeHoach.md)
    │   ├── KeHoach_Train_TichHop_U2Net_YOLO.md  (bản gốc của bạn)
    │   ├── KeHoach_TrienKhai_Detail.md          (plan trung gian)
    │   ├── PlanB_Datasets_Final.md              (quyết định datasets)
    │   ├── TomTat_1Trang.md                     (summary cũ)
    │   └── Checklist_Review.md                  (checklist cũ)
    │
    └── research_docs/       ← 9 file research nền (đã tóm trong 03_Research_Note.md)
        ├── 00_TongHop_Claude.md
        ├── 01_U2NET_ChiTiet_Claude.md
        ├── 02_YOLO_ChiTiet_Claude.md
        ├── 03_Integration_KPI_Claude.md
        ├── KeHoach_TongQuan_Claude.md
        ├── DeXuat_ChuDe_DeepLearning_Claude.md
        ├── SoSanh_YOLO_vs_SOTA_DocumentLayout_Claude.md
        ├── XuHuong_Gap_HuongNghienCuu_YOLO_DLA_Claude.md
        └── DeCuong_DoAn_3CapDo_DocLayoutYOLOv2_Claude.md
```

---

## 🎯 Đọc file nào theo mục đích

| Mục đích của bạn | File cần đọc |
|------------------|--------------|
| **Xem nhanh tổng thể (5 phút)** | README.md (file này) |
| **Duyệt plan + tick checklist** | [01_KeHoach.md](01_KeHoach.md) |
| **Build code: tham khảo spec từng file** | [02_Spec_KyThuat.md](02_Spec_KyThuat.md) |
| **Viết báo cáo, tìm references** | [03_Research_Note.md](03_Research_Note.md) |
| **Defense Q&A về dataset choices** | [03_Research_Note.md](03_Research_Note.md) §4 |
| **Defense Q&A về kiến trúc U²-Net** | [02_Spec_KyThuat.md](02_Spec_KyThuat.md) §1 + [_archive/research_docs/01_U2NET_ChiTiet_Claude.md](_archive/research_docs/01_U2NET_ChiTiet_Claude.md) |
| **Defense Q&A về YOLOv11** | [02_Spec_KyThuat.md](02_Spec_KyThuat.md) §2 + [_archive/research_docs/02_YOLO_ChiTiet_Claude.md](_archive/research_docs/02_YOLO_ChiTiet_Claude.md) |
| **Defense Q&A về KPI** | [02_Spec_KyThuat.md](02_Spec_KyThuat.md) §5 |

---

## ⚡ TL;DR — 30 giây tóm tắt

**Bạn đang xây dựng gì:** Train 2 model Deep Learning để thay phần Step 1 (Document Detection) trong pipeline scan tài liệu.

| Model | Train | Vai trò |
|-------|-------|---------|
| **U²-Netp lite** (1.1M, 4.7MB) | Train from scratch 1 stage | Tách nền — thay `rembg` |
| **YOLOv11n-seg** (2.9M, 6MB) | Fine-tune COCO | Phân vùng + bbox + viz |

**Datasets:** SmartDoc ICDAR 2015 (7K) + Doc3D (5K) = **12,000 ảnh online**

**Timeline:** Build code 6h (Claude) → Train 5-7 ngày (M4 Max) → Benchmark 1 ngày

**KPI mục tiêu:** mIoU ≥ 0.83, FPS ≥ 20 (MPS), Model size ≤ 6MB

---

## 🚀 Bước tiếp theo

1. **Đọc** [01_KeHoach.md](01_KeHoach.md) — tick checkbox section 5
2. **Trả lời** 6 câu hỏi mục 6 trong `01_KeHoach.md` (edit trực tiếp file)
3. **Báo Claude** đã duyệt xong → Claude build 40 file còn lại (~6h)
4. **Tự** tải datasets + train trên M4 Max (~7 ngày)
5. **Viết** báo cáo bảo vệ (Claude giải thích code khi cần)

---

## 📊 Trạng thái

⏸️ **Chưa build code** — đang ở giai đoạn duyệt plan.

```
docs_ml2/   ← 4 file plan đã sẵn sàng review
ml2/        ← CHƯA tạo (sẽ tạo khi user nói "OK build")
```

→ Khi bạn tick xong checklist trong [01_KeHoach.md](01_KeHoach.md) và nói "Bắt đầu build", Claude sẽ tạo `ml2/` với 41 file Python + 4 notebook (~4,400 dòng) theo spec trong [02_Spec_KyThuat.md](02_Spec_KyThuat.md).

---

## 📞 Khi bạn cần Claude

- "Build tiếp theo plan" → Claude tiếp tục build 40 file còn lại
- "Giải thích U2NET RSU block" → Claude giải thích kèm code mẫu
- "Thay đổi … trong plan" → Claude update [01_KeHoach.md](01_KeHoach.md)
- "Debug lỗi train" → Claude xem log + fix code

---

*Đã hệ thống lại từ 14 file → 4 file active + archive. Reference đầy đủ vẫn còn trong `_archive/`.*
