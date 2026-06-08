"""So sánh 5 model trên cùng tập test (single-document GT masks).

Models:
  1. yolo_old      : yolo11n-seg.pt                              (COCO pretrained - chưa train doc)
  2. u2net_old     : ml2/checkpoints/u2net_full.pth             (U2NET full pretrained salient - chưa train doc)
  3. u2net_trained : exported_models/u2netp_doc_final.pth       (U2NETp lite - tự train doc)
  4. yolo_trained  : exported_models/yolo11n_seg_doc.pt         (YOLOv11n-seg - tự train doc)
  5. yolo_tuned    : exported_models/yolo11n_seg_spine_exclusion_best.pt (YOLO tuning 2-class)

Chỉ số: IoU, Dice, MAE, Boundary-F1 (accuracy) + latency median (speed).
YOLO: union TẤT CẢ instance mask -> vùng tài liệu, để công bằng giữa model 1-class & 2-class.
"""
from __future__ import annotations
import argparse, csv, os, sys, time
from pathlib import Path
import cv2, numpy as np, torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ml2.u2net.eval import metrics_pair
from ml2.u2net.infer import load_model, infer_single

MODELS = [
    ("yolo_old",      "yolo",     "yolo11n-seg.pt"),
    ("u2net_old",     "u2net_off", "ml2/checkpoints/u2net_full.pth"),
    ("u2net_trained", "u2net",     "exported_models/u2netp_doc_final.pth"),
    ("yolo_trained",  "yolo",     "exported_models/yolo11n_seg_doc.pt"),
    ("yolo_tuned",    "yolo",     "exported_models/yolo11n_seg_spine_exclusion_best.pt"),
]


def collect(roots, split, limit):
    samples = []
    for root in roots:
        root = Path(root)
        f = root / f"{split}.txt"
        if not f.exists():
            continue
        names = [n.strip() for n in f.read_text().splitlines() if n.strip()]
        if limit and len(names) > limit:                      # lấy đều
            idx = np.linspace(0, len(names) - 1, limit).astype(int)
            names = [names[i] for i in idx]
        for name in names:
            mask = root / "masks" / f"{name}.png"
            img = None
            for ext in (".jpg", ".png", ".jpeg"):
                p = root / "images" / f"{name}{ext}"
                if p.exists():
                    img = p; break
            if img and mask.exists():
                samples.append((img, mask))
    return samples


def get_params_count(model, is_yolo=False):
    if is_yolo:
        return sum(p.numel() for p in model.model.parameters())
    return sum(p.numel() for p in model.parameters())


def eval_u2net(ckpt, samples, device):
    """U2NETp lite (tự train doc)."""
    dev = torch.device(device if (device != "mps" or torch.backends.mps.is_available()) else "cpu")
    model = load_model(ckpt, dev, is_lite=True)
    params = get_params_count(model, is_yolo=False)
    mets, times = [], []
    for img_p, mask_p in samples:
        img = cv2.imread(str(img_p))
        gt = (cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        t = time.perf_counter()
        pred = infer_single(model, img, dev, size=320)
        times.append((time.perf_counter() - t) * 1000)
        mets.append(metrics_pair(pred, gt))
    res = agg(mets, times)
    res["params"] = params
    return res


def eval_u2net_official(ckpt, samples, device):
    """U2NET full GỐC (pretrained salient, chưa train doc)."""
    from u2net_official import load_official, infer_official
    dev = torch.device(device if (device != "mps" or torch.backends.mps.is_available()) else "cpu")
    model = load_official(ckpt, dev)
    params = get_params_count(model, is_yolo=False)
    mets, times = [], []
    for img_p, mask_p in samples:
        img = cv2.imread(str(img_p))
        gt = (cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        t = time.perf_counter()
        pred = infer_official(model, img, dev, size=320)
        times.append((time.perf_counter() - t) * 1000)
        mets.append(metrics_pair(pred, gt))
    res = agg(mets, times)
    res["params"] = params
    return res


def eval_yolo(weights, samples, device):
    from ultralytics import YOLO
    model = YOLO(weights)
    model(np.zeros((640, 640, 3), np.uint8), device=device, verbose=False)   # warmup
    params = get_params_count(model, is_yolo=True)
    mets, times = [], []
    for img_p, mask_p in samples:
        img = cv2.imread(str(img_p)); h, w = img.shape[:2]
        gt = (cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
        t = time.perf_counter()
        r = model.predict(img, device=device, verbose=False)
        times.append((time.perf_counter() - t) * 1000)
        if not r or r[0].masks is None or len(r[0].masks.data) == 0:
            mets.append({"iou": 0, "dice": 0, "mae": 1, "bf": 0}); continue
        masks = r[0].masks.data.cpu().numpy()
        union = (masks.sum(axis=0) > 0.5).astype(np.float32)          # union mọi instance
        pred = cv2.resize(union, (w, h), interpolation=cv2.INTER_LINEAR)
        mets.append(metrics_pair(pred, gt))
    res = agg(mets, times)
    res["params"] = params
    return res


def agg(mets, times):
    return {
        "iou":  float(np.mean([m["iou"] for m in mets])),
        "dice": float(np.mean([m["dice"] for m in mets])),
        "mae":  float(np.mean([m["mae"] for m in mets])),
        "bf":   float(np.mean([m["bf"] for m in mets])),
        "lat_ms": float(np.median(times)),
        "n": len(mets),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+",
                    default=["ml2/data/smartdoc", "ml2/data/doc3d", "ml2/data/kaggle_real"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=200, help="ảnh tối đa mỗi dataset")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="ml2/results/compare_5models.csv")
    args = ap.parse_args()

    samples = collect(args.roots, args.split, args.limit)
    print(f"[samples] {len(samples)} ảnh từ {len(args.roots)} dataset")
    rows = []
    for name, kind, path in MODELS:
        if not Path(path).exists():
            print(f"[skip] {name}: không thấy {path}"); continue
        print(f"\n>>> {name} ({path})")
        try:
            if kind == "yolo":
                m = eval_yolo(path, samples, args.device)
            elif kind == "u2net_off":
                m = eval_u2net_official(path, samples, args.device)
            else:
                m = eval_u2net(path, samples, args.device)
        except Exception as e:
            print(f"[ERR] {name}: {e}"); continue
        m = {"model": name, "weights": Path(path).name, **m}
        rows.append(m)
        print(f"    IoU={m['iou']:.4f} Dice={m['dice']:.4f} MAE={m['mae']:.4f} "
              f"BF={m['bf']:.4f} lat={m['lat_ms']:.1f}ms params={m['params']/1e6:.2f}M n={m['n']}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[saved] {out}")
    print("\n=== BẢNG SO SÁNH ===")
    print(f"{'model':<15}{'IoU':>8}{'Dice':>8}{'MAE':>8}{'BF':>8}{'lat(ms)':>9}{'params(M)':>11}")
    for r in rows:
        print(f"{r['model']:<15}{r['iou']:>8.4f}{r['dice']:>8.4f}{r['mae']:>8.4f}{r['bf']:>8.4f}{r['lat_ms']:>9.1f}{r['params']/1e6:>11.2f}")


if __name__ == "__main__":
    main()
