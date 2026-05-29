"""Convert Doc3D HuggingFace zips → DocSegDataset format.

Source layout (~/ml2_datasets_full/doc3d_hf/doc3d/):
  img_{1..21}.zip      (images, 448x448 PNG inside img/<chunk>/*.png)
  uv_{1..21}.zip       (UV maps, EXR inside uv/<chunk>/*.exr — magnitude=0 outside doc)

Output:
  ml2/data/doc3d/
    images/<id>.png
    masks/<id>.png    (binary from UV magnitude > epsilon)
    train.txt val.txt test.txt
"""
from __future__ import annotations

import argparse
import os
import random
import zipfile
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
from tqdm import tqdm


def extract_zip_to(zip_path: Path, member_filter: str, out_dir: Path):
    """Extract files from zip matching prefix into out_dir flat."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.startswith(member_filter) and not n.endswith("/")]
        for m in members:
            data = zf.read(m)
            target = out_dir / Path(m).name
            with open(target, "wb") as f:
                f.write(data)
    return len(members)


def uv_to_mask(uv_path: Path, eps: float = 0.001) -> np.ndarray:
    """Read UV EXR, derive binary mask (255 = doc)."""
    uv = cv2.imread(str(uv_path), cv2.IMREAD_UNCHANGED)
    if uv is None:
        return None
    if uv.ndim == 3:
        mag = np.linalg.norm(uv, axis=-1)
    else:
        mag = np.abs(uv)
    mask = (mag > eps).astype(np.uint8) * 255
    return mask


def process_chunk(src_dir: Path, chunk_id: int, img_dst: Path, mask_dst: Path, max_n: int | None) -> list[str]:
    """Process one chunk_id (1..21): extract img zip + uv zip, build masks."""
    img_zip = src_dir / f"img_{chunk_id}.zip"
    uv_zip = src_dir / f"uv_{chunk_id}.zip"
    if not img_zip.exists() or not uv_zip.exists():
        print(f"[skip chunk {chunk_id}] missing zip")
        return []

    tmp_img = Path(f"/tmp/doc3d_chunk{chunk_id}_img")
    tmp_uv = Path(f"/tmp/doc3d_chunk{chunk_id}_uv")
    tmp_img.mkdir(parents=True, exist_ok=True)
    tmp_uv.mkdir(parents=True, exist_ok=True)

    n_img = extract_zip_to(img_zip, f"img/{chunk_id}/", tmp_img)
    n_uv = extract_zip_to(uv_zip, f"uv/{chunk_id}/", tmp_uv)
    print(f"[chunk {chunk_id}] img={n_img} uv={n_uv}")

    names = []
    img_files = sorted(tmp_img.glob("*.png"))
    if max_n:
        img_files = img_files[:max_n]

    for img_p in tqdm(img_files, desc=f"chunk{chunk_id}"):
        stem = img_p.stem
        uv_p = tmp_uv / f"{stem}.exr"
        if not uv_p.exists():
            continue
        mask = uv_to_mask(uv_p)
        if mask is None or mask.sum() < 1000:
            continue
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        out_name = f"d3d_{chunk_id}_{stem}"
        if not cv2.imwrite(str(img_dst / f"{out_name}.png"), img):
            continue
        cv2.imwrite(str(mask_dst / f"{out_name}.png"), mask)
        names.append(out_name)

    # Cleanup temp
    for f in tmp_img.glob("*"):
        f.unlink()
    for f in tmp_uv.glob("*"):
        f.unlink()
    tmp_img.rmdir()
    tmp_uv.rmdir()
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "ml2_datasets_full" / "doc3d_hf" / "doc3d"))
    ap.add_argument("--out", default="ml2/data/doc3d")
    ap.add_argument("--chunks", nargs="+", type=int, default=list(range(1, 22)))
    ap.add_argument("--max_per_chunk", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    img_dst = out / "images"
    mask_dst = out / "masks"
    img_dst.mkdir(parents=True, exist_ok=True)
    mask_dst.mkdir(parents=True, exist_ok=True)

    all_names = []
    for c in args.chunks:
        names = process_chunk(src, c, img_dst, mask_dst, args.max_per_chunk)
        all_names.extend(names)
        print(f"[total so far] {len(all_names)}")

    random.seed(args.seed)
    random.shuffle(all_names)
    n = len(all_names)
    n_train = int(n * 0.90)
    n_val = int(n * 0.05)
    (out / "train.txt").write_text("\n".join(all_names[:n_train]))
    (out / "val.txt").write_text("\n".join(all_names[n_train:n_train + n_val]))
    (out / "test.txt").write_text("\n".join(all_names[n_train + n_val:]))
    print(f"\n[done] total={n} train={n_train} val={n_val} test={n - n_train - n_val}")


if __name__ == "__main__":
    main()
