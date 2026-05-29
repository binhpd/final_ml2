"""Download datasets cho Plan B: SmartDoc + Doc3D + DocAligner optional.

Lưu ý:
- SmartDoc tải qua Kaggle CLI (cần `kaggle.json` ở ~/.kaggle/)
- Doc3D: clone repo + tải subset qua script gốc (cần Git LFS hoặc gdown)
- DocAligner: tải qua repo GitHub
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve


DATA_DIR = Path("ml2/data")


def run(cmd: list[str], cwd: str | Path | None = None):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def download_smartdoc(out: Path):
    """Download SmartDoc ICDAR 2015 dataset (~2GB)."""
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    print("\n=== SmartDoc ICDAR 2015 ===")

    if have("kaggle"):
        try:
            run([
                "kaggle", "datasets", "download",
                "-d", "jmourad/smartdoc15-dataset",
                "-p", str(raw), "--unzip",
            ])
            return
        except subprocess.CalledProcessError:
            print("[warn] Kaggle download failed - thử method khác")

    print("=" * 60)
    print("Tải thủ công SmartDoc:")
    print("  1. https://www.kaggle.com/datasets/jmourad/smartdoc15-dataset")
    print(f"  2. Extract vào: {raw.resolve()}")
    print("  Hoặc:")
    print("  - http://smartdoc.univ-lr.fr/")
    print("=" * 60)


def download_doc3d(out: Path, subset: bool = True):
    """Clone repo doc3D-dataset + tải subset."""
    out.mkdir(parents=True, exist_ok=True)
    repo = out / "repo"
    print("\n=== Doc3D ===")

    if not repo.exists():
        run(["git", "clone", "https://github.com/cvlab-stonybrook/doc3D-dataset.git", str(repo)])

    print("Doc3D download links (full ~100GB - dùng subset):")
    print("  1. Foreground masks: https://drive.google.com/file/d/1iU-h5lZ5g6gPVgUTQOFqOMsThSGuvHF8/view")
    print("  2. Images: chia thành 7 chunks, tải từng cái")
    print(f"  -> Extract foreground masks + select 5000 ảnh tương ứng vào {out / 'raw'}")
    if subset:
        print("  Dùng prepare_doc3d.py để extract subset 5K sau khi tải xong.")


def download_docaligner(out: Path):
    """Clone DocAligner repo + DocAlign12K synthetic data."""
    out.mkdir(parents=True, exist_ok=True)
    repo = out / "repo"
    print("\n=== DocAligner (optional pretrain) ===")
    if not repo.exists():
        run(["git", "clone", "https://github.com/ZZZHANG-jx/DocAligner.git", str(repo)])
    print("Sau khi clone, theo README repo để tải DocAlign12K:")
    print(f"  cd {repo} && python synthv2.py  # generate synthetic samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smartdoc", action="store_true")
    ap.add_argument("--doc3d", action="store_true")
    ap.add_argument("--docaligner", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=str(DATA_DIR))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    do_sd = args.smartdoc or args.all
    do_d3d = args.doc3d or args.all
    do_da = args.docaligner or args.all

    if not (do_sd or do_d3d or do_da):
        ap.print_help()
        print("\n[!] Phải chọn ít nhất 1 cờ: --smartdoc / --doc3d / --docaligner / --all")
        sys.exit(1)

    if do_sd:
        download_smartdoc(out / "smartdoc")
    if do_d3d:
        download_doc3d(out / "doc3d")
    if do_da:
        download_docaligner(out / "docaligner")

    print("\n[done] Sau khi tải xong:")
    print("  python ml2/scripts/prepare_smartdoc.py")
    print("  python ml2/scripts/prepare_doc3d.py")
    print("  python ml2/scripts/prepare_docaligner.py  # nếu cần pretrain")


if __name__ == "__main__":
    main()
