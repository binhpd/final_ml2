"""Verify environment cho ML2 Plan B - Mac Studio M4 Max."""
import importlib
import platform
import sys


def check_python():
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    print(f"[{'OK' if ok else 'FAIL'}] Python {v.major}.{v.minor}.{v.micro} (need >= 3.10)")
    return ok


def check_platform():
    sysname = platform.system()
    machine = platform.machine()
    ok = sysname == "Darwin" and machine == "arm64"
    note = " (M-series ARM)" if ok else ""
    print(f"[{'OK' if ok else 'WARN'}] Platform: {sysname} {machine}{note}")
    return True


def check_package(name: str, min_version: str | None = None):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        print(f"[OK] {name} {ver}")
        return True
    except ImportError:
        print(f"[FAIL] {name} chưa cài")
        return False


def check_torch_mps():
    try:
        import torch
        has_mps = torch.backends.mps.is_available()
        built = torch.backends.mps.is_built()
        print(f"[{'OK' if has_mps else 'FAIL'}] MPS available={has_mps} built={built}")
        if has_mps:
            x = torch.randn(4, 4, device="mps")
            y = x @ x.T
            torch.mps.synchronize()
            print(f"[OK] MPS matmul test passed: {y.shape}")
        return has_mps
    except Exception as e:
        print(f"[FAIL] torch MPS error: {e}")
        return False


def main():
    print("=" * 60)
    print("ML2 Plan B - Environment Check")
    print("=" * 60)

    results = []
    results.append(check_python())
    results.append(check_platform())

    print("\n--- Core packages ---")
    for pkg in ["torch", "torchvision", "numpy", "cv2", "PIL"]:
        check_package(pkg)

    print("\n--- DL packages ---")
    for pkg in ["ultralytics", "albumentations", "pytorch_msssim", "einops"]:
        check_package(pkg)

    print("\n--- Utils ---")
    for pkg in ["yaml", "tqdm", "matplotlib", "tensorboard"]:
        check_package(pkg)

    print("\n--- MPS backend ---")
    check_torch_mps()

    print("\n" + "=" * 60)
    if all(results):
        print("Sẵn sàng train. Chạy build_dummy_data.py để test.")
    else:
        print("Thiếu deps. Chạy: pip install -r ml2/requirements.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
