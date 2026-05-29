"""Export U2-Netp trained weights → ONNX cho production inference."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from ml2.u2net.model import U2NETp


class U2NetpInferWrapper(torch.nn.Module):
    """Wrapper trả về single fused output (sigmoid applied) - ONNX-friendly."""

    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x):
        outs = self.base(x)
        return torch.sigmoid(outs[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ml2/checkpoints/u2netp_doc_final.pth")
    ap.add_argument("--out", default="ml2/checkpoints/u2netp_doc.onnx")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = U2NETp()
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    wrapper = U2NetpInferWrapper(model).eval()
    dummy = torch.randn(1, 3, args.size, args.size)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(out),
        input_names=["input"],
        output_names=["mask"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "mask": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    size_mb = out.stat().st_size / 1e6
    print(f"[ok] ONNX exported -> {out} ({size_mb:.2f}MB)")

    # Verify ONNX
    try:
        import onnx
        m = onnx.load(str(out))
        onnx.checker.check_model(m)
        print(f"[ok] ONNX validation passed | opset={args.opset}")
    except Exception as e:
        print(f"[warn] ONNX check: {e}")


if __name__ == "__main__":
    main()
