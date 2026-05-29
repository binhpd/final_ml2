"""U²-Net + U²-Netp (lite) — Document Salient Object Detection.

Tham khảo: Qin et al. 2020 "U²-Net: Going Deeper with Nested U-Structure
for Salient Object Detection" - arXiv 2005.09007.

Plan B dùng U2NETp (1.1M params, 4.7MB) — mid_ch=16, out_ch=64 cho tất cả stages.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class REBNCONV(nn.Module):
    """Conv 3x3 + BatchNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


def _upsample_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# RSU — Residual U-blocks
# ---------------------------------------------------------------------------
class RSU(nn.Module):
    """RSU-N block with pooling. N = depth of inner U-Net."""

    def __init__(self, depth: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        assert depth >= 2
        self.depth = depth

        self.in_conv = REBNCONV(in_ch, out_ch, dilation=1)
        self.enc_convs = nn.ModuleList()
        self.enc_convs.append(REBNCONV(out_ch, mid_ch, dilation=1))
        for _ in range(depth - 1):
            self.enc_convs.append(REBNCONV(mid_ch, mid_ch, dilation=1))

        self.bottleneck = REBNCONV(mid_ch, mid_ch, dilation=2)

        self.dec_convs = nn.ModuleList()
        for _ in range(depth - 1):
            self.dec_convs.append(REBNCONV(mid_ch * 2, mid_ch, dilation=1))
        self.dec_out = REBNCONV(mid_ch * 2, out_ch, dilation=1)

        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.in_conv(x)

        feats = []
        h = self.enc_convs[0](x_in)
        feats.append(h)
        for i in range(1, self.depth):
            h = self.pool(h)
            h = self.enc_convs[i](h)
            feats.append(h)

        h = self.bottleneck(h)

        for i in range(self.depth - 1):
            h = torch.cat([h, feats[self.depth - 1 - i]], dim=1)
            h = self.dec_convs[i](h)
            h = _upsample_like(h, feats[self.depth - 2 - i])

        h = torch.cat([h, feats[0]], dim=1)
        h = self.dec_out(h)
        return h + x_in


class RSU_F(nn.Module):
    """RSU-NF block — không có pooling, dùng dilation (cho En_5, En_6)."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.in_conv = REBNCONV(in_ch, out_ch, dilation=1)
        self.enc1 = REBNCONV(out_ch, mid_ch, dilation=1)
        self.enc2 = REBNCONV(mid_ch, mid_ch, dilation=2)
        self.enc3 = REBNCONV(mid_ch, mid_ch, dilation=4)
        self.bottleneck = REBNCONV(mid_ch, mid_ch, dilation=8)
        self.dec3 = REBNCONV(mid_ch * 2, mid_ch, dilation=4)
        self.dec2 = REBNCONV(mid_ch * 2, mid_ch, dilation=2)
        self.dec1 = REBNCONV(mid_ch * 2, out_ch, dilation=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.in_conv(x)
        h1 = self.enc1(x_in)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        h = self.bottleneck(h3)
        h = self.dec3(torch.cat([h, h3], dim=1))
        h = self.dec2(torch.cat([h, h2], dim=1))
        h = self.dec1(torch.cat([h, h1], dim=1))
        return h + x_in


# ---------------------------------------------------------------------------
# U²-Net (full + lite)
# ---------------------------------------------------------------------------
class _U2NetBase(nn.Module):
    """Khung chung U²-Net với 6 encoder + 5 decoder + 6 side outputs."""

    def __init__(self, in_ch: int, out_ch: int, cfg: list[tuple]):
        super().__init__()
        # cfg: list of 11 entries (En1-En6, De5-De1) - each (block_type, in, mid, out, depth_or_None)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        def build(entry):
            kind, ic, mc, oc, depth = entry
            if kind == "RSU":
                return RSU(depth, ic, mc, oc)
            return RSU_F(ic, mc, oc)

        # Encoder
        self.stage1 = build(cfg[0])
        self.stage2 = build(cfg[1])
        self.stage3 = build(cfg[2])
        self.stage4 = build(cfg[3])
        self.stage5 = build(cfg[4])
        self.stage6 = build(cfg[5])

        # Decoder
        self.stage5d = build(cfg[6])
        self.stage4d = build(cfg[7])
        self.stage3d = build(cfg[8])
        self.stage2d = build(cfg[9])
        self.stage1d = build(cfg[10])

        # Side outputs
        side_chs = [cfg[10][3], cfg[9][3], cfg[8][3], cfg[7][3], cfg[6][3], cfg[5][3]]
        self.side_convs = nn.ModuleList([nn.Conv2d(c, out_ch, 3, padding=1) for c in side_chs])
        self.fuse_conv = nn.Conv2d(out_ch * 6, out_ch, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Encoder
        h1 = self.stage1(x)
        h = self.pool(h1)
        h2 = self.stage2(h)
        h = self.pool(h2)
        h3 = self.stage3(h)
        h = self.pool(h3)
        h4 = self.stage4(h)
        h = self.pool(h4)
        h5 = self.stage5(h)
        h = self.pool(h5)
        h6 = self.stage6(h)
        h6_up = _upsample_like(h6, h5)

        # Decoder
        d5 = self.stage5d(torch.cat([h6_up, h5], dim=1))
        d5_up = _upsample_like(d5, h4)
        d4 = self.stage4d(torch.cat([d5_up, h4], dim=1))
        d4_up = _upsample_like(d4, h3)
        d3 = self.stage3d(torch.cat([d4_up, h3], dim=1))
        d3_up = _upsample_like(d3, h2)
        d2 = self.stage2d(torch.cat([d3_up, h2], dim=1))
        d2_up = _upsample_like(d2, h1)
        d1 = self.stage1d(torch.cat([d2_up, h1], dim=1))

        # Side outputs
        s1 = self.side_convs[0](d1)
        s2 = _upsample_like(self.side_convs[1](d2), x)
        s3 = _upsample_like(self.side_convs[2](d3), x)
        s4 = _upsample_like(self.side_convs[3](d4), x)
        s5 = _upsample_like(self.side_convs[4](d5), x)
        s6 = _upsample_like(self.side_convs[5](h6), x)
        s1 = _upsample_like(s1, x)

        s0 = self.fuse_conv(torch.cat([s1, s2, s3, s4, s5, s6], dim=1))
        # Trả về (fused, s1..s6) — train dùng raw logits, infer dùng sigmoid bên ngoài
        return s0, s1, s2, s3, s4, s5, s6


def U2NET(in_ch: int = 3, out_ch: int = 1) -> _U2NetBase:
    """Full U²-Net (~44M params)."""
    cfg = [
        ("RSU", in_ch, 32, 64, 7),       # En1
        ("RSU", 64, 32, 128, 6),         # En2
        ("RSU", 128, 64, 256, 5),        # En3
        ("RSU", 256, 128, 512, 4),       # En4
        ("RSUF", 512, 256, 512, None),   # En5
        ("RSUF", 512, 256, 512, None),   # En6
        ("RSUF", 1024, 256, 512, None),  # De5
        ("RSU", 1024, 128, 256, 4),      # De4
        ("RSU", 512, 64, 128, 5),        # De3
        ("RSU", 256, 32, 64, 6),         # De2
        ("RSU", 128, 16, 64, 7),         # De1
    ]
    return _U2NetBase(in_ch, out_ch, cfg)


def U2NETp(in_ch: int = 3, out_ch: int = 1) -> _U2NetBase:
    """U²-Netp lite (~1.1M params, 4.7MB) — mid_ch=16, out_ch=64 đồng nhất."""
    cfg = [
        ("RSU", in_ch, 16, 64, 7),       # En1
        ("RSU", 64, 16, 64, 6),          # En2
        ("RSU", 64, 16, 64, 5),          # En3
        ("RSU", 64, 16, 64, 4),          # En4
        ("RSUF", 64, 16, 64, None),      # En5
        ("RSUF", 64, 16, 64, None),      # En6
        ("RSUF", 128, 16, 64, None),     # De5
        ("RSU", 128, 16, 64, 4),         # De4
        ("RSU", 128, 16, 64, 5),         # De3
        ("RSU", 128, 16, 64, 6),         # De2
        ("RSU", 128, 16, 64, 7),         # De1
    ]
    return _U2NetBase(in_ch, out_ch, cfg)


if __name__ == "__main__":
    import torch
    m = U2NETp()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"U2NETp params: {n_params:,} (~{n_params * 4 / 1e6:.1f}MB fp32)")
    x = torch.randn(1, 3, 320, 320)
    outs = m(x)
    print(f"Outputs: {len(outs)} tensors, fused shape: {outs[0].shape}")
