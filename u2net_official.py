"""Kiến trúc U2NET GỐC (xuebinqin/U-2-Net) để load weights pretrained salient
(u2net_full.pth, naming rebnconv...). Kèm tiền xử lý gốc (chia max + ImageNet norm)."""
import cv2, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super().__init__()
        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


def _up(src, tar):
    return F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=False)


class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, 1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, 1); self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, 1); self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, 1); self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, 1); self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, 1); self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, 1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, 2)
        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, 1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, 1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, 1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, 1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, 1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, 1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        h1 = self.rebnconv1(hxin); h = self.pool1(h1)
        h2 = self.rebnconv2(h); h = self.pool2(h2)
        h3 = self.rebnconv3(h); h = self.pool3(h3)
        h4 = self.rebnconv4(h); h = self.pool4(h4)
        h5 = self.rebnconv5(h); h = self.pool5(h5)
        h6 = self.rebnconv6(h)
        h7 = self.rebnconv7(h6)
        h6d = self.rebnconv6d(torch.cat((h7, h6), 1))
        h5d = self.rebnconv5d(torch.cat((_up(h6d, h5), h5), 1))
        h4d = self.rebnconv4d(torch.cat((_up(h5d, h4), h4), 1))
        h3d = self.rebnconv3d(torch.cat((_up(h4d, h3), h3), 1))
        h2d = self.rebnconv2d(torch.cat((_up(h3d, h2), h2), 1))
        h1d = self.rebnconv1d(torch.cat((_up(h2d, h1), h1), 1))
        return h1d + hxin


class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, 1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, 1); self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, 1); self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, 1); self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, 1); self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, 1)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, 2)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, 1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, 1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, 1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, 1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, 1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        h1 = self.rebnconv1(hxin); h = self.pool1(h1)
        h2 = self.rebnconv2(h); h = self.pool2(h2)
        h3 = self.rebnconv3(h); h = self.pool3(h3)
        h4 = self.rebnconv4(h); h = self.pool4(h4)
        h5 = self.rebnconv5(h)
        h6 = self.rebnconv6(h5)
        h5d = self.rebnconv5d(torch.cat((h6, h5), 1))
        h4d = self.rebnconv4d(torch.cat((_up(h5d, h4), h4), 1))
        h3d = self.rebnconv3d(torch.cat((_up(h4d, h3), h3), 1))
        h2d = self.rebnconv2d(torch.cat((_up(h3d, h2), h2), 1))
        h1d = self.rebnconv1d(torch.cat((_up(h2d, h1), h1), 1))
        return h1d + hxin


class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, 1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, 1); self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, 1); self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, 1); self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, 1)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, 2)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, 1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, 1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, 1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, 1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        h1 = self.rebnconv1(hxin); h = self.pool1(h1)
        h2 = self.rebnconv2(h); h = self.pool2(h2)
        h3 = self.rebnconv3(h); h = self.pool3(h3)
        h4 = self.rebnconv4(h)
        h5 = self.rebnconv5(h4)
        h4d = self.rebnconv4d(torch.cat((h5, h4), 1))
        h3d = self.rebnconv3d(torch.cat((_up(h4d, h3), h3), 1))
        h2d = self.rebnconv2d(torch.cat((_up(h3d, h2), h2), 1))
        h1d = self.rebnconv1d(torch.cat((_up(h2d, h1), h1), 1))
        return h1d + hxin


class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, 1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, 1); self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, 1); self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, 1)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, 2)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, 1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, 1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, 1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        h1 = self.rebnconv1(hxin); h = self.pool1(h1)
        h2 = self.rebnconv2(h); h = self.pool2(h2)
        h3 = self.rebnconv3(h)
        h4 = self.rebnconv4(h3)
        h3d = self.rebnconv3d(torch.cat((h4, h3), 1))
        h2d = self.rebnconv2d(torch.cat((_up(h3d, h2), h2), 1))
        h1d = self.rebnconv1d(torch.cat((_up(h2d, h1), h1), 1))
        return h1d + hxin


class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super().__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, 1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, 1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, 2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, 4)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, 8)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, 4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, 2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, 1)

    def forward(self, x):
        hxin = self.rebnconvin(x)
        h1 = self.rebnconv1(hxin)
        h2 = self.rebnconv2(h1)
        h3 = self.rebnconv3(h2)
        h4 = self.rebnconv4(h3)
        h3d = self.rebnconv3d(torch.cat((h4, h3), 1))
        h2d = self.rebnconv2d(torch.cat((h3d, h2), 1))
        h1d = self.rebnconv1d(torch.cat((h2d, h1), 1))
        return h1d + hxin


class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.stage1 = RSU7(in_ch, 32, 64); self.pool12 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage2 = RSU6(64, 32, 128); self.pool23 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage3 = RSU5(128, 64, 256); self.pool34 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage4 = RSU4(256, 128, 512); self.pool45 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage5 = RSU4F(512, 256, 512); self.pool56 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage6 = RSU4F(512, 256, 512)
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        h1 = self.stage1(x); h = self.pool12(h1)
        h2 = self.stage2(h); h = self.pool23(h2)
        h3 = self.stage3(h); h = self.pool34(h3)
        h4 = self.stage4(h); h = self.pool45(h4)
        h5 = self.stage5(h); h = self.pool56(h5)
        h6 = self.stage6(h)
        h5d = self.stage5d(torch.cat((_up(h6, h5), h5), 1))
        h4d = self.stage4d(torch.cat((_up(h5d, h4), h4), 1))
        h3d = self.stage3d(torch.cat((_up(h4d, h3), h3), 1))
        h2d = self.stage2d(torch.cat((_up(h3d, h2), h2), 1))
        h1d = self.stage1d(torch.cat((_up(h2d, h1), h1), 1))
        d1 = self.side1(h1d)
        d2 = _up(self.side2(h2d), d1)
        d3 = _up(self.side3(h3d), d1)
        d4 = _up(self.side4(h4d), d1)
        d5 = _up(self.side5(h5d), d1)
        d6 = _up(self.side6(h6), d1)
        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))
        return torch.sigmoid(d0)


def load_official(ckpt, device):
    m = U2NET(3, 1).to(device)
    sd = torch.load(ckpt, map_location=device)
    sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
    m.load_state_dict(sd)
    m.eval()
    return m


@torch.no_grad()
def infer_official(model, img_bgr, device, size=320):
    """Tiền xử lý GỐC: resize -> /max -> ImageNet norm. Trả mask prob (H,W) 0..1."""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    mx = r.max()
    r = r / (mx if mx > 0 else 1.0)
    r[:, :, 0] = (r[:, :, 0] - 0.485) / 0.229
    r[:, :, 1] = (r[:, :, 1] - 0.456) / 0.224
    r[:, :, 2] = (r[:, :, 2] - 0.406) / 0.225
    t = torch.from_numpy(r.transpose(2, 0, 1)).unsqueeze(0).to(device)
    d0 = model(t)[0, 0].cpu().numpy()
    mn, mx = d0.min(), d0.max()
    d0 = (d0 - mn) / (mx - mn + 1e-8)              # normalize prediction (như repo gốc)
    return cv2.resize(d0, (w, h), interpolation=cv2.INTER_LINEAR)
