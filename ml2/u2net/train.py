"""Training loop U²-Netp lite cho document segmentation.

Hỗ trợ: MPS (M4 Max), CUDA, CPU. AMP tắt mặc định trên MPS (buggy).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# MPS fallback for unsupported ops (Sobel, dilated conv edge cases)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml2.u2net.augmentation import get_transform
from ml2.u2net.dataset import DocSegDataset
from ml2.u2net.loss import ComboLoss
from ml2.u2net.model import U2NET, U2NETp


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device(name: str) -> torch.device:
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(cfg: dict) -> torch.nn.Module:
    arch = cfg["model"]["arch"]
    in_ch = cfg["model"].get("in_channels", 3)
    out_ch = cfg["model"].get("out_channels", 1)
    if arch == "U2NET":
        return U2NET(in_ch, out_ch)
    return U2NETp(in_ch, out_ch)


def build_dataloaders(cfg: dict, dummy: bool = False) -> tuple[DataLoader, DataLoader | None]:
    size = cfg["data"]["image_size"]
    strong = cfg["data"].get("augmentation", "strong") == "strong"
    roots = cfg["data"]["roots"] if not dummy else ["ml2/data/dummy"]

    train_tf = get_transform("train", size=size, strong=strong)
    val_tf = get_transform("val", size=size)

    train_ds = DocSegDataset(roots, split="train", transform=train_tf)
    try:
        val_ds = DocSegDataset(roots, split="val", transform=val_tf)
    except RuntimeError:
        val_ds = None

    bs = cfg["train"]["batch_size"]
    nw = cfg["data"].get("num_workers", 0)
    train_dl = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw,
        pin_memory=False, drop_last=True,
    )
    val_dl = None
    if val_ds is not None:
        val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=False)
    return train_dl, val_dl


def cosine_lr(step: int, total: int, base_lr: float, warmup: int = 0) -> float:
    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    inter = union = pos_p = pos_t = 0.0
    n_pix = 0.0
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        gt = batch["mask"].to(device, non_blocking=True)
        out = model(img)[0]
        pred = (torch.sigmoid(out) > 0.5).float()
        inter += (pred * gt).sum().item()
        union += (pred + gt - pred * gt).sum().item()
        pos_p += pred.sum().item()
        pos_t += gt.sum().item()
        n_pix += gt.numel()
    iou = inter / max(1.0, union)
    dice = 2 * inter / max(1.0, pos_p + pos_t)
    return {"iou": iou, "dice": dice}


def train_one_stage(cfg: dict, args, model: torch.nn.Module, device: torch.device, stage: str = "main"):
    train_dl, val_dl = build_dataloaders(cfg, dummy=args.dummy)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
        betas=tuple(cfg["train"].get("betas", [0.9, 0.999])),
    )
    criterion = ComboLoss(
        w_bce=cfg["loss"]["w_bce"],
        w_iou=cfg["loss"]["w_iou"],
        w_ssim=cfg["loss"]["w_ssim"],
        w_edge=cfg["loss"].get("w_edge", 0.0),
    )
    epochs = args.epochs if args.epochs else cfg["train"]["epochs"]
    grad_clip = cfg["train"].get("grad_clip", 0.0)
    base_lr = cfg["train"]["lr"]
    use_cosine = cfg["train"].get("scheduler", "none") == "cosine"
    warmup = cfg["train"].get("warmup_epochs", 0)

    out_dir = Path(cfg["checkpoint"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg["log"]["tb_dir"] + f"_{stage}")

    best_iou = -1.0
    global_step = 0
    for epoch in range(epochs):
        model.train()
        bar = tqdm(train_dl, desc=f"[{stage}] epoch {epoch + 1}/{epochs}")
        loss_acc = 0.0
        for batch in bar:
            if use_cosine:
                lr = cosine_lr(epoch, epochs, base_lr, warmup)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            img = batch["image"].to(device, non_blocking=True)
            gt = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad()
            outs = model(img)
            loss, per_side = criterion(outs, gt)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            loss_acc += loss.item()
            global_step += 1
            if global_step % cfg["log"].get("print_every", 50) == 0:
                writer.add_scalar(f"{stage}/loss", loss.item(), global_step)
                writer.add_scalar(f"{stage}/lr", optimizer.param_groups[0]["lr"], global_step)
            bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = loss_acc / max(1, len(train_dl))
        writer.add_scalar(f"{stage}/epoch_loss", avg_loss, epoch)

        if val_dl is not None and (epoch + 1) % max(1, epochs // 20) == 0:
            metrics = evaluate(model, val_dl, device)
            writer.add_scalar(f"{stage}/val_iou", metrics["iou"], epoch)
            writer.add_scalar(f"{stage}/val_dice", metrics["dice"], epoch)
            print(f"[val] epoch {epoch + 1}: IoU={metrics['iou']:.4f} Dice={metrics['dice']:.4f}")
            if metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                torch.save(model.state_dict(), out_dir / f"u2netp_{stage}_best.pth")

        if (epoch + 1) % cfg["checkpoint"].get("save_every", 10) == 0:
            torch.save(model.state_dict(), out_dir / f"u2netp_{stage}_epoch{epoch + 1}.pth")

    torch.save(model.state_dict(), out_dir / f"u2netp_{stage}_final.pth")
    writer.close()
    return best_iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--dummy", action="store_true", help="Train trên dummy data")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = get_device(cfg["train"]["device"])
    print(f"[device] {device}")

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {cfg['model']['arch']} params={n_params:,}")

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"[resume] loaded {args.resume}")

    # Optional pretrain stage trên DocAligner
    if cfg.get("pretrain", {}).get("enabled", False) and not args.dummy:
        pre_cfg = dict(cfg)
        pre_cfg = {**cfg, "data": {**cfg["data"], "roots": cfg["pretrain"]["roots"]}}
        pre_cfg["train"] = {**cfg["train"], "epochs": cfg["pretrain"]["epochs"], "lr": cfg["pretrain"]["lr"]}
        print(f"[pretrain] DocAligner stage - {cfg['pretrain']['epochs']} epochs")
        train_one_stage(pre_cfg, args, model, device, stage="pretrain")

    t0 = time.time()
    best = train_one_stage(cfg, args, model, device, stage="main")
    print(f"[done] best val IoU: {best:.4f} | time {(time.time() - t0) / 60:.1f}m")


if __name__ == "__main__":
    main()
