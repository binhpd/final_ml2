"""Gộp tất cả KPI CSV → 1 báo cáo + figures."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_accuracy_compare(rows: list[dict], out: Path):
    if not rows:
        return
    models = [r["model"] for r in rows]
    metrics = ["iou", "dice", "bf"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, r in enumerate(rows):
        vals = [float(r.get(m, 0)) for m in metrics]
        ax.bar(x + i * width, vals, width, label=r["model"])
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Accuracy comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_speed_compare(rows: list[dict], out: Path):
    if not rows:
        return
    labels = [f"{r['model']}-{r['device']}" for r in rows]
    fps = [float(r["fps"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, fps, color=["steelblue", "darkorange", "seagreen", "crimson"][:len(labels)])
    ax.set_ylabel("FPS")
    ax.set_title("Inference Speed")
    for b, v in zip(bars, fps):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, f"{v:.1f}", ha="center")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="ml2/results")
    ap.add_argument("--out", default="ml2/results/aggregate.md")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    rd.mkdir(parents=True, exist_ok=True)

    speed = read_csv(rd / "kpi_speed.csv")
    accuracy = read_csv(rd / "kpi_accuracy.csv")
    robust = read_csv(rd / "kpi_robustness.csv")
    e2e = read_csv(rd / "kpi_e2e.csv")

    plot_accuracy_compare(accuracy, rd / "fig_accuracy.png")
    plot_speed_compare(speed, rd / "fig_speed.png")

    lines = ["# ML2 Plan B - Aggregate Results\n"]

    if accuracy:
        lines.append("## Accuracy (test set)\n")
        lines.append("| Model | IoU | Dice | MAE | BF | N |")
        lines.append("|---|---|---|---|---|---|")
        for r in accuracy:
            lines.append(f"| {r['model']} | {float(r.get('iou', 0)):.4f} | {float(r.get('dice', 0)):.4f} | {float(r.get('mae', 1)):.4f} | {float(r.get('bf', 0)):.4f} | {r.get('n', '?')} |")
        lines.append("")

    if speed:
        lines.append("## Speed\n")
        lines.append("| Model | Device | Median (ms) | p95 (ms) | FPS |")
        lines.append("|---|---|---|---|---|")
        for r in speed:
            lines.append(f"| {r['model']} | {r['device']} | {float(r['median_ms']):.1f} | {float(r['p95_ms']):.1f} | {float(r['fps']):.1f} |")
        lines.append("")

    if robust:
        lines.append("## Per-dataset (Robustness)\n")
        lines.append("| Dataset | Model | IoU | Dice | BF |")
        lines.append("|---|---|---|---|---|")
        for r in robust:
            lines.append(f"| {r['dataset']} | {r['model']} | {float(r['iou']):.4f} | {float(r['dice']):.4f} | {float(r['bf']):.4f} |")
        lines.append("")

    if e2e:
        lines.append("## End-to-End\n")
        lines.append("| Pipeline | N | Median (ms) | PSNR | SSIM | CER |")
        lines.append("|---|---|---|---|---|---|")
        for r in e2e:
            lines.append(f"| {r['pipeline']} | {r['n']} | {float(r['median_ms']):.1f} | {float(r['psnr']):.2f} | {float(r['ssim']):.4f} | {float(r['cer']):.4f} |")
        lines.append("")

    lines.append("## Figures\n")
    lines.append("- ![Accuracy](fig_accuracy.png)")
    lines.append("- ![Speed](fig_speed.png)")

    out = Path(args.out)
    out.write_text("\n".join(lines))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
