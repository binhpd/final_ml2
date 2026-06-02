# ML2 Plan B - Aggregate Results

## Accuracy (test set)

| Model | IoU | Dice | MAE | BF | N |
|---|---|---|---|---|---|
| u2net | 0.7179 | 0.7847 | 0.1295 | 0.1344 | 7008 |
| yolo | 0.8013 | 0.8355 | 0.1401 | 0.2606 | 7008 |
| rembg | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0 |

## Speed

| Model | Device | Median (ms) | p95 (ms) | FPS |
|---|---|---|---|---|
| u2net | cpu | 465.5 | 493.1 | 2.1 |
| u2net | mps | 62.5 | 68.6 | 16.0 |
| yolo | cpu | 19.0 | 20.1 | 52.5 |
| yolo | mps | 8.2 | 8.7 | 121.6 |

## Per-dataset (Robustness)

| Dataset | Model | IoU | Dice | BF |
|---|---|---|---|---|
| smartdoc | u2net | 0.9639 | 0.9813 | 0.1107 |
| smartdoc | yolo | 0.9398 | 0.9690 | 0.1096 |
| doc3d | u2net | 0.5825 | 0.6765 | 0.1475 |
| doc3d | yolo | 0.7250 | 0.7621 | 0.3437 |

## End-to-End

| Pipeline | N | Median (ms) | PSNR | SSIM | CER |
|---|---|---|---|---|---|
| u2net | 620 | 87.9 | 0.00 | 0.0000 | 0.0000 |
| yolo | 620 | 144.7 | 0.00 | 0.0000 | 0.0000 |

## Figures

- ![Accuracy](fig_accuracy.png)
- ![Speed](fig_speed.png)