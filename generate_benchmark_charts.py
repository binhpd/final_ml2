import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Thiết lập style cho biểu đồ đẹp mắt
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

def generate_charts():
    mps_path = "ml2/results/compare_5models_mps.csv"
    cpu_path = "ml2/results/compare_5models_cpu.csv"
    out_dir = "ml2/results/charts"
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(mps_path) or not os.path.exists(cpu_path):
        print("❌ Error: CSV benchmark files not found. Please run run_full_benchmark.py first.")
        return
        
    df_mps = pd.read_csv(mps_path)
    df_cpu = pd.read_csv(cpu_path)
    
    # Chuẩn hóa tên model hiển thị
    model_mapping = {
        "yolo_old": "YOLOv11n-seg (COCO)",
        "u2net_old": "U2-Net Full (Salient)",
        "u2net_trained": "U2-Netp Lite (Trained)",
        "yolo_trained": "YOLOv11n-seg (Trained)",
        "yolo_tuned": "YOLOv11n-seg (Spine Tuned)"
    }
    
    df_mps['model_display'] = df_mps['model'].map(model_mapping)
    df_cpu['model_display'] = df_cpu['model'].map(model_mapping)
    
    # ----------------------------------------------------
    # BIỂU ĐỒ 1: SO SÁNH LATENCY (CPU VS MPS)
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df_mps['model_display'].tolist()
    lat_mps = df_mps['lat_ms'].tolist()
    # Map CPU latency to match the models order in MPS
    lat_cpu = [df_cpu[df_cpu['model'] == m]['lat_ms'].values[0] for m in df_mps['model'].tolist()]
    
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, lat_mps, width, label='GPU / MPS (Mac Studio)', color='#1f77b4')
    rects2 = ax.bar(x + width/2, lat_cpu, width, label='CPU (Mac Studio)', color='#ff7f0e')
    
    ax.set_ylabel('Inference Latency (ms) - log scale')
    ax.set_yscale('log') # Dùng log scale vì sự chênh lệch quá lớn giữa CPU và GPU
    ax.set_title('So sánh tốc độ suy luận (Latency) trên CPU vs MPS (MPS)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    
    # Thêm số liệu lên đầu cột
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}ms',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "benchmark_latency_comparison.png"), dpi=200)
    plt.close()
    print("Saved latency comparison chart.")

    # ----------------------------------------------------
    # BIỂU ĐỒ 2: SO SÁNH IoU TRÊN CÁC TẬP DỮ LIỆU
    # ----------------------------------------------------
    # Bộ dữ liệu IoU lấy từ kết quả thực tế phân tách
    datasets_iou = {
        'SmartDoc (Ảnh phẳng thật)': {
            'YOLOv11n-seg (COCO)': 0.261,
            'U2-Net Full (Salient)': 0.928,
            'U2-Netp Lite (Trained)': 0.974,
            'YOLOv11n-seg (Trained)': 0.940,
            'YOLOv11n-seg (Spine Tuned)': 0.233
        },
        'kaggle_real (Ảnh chụp thật)': {
            'YOLOv11n-seg (COCO)': 0.667,
            'U2-Net Full (Salient)': 0.863,
            'U2-Netp Lite (Trained)': 0.972,
            'YOLOv11n-seg (Trained)': 0.960,
            'YOLOv11n-seg (Spine Tuned)': 0.685
        },
        'Doc3D (Ảnh cong 3D)': {
            'YOLOv11n-seg (COCO)': 0.679,
            'U2-Net Full (Salient)': 0.953,
            'U2-Netp Lite (Trained)': 0.733,
            'YOLOv11n-seg (Trained)': 0.709,
            'YOLOv11n-seg (Spine Tuned)': 0.696
        }
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ds_names = list(datasets_iou.keys())
    model_names = list(model_mapping.values())
    
    x = np.arange(len(ds_names))
    width = 0.15
    
    colors = ['#d62728', '#9467bd', '#2ca02c', '#bcbd22', '#8c564b']
    
    for idx, model_name in enumerate(model_names):
        iou_vals = [datasets_iou[ds][model_name] for ds in ds_names]
        ax.bar(x + (idx - 2) * width, iou_vals, width, label=model_name, color=colors[idx])
        
    ax.set_ylabel('mIoU (Mean Intersection over Union)')
    ax.set_title('So sánh chất lượng phân vùng (mIoU) trên từng Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(ds_names)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower left')
    
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "benchmark_iou_comparison.png"), dpi=200)
    plt.close()
    print("Saved IoU comparison chart.")

    # ----------------------------------------------------
    # BIỂU ĐỒ 3: TRADE-OFF (ACCURACY VS SPEED & SIZE)
    # ----------------------------------------------------
    # Sử dụng mIoU trên ảnh thật làm trục Y, Latency trên MPS làm trục X, Size của bong bóng đại diện cho Params
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Tính mIoU trung bình trên ảnh thật (SmartDoc & kaggle_real)
    # SmartDoc và kaggle_real là ảnh thực tế
    real_iou = {
        "yolo_old": 0.464,
        "u2net_old": 0.896,
        "u2net_trained": 0.973,
        "yolo_trained": 0.950,
        "yolo_tuned": 0.459 # Bị thấp do nhãn 1 lớp, ta sẽ chú thích điểm này
    }
    
    # u2net_old size: 176MB, params: 44M
    # u2net_trained size: 4.7MB, params: 1.1M
    # yolo_trained size: 6.0MB, params: 2.8M
    # yolo_old size: 6.0MB, params: 2.8M
    # yolo_tuned size: 6.0MB, params: 2.8M
    
    x_lat = df_mps['lat_ms'].tolist()
    y_iou = [real_iou[m] for m in df_mps['model'].tolist()]
    sizes = [m["params"]/1e6 * 50 for idx, m in df_mps.iterrows()] # Tỷ lệ size bong bóng theo số triệu params
    labels = df_mps['model_display'].tolist()
    
    scatter = ax.scatter(x_lat, y_iou, s=sizes, alpha=0.6, c=colors[:len(labels)], edgecolors='black')
    
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x_lat[i], y_iou[i]), xytext=(10, 0), textcoords='offset points', va='center')
        
    ax.set_xlabel('Inference Latency (ms) - GPU / MPS')
    ax.set_ylabel('mIoU (Trung bình 2 tập ảnh thật)')
    ax.set_title('Đánh đổi hiệu năng: Độ chính xác vs Tốc độ & Kích thước mô hình')
    ax.set_ylim(0.3, 1.05)
    ax.set_xlim(min(x_lat) - 2, max(x_lat) + 5)
    
    # Tạo legend cho kích thước bong bóng
    p1 = ax.scatter([], [], s=1.1*50, c='gray', alpha=0.6, label='U2-Netp Lite (~1.1M Params)')
    p2 = ax.scatter([], [], s=2.8*50, c='gray', alpha=0.6, label='YOLOv11n-seg (~2.8M Params)')
    p3 = ax.scatter([], [], s=44*50, c='gray', alpha=0.6, label='U2-Net Full (~44M Params)')
    ax.legend(handles=[p1, p2, p3], loc='lower right', title='Kích thước mô hình')
    
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "benchmark_tradeoff_accuracy_speed.png"), dpi=200)
    plt.close()
    print("Saved accuracy-speed trade-off chart.")

if __name__ == "__main__":
    generate_charts()
