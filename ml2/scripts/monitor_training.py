#!/usr/bin/env python3
"""Training Progress Monitor for YOLO & U2-Net.

Monitors active training processes, parses training metrics, writes logs,
and sends native macOS desktop notifications.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path


def send_notification(title: str, message: str, sound: str = "Glass") -> None:
    """Send native macOS desktop notification via AppleScript."""
    try:
        title_escaped = title.replace('"', '\\"')
        message_escaped = message.replace('"', '\\"')
        cmd = f'display notification "{message_escaped}" with title "{title_escaped}" sound name "{sound}"'
        subprocess.run(["osascript", "-e", cmd], check=True)
    except Exception as e:
        print(f"[Warn] Failed to send macOS notification: {e}")


def log_message(log_file: Path, message: str) -> None:
    """Write message to terminal and append to log file with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    print(formatted)
    
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(formatted + "\n")
    except Exception as e:
        print(f"[Error] Failed to write log: {e}")


def find_training_processes() -> list[dict]:
    """Find running python training processes using `ps`."""
    processes = []
    try:
        # Get active processes with PID and full command line
        output = subprocess.check_output(["ps", "-ax", "-o", "pid,command"]).decode("utf-8")
    except Exception:
        try:
            output = subprocess.check_output(["ps", "aux"]).decode("utf-8")
        except Exception as e:
            print(f"[Error] Failed to list processes: {e}")
            return []

    for line in output.splitlines():
        line = line.strip()
        if not line or "monitor_training.py" in line:
            continue
        
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        
        pid_str, cmd = parts
        if not pid_str.isdigit():
            continue
            
        pid = int(pid_str)
        
        # Match training indicators
        is_yolo = "yolo_seg/train.py" in cmd or "ultralytics" in cmd
        is_u2net = "u2net/train.py" in cmd
        
        if is_yolo or is_u2net:
            # Try to identify name and project
            name = None
            project = None
            epochs = None
            config = None
            
            # Parse arguments
            name_match = re.search(r"--name\s+([^\s]+)", cmd)
            if name_match:
                name = name_match.group(1)
                
            project_match = re.search(r"--project\s+([^\s]+)", cmd)
            if project_match:
                project = project_match.group(1)
                
            epochs_match = re.search(r"--epochs\s+(\d+)", cmd)
            if epochs_match:
                epochs = int(epochs_match.group(1))
                
            config_match = re.search(r"--config\s+([^\s]+)", cmd)
            if config_match:
                config = config_match.group(1)
                
            processes.append({
                "pid": pid,
                "cmd": cmd,
                "type": "YOLO" if is_yolo else "U2-Net",
                "name": name,
                "project": project,
                "epochs": epochs,
                "config": config
            })
            
    return processes


def locate_yolo_save_dir(workspace_root: Path, name: str | None, project: str | None) -> Path | None:
    """Find the directory where YOLO writes results, taking into account Ultralytics nesting."""
    search_dirs = [
        workspace_root / "runs" / "segment" / "ml2" / "runs" / "yolo",
        workspace_root / "runs" / "segment",
        workspace_root / "ml2" / "runs" / "yolo",
        workspace_root / "runs",
    ]
    if project:
        proj_path = Path(project)
        if proj_path.is_absolute():
            search_dirs.insert(0, proj_path)
        else:
            search_dirs.insert(0, workspace_root / proj_path)
            
    # If we have a specific name, look for it
    if name:
        for d in search_dirs:
            p = d / name
            if (p / "results.csv").exists():
                return p

    # Fallback: search recursively for any results.csv under segment/runs/ml2
    for d in search_dirs:
        if d.exists():
            for csv_path in d.glob("**/results.csv"):
                # If modified within the last day, it's likely our active training
                if time.time() - csv_path.stat().st_mtime < 86400:
                    return csv_path.parent
                    
    return None


def get_yolo_progress(save_dir: Path, target_epochs: int | None) -> dict | None:
    """Parse results.csv and args.yaml to extract current YOLO progress."""
    csv_file = save_dir / "results.csv"
    if not csv_file.exists():
        return None
        
    # Read target epochs from args.yaml if not explicitly set
    epochs = target_epochs
    args_file = save_dir / "args.yaml"
    if epochs is None and args_file.exists():
        try:
            with open(args_file, "r") as f:
                for line in f:
                    if line.strip().startswith("epochs:"):
                        epochs = int(line.split(":")[1].strip())
                        break
        except Exception:
            pass
            
    if epochs is None:
        epochs = 150  # Fallback YOLO default
        
    # Parse results.csv
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        if not rows:
            return {
                "epoch": 0,
                "total_epochs": epochs,
                "loss": 0.0,
                "map50": 0.0,
                "map50_95": 0.0,
                "box_loss": 0.0,
                "seg_loss": 0.0
            }
            
        # Get the latest row
        latest = rows[-1]
        
        # Clean headers because Ultralytics headers might contain leading/trailing whitespaces
        latest_clean = {k.strip(): v.strip() for k, v in latest.items() if k is not None}
        
        # Extract fields with safe fallbacks
        epoch = int(latest_clean.get("epoch", len(rows)))
        
        # Get metrics/mAP50-95(M) or similar
        box_loss = float(latest_clean.get("train/box_loss", 0.0))
        seg_loss = float(latest_clean.get("train/seg_loss", 0.0))
        val_box_loss = float(latest_clean.get("val/box_loss", 0.0))
        val_seg_loss = float(latest_clean.get("val/seg_loss", 0.0))
        
        # Try to find mAP keys
        map50_m = float(latest_clean.get("metrics/mAP50(M)", 0.0))
        map50_95_m = float(latest_clean.get("metrics/mAP50-95(M)", 0.0))
        map50_b = float(latest_clean.get("metrics/mAP50(B)", 0.0))
        map50_95_b = float(latest_clean.get("metrics/mAP50-95(B)", 0.0))
        
        return {
            "epoch": epoch,
            "total_epochs": epochs,
            "loss": box_loss + seg_loss,
            "box_loss": box_loss,
            "seg_loss": seg_loss,
            "val_box_loss": val_box_loss,
            "val_seg_loss": val_seg_loss,
            "map50_mask": map50_m,
            "map50_95_mask": map50_95_m,
            "map50_box": map50_b,
            "map50_95_box": map50_95_b,
        }
    except Exception as e:
        print(f"[Error] Parsing YOLO results: {e}")
        return None


def get_u2net_progress(workspace_root: Path, config_file: str | None, target_epochs: int | None) -> dict | None:
    """Check u2net checkpoints to determine progress."""
    epochs = target_epochs
    if epochs is None and config_file:
        conf_path = workspace_root / config_file
        if conf_path.exists():
            try:
                import yaml
                with open(conf_path) as f:
                    cfg = yaml.safe_load(f)
                    epochs = cfg.get("train", {}).get("epochs", None)
            except Exception:
                pass
                
    if epochs is None:
        epochs = 80  # Default U2-Net config
        
    checkpoint_dir = workspace_root / "ml2" / "checkpoints"
    if not checkpoint_dir.exists():
        return None
        
    # Look for files like u2netp_main_epoch*.pth
    checkpoints = list(checkpoint_dir.glob("u2netp_main_epoch*.pth"))
    if not checkpoints:
        return {
            "epoch": 0,
            "total_epochs": epochs,
            "latest_ckpt": "None"
        }
        
    # Extract numbers
    epoch_numbers = []
    for ckpt in checkpoints:
        match = re.search(r"epoch(\d+)", ckpt.name)
        if match:
            epoch_numbers.append(int(match.group(1)))
            
    current_epoch = max(epoch_numbers) if epoch_numbers else 0
    latest_ckpt = checkpoint_dir / f"u2netp_main_epoch{current_epoch}.pth"
    
    return {
        "epoch": current_epoch,
        "total_epochs": epochs,
        "latest_ckpt": latest_ckpt.name if latest_ckpt.exists() else "None",
        "mtime": latest_ckpt.stat().st_mtime if latest_ckpt.exists() else 0
    }


def is_pid_running(pid: int) -> bool:
    """Check if process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor Training Progress and notify on macOS.")
    parser.add_argument("--pid", type=int, help="Target process PID to monitor")
    parser.add_argument("--interval", type=int, default=600, help="Check interval in seconds (default: 600s/10m)")
    parser.add_argument("--oneshot", action="store_true", help="Print progress once and exit immediately")
    args = parser.parse_args()

    workspace_root = Path(__file__).resolve().parents[2]
    log_file = workspace_root / "ml2" / "runs" / "monitor_training.log"
    
    log_message(log_file, "=============================================")
    log_message(log_file, "Starting ML2 Training Progress Monitor")
    log_message(log_file, f"Workspace root: {workspace_root}")
    
    # 1. Identify Target Process
    target_proc = None
    running_procs = find_training_processes()
    
    if args.pid:
        # Find specified PID
        for p in running_procs:
            if p["pid"] == args.pid:
                target_proc = p
                break
        if not target_proc:
            log_message(log_file, f"Specified PID {args.pid} not found in training processes. Monitoring PID {args.pid} generically.")
            target_proc = {
                "pid": args.pid,
                "type": "Generic",
                "name": None,
                "project": None,
                "epochs": None,
                "config": None,
                "cmd": "Unknown"
            }
    else:
        # Auto-detect
        if not running_procs:
            log_message(log_file, "No active training processes found (train.py).")
            # Try to see if there is a recently updated results.csv to report progress
            save_dir = locate_yolo_save_dir(workspace_root, None, None)
            if save_dir:
                prog = get_yolo_progress(save_dir, None)
                if prog:
                    log_message(log_file, f"Found recently active YOLO directory: {save_dir}")
                    log_message(log_file, f"Latest Progress: Epoch {prog['epoch']}/{prog['total_epochs']}")
                    if 'map50_95_mask' in prog:
                        log_message(log_file, f"Metrics: Box Loss={prog['box_loss']:.4f}, Seg Loss={prog['seg_loss']:.4f}, mAP50-95(Mask)={prog['map50_95_mask']:.4f}")
            send_notification("ML2 Monitor", "Không thấy tiến trình training nào đang chạy.", "Basso")
            return
            
        # Select the first one found (usually the active training)
        target_proc = running_procs[0]
        log_message(log_file, f"Auto-detected active training process:")
        log_message(log_file, f"  - PID: {target_proc['pid']}")
        log_message(log_file, f"  - Type: {target_proc['type']}")
        log_message(log_file, f"  - Command: {target_proc['cmd']}")

    pid = target_proc["pid"]
    train_type = target_proc["type"]
    
    # 2. Perform first progress check
    save_dir = None
    if train_type == "YOLO":
        save_dir = locate_yolo_save_dir(workspace_root, target_proc["name"], target_proc["project"])
        if save_dir:
            log_message(log_file, f"Located YOLO run directory: {save_dir}")
        else:
            log_message(log_file, "[Warn] Could not locate YOLO run directory yet. Will retry...")
            
    # Initial status display
    def check_and_report() -> tuple[bool, int, int]:
        """Check progress, log, and send notification. Returns (is_success, current_epoch, total_epochs)."""
        nonlocal save_dir
        
        # Check process status
        alive = is_pid_running(pid)
        
        # Try to read progress
        current_epoch = 0
        total_epochs = 150 # default
        
        if train_type == "YOLO":
            if not save_dir:
                save_dir = locate_yolo_save_dir(workspace_root, target_proc["name"], target_proc["project"])
            if save_dir:
                prog = get_yolo_progress(save_dir, target_proc["epochs"])
                if prog:
                    current_epoch = prog["epoch"]
                    total_epochs = prog["total_epochs"]
                    
                    status_str = f"Epoch {current_epoch}/{total_epochs} | Box Loss: {prog['box_loss']:.4f} | Seg Loss: {prog['seg_loss']:.4f}"
                    if prog.get("map50_95_mask", 0.0) > 0.0:
                        status_str += f" | mAP50-95(Mask): {prog['map50_95_mask']:.4f}"
                    
                    log_message(log_file, f"[YOLO Progress] {status_str}")
                    
                    # Notify
                    noti_msg = f"Epoch {current_epoch}/{total_epochs}\nBox Loss: {prog['box_loss']:.4f} | Seg Loss: {prog['seg_loss']:.4f}"
                    if prog.get("map50_95_mask", 0.0) > 0.0:
                        noti_msg += f"\nmAP50-95(Mask): {prog['map50_95_mask']:.4f}"
                    
                    send_notification(f"YOLO Training {target_proc['name'] or ''}", noti_msg)
                else:
                    log_message(log_file, "[YOLO] Results file empty or not created yet.")
            else:
                log_message(log_file, "[YOLO] Training directory not created yet.")
                
        elif train_type == "U2-Net":
            prog = get_u2net_progress(workspace_root, target_proc["config"], target_proc["epochs"])
            if prog:
                current_epoch = prog["epoch"]
                total_epochs = prog["total_epochs"]
                log_message(log_file, f"[U2-Net Progress] Epoch {current_epoch}/{total_epochs} | Latest Checkpoint: {prog['latest_ckpt']}")
                send_notification("U2-Net Training", f"Epoch {current_epoch}/{total_epochs}\nLatest Checkpoint: {prog['latest_ckpt']}")
            else:
                log_message(log_file, "[U2-Net] Checkpoints directory or files not found.")
                
        else: # Generic PID monitoring
            log_message(log_file, f"[Generic Progress] Process PID {pid} is {'RUNNING' if alive else 'STOPPED'}")
            send_notification("Training Monitor", f"Tiến trình PID {pid} đang hoạt động.")
            
        return alive, current_epoch, total_epochs

    if args.oneshot:
        check_and_report()
        log_message(log_file, "Oneshot check completed. Exiting.")
        return
        
    # 3. Main Loop
    log_message(log_file, f"Starting periodic monitoring loop every {args.interval} seconds ({(args.interval / 60):.1f}m)...")
    
    # Notify startup
    send_notification("ML2 Monitor Started", f"Đang theo dõi PID {pid} ({train_type})")
    
    try:
        consecutive_dead_checks = 0
        while True:
            alive, current_epoch, total_epochs = check_and_report()
            
            if not alive:
                # Double check to prevent temporary race conditions
                time.sleep(5)
                if not is_pid_running(pid):
                    consecutive_dead_checks += 1
                    if consecutive_dead_checks >= 2:
                        # Process terminated
                        log_message(log_file, f"Process PID {pid} has terminated.")
                        
                        # Decide if it completed successfully
                        # For YOLO/U2NET, check if we reached total epochs or within 1 epoch of completion
                        if current_epoch >= total_epochs - 1 and current_epoch > 0:
                            success_msg = f"Training completed successfully! Finished all {total_epochs} epochs."
                            log_message(log_file, f"[SUCCESS] {success_msg}")
                            send_notification("Training COMPLETED! 🎉", success_msg, sound="Glass")
                        else:
                            fail_msg = f"Training stopped unexpectedly at Epoch {current_epoch}/{total_epochs}."
                            log_message(log_file, f"[WARNING] {fail_msg}")
                            send_notification("Training STOPPED! ⚠️", fail_msg, sound="Basso")
                        break
            else:
                consecutive_dead_checks = 0
                
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        log_message(log_file, "Monitor stopped by user (Ctrl+C).")
        send_notification("ML2 Monitor Stopped", "Tiến trình giám sát đã tắt.")
        
    log_message(log_file, "Monitor exiting.")


if __name__ == "__main__":
    main()
