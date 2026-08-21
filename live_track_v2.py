"""
Real-Time Dual Progress Bar Live Monitor (Next-Gen Champion Training)
====================================================================
Displays 2 clean progress bars + Live Metrics, Loss, and GPU Telemetry:
1. Current Epoch % (Batch counter, live batch loss, Epoch ETA, Last Epoch Dice)
2. Total Net % (All epochs progress bar, All-Time Best Val Dice, Adaptive Patience)

Usage:
    python live_track_v2.py
"""

import os
import sys
import time
import json
import subprocess

def get_gpu_info():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if res.returncode == 0:
            util, mem_used, temp = res.stdout.strip().split(", ")
            return f"GPU {util}% | {int(mem_used)/1024:.1f}GB | {temp}°C"
    except Exception:
        pass
    return "NVIDIA RTX 4050 (Active)"

def render_bar(fraction, length=24, fill_char="█", empty_char="░"):
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(length * fraction))
    return fill_char * filled + empty_char * (length - filled)

def format_time(seconds):
    if seconds <= 0:
        return "--m --s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}m {s:02d}s"

def main():
    live_batch_file = "experiments/champion_v2_live_batch.json"
    progress_file = "experiments/champion_v2_live_progress.json"
    total_epochs = 25

    try:
        while True:
            epoch = 1
            phase = "Training"
            batch = 0
            total_batches = 1880
            batch_pct = 0.0
            cur_loss = 0.0
            eta_sec = 0.0
            best_val_dice = 0.7304
            best_epoch = 0
            last_val_dice = 0.7304
            last_epoch_num = 0
            patience_counter = 0
            max_patience = 5
            history = []

            # 1. Read fine-grained live batch state
            if os.path.exists(live_batch_file):
                try:
                    with open(live_batch_file, 'r') as f:
                        bdata = json.load(f)
                    epoch = bdata.get('epoch', 1)
                    total_epochs = bdata.get('max_epochs', 25)
                    phase = bdata.get('phase', 'Training')
                    batch = bdata.get('batch', 0)
                    total_batches = bdata.get('total_batches', 1880)
                    batch_pct = bdata.get('percent', 0.0)
                    cur_loss = bdata.get('running_loss', 0.0)
                    eta_sec = bdata.get('eta_seconds', 0.0)
                    best_val_dice = bdata.get('best_val_dice', 0.7304)
                    best_epoch = bdata.get('best_epoch', 0)
                    last_val_dice = bdata.get('last_val_dice', 0.7304)
                    last_epoch_num = bdata.get('last_epoch_num', 0)
                    patience_counter = bdata.get('patience_counter', 0)
                    max_patience = bdata.get('max_patience', 5)
                except Exception:
                    pass

            # 2. Read full epoch history
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r') as f:
                        pdata = json.load(f)
                    history = pdata.get('history', [])
                    if history:
                        last_h = history[-1]
                        last_val_dice = last_h.get('val_dice', last_val_dice)
                        last_epoch_num = last_h.get('epoch', last_epoch_num)
                except Exception:
                    pass

            # Progress Bars
            epoch_frac = (batch / max(1, total_batches)) if total_batches > 0 else 0.0
            epoch_bar = render_bar(epoch_frac, length=24)

            total_frac = (epoch - 1 + epoch_frac) / max(1, total_epochs)
            total_pct = total_frac * 100.0
            total_bar = render_bar(total_frac, length=24)

            gpu_str = get_gpu_info()

            # Format Screen
            output = []
            output.append("=" * 82)
            output.append("  SOLAR FILAMENT NEXT-GEN CHAMPION — LIVE DUAL PROGRESS MONITOR")
            output.append("  Target: 0.80+ Dice | Press Ctrl+C to exit viewer (Training keeps running)")
            output.append("=" * 82)
            output.append("")
            output.append(f"  [1] Current Epoch {epoch:02d}/{total_epochs:02d} [{phase}] Progress:")
            output.append(f"  [{epoch_bar}] {batch_pct:5.1f}% ({batch:4d}/{total_batches} Batches)")
            last_dice_str = f"Epoch {last_epoch_num} Dice: {last_val_dice:.4f}" if last_epoch_num > 0 else "Base Dice: 0.7304"
            output.append(f"      Batch Loss: {cur_loss:.4f} | Epoch ETA: {format_time(eta_sec)} | Last: {last_dice_str}")
            output.append("")
            output.append(f"  [2] Total Net Training Progress (Target {total_epochs} Epochs max):")
            output.append(f"  [{total_bar}] {total_pct:5.1f}% (Epoch {epoch-1 + epoch_frac:4.1f} / {total_epochs})")
            output.append(f"      🏆 All-Time Best Val Dice: {best_val_dice:.4f} | Adaptive Patience: {patience_counter}/{max_patience}")
            output.append("")
            output.append("=" * 82)
            output.append(f"  Telemetry: {gpu_str} | Hardware: NVIDIA RTX 4050 GPU")
            output.append("=" * 82)

            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n".join(output))

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n" + "=" * 82)
        print(" [*] Live monitor closed. Training is running smoothly in the background.")
        print(" [*] To reopen the monitor anytime, run: python live_track_v2.py")
        print("=" * 82 + "\n")

if __name__ == '__main__':
    main()
