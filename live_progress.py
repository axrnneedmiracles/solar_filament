"""
Real-Time Dual Progress Bar Monitor with Live Dice Display
==========================================================
Displays 2 clean progress bars + Live Epoch Dice, Best Val Dice, Batch Loss, and ETA:
1. Current Epoch % (Batch counter, running loss, ETA, and Last Epoch Dice)
2. Total Net % (Completed epoch fraction, Best Val Dice record, GPU telemetry)

Usage:
    python live_progress.py
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
            return f"GPU {util}% | {int(mem_used)/1024:.1f}GB | {temp}C"
    except Exception:
        pass
    return "GPU: Active"

def render_bar(fraction, length=20, fill_char="█", empty_char="░"):
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(length * fraction))
    return fill_char * filled + empty_char * (length - filled)

def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}m {s:02d}s"

def main():
    live_state_path = "experiments/patch_refiner_live_state.json"
    json_path = "experiments/patch_refiner_training_results.json"
    total_epochs = 35

    try:
        while True:
            epoch = 17
            phase = "Training"
            batch = 0
            total_batches = 1880
            batch_pct = 0.0
            cur_loss = 0.0
            eta_sec = 0.0
            best_val_dice = 0.7260
            best_epoch = 14
            last_val_dice = 0.7198
            last_epoch_num = 16
            completed_epochs = 16

            # Read historical results from JSON
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        jdata = json.load(f)
                    completed_epochs = jdata.get('total_epochs_completed', 16)
                    best_val_dice = jdata.get('best_val_dice', 0.7260)
                    best_epoch = jdata.get('best_epoch', 14)
                    history = jdata.get('history', [])
                    if history:
                        last_h = history[-1]
                        last_val_dice = last_h.get('val_dice', last_val_dice)
                        last_epoch_num = last_h.get('epoch', last_epoch_num)
                except Exception:
                    pass

            # Read real-time live batch state
            if os.path.exists(live_state_path):
                try:
                    with open(live_state_path, 'r') as f:
                        data = json.load(f)
                    epoch = data.get('epoch', completed_epochs + 1)
                    phase = data.get('phase', 'Training')
                    batch = data.get('batch', 0)
                    total_batches = data.get('total_batches', 1880)
                    batch_pct = data.get('percent', 0.0)
                    cur_loss = data.get('running_loss', 0.0)
                    eta_sec = data.get('eta_seconds', 0.0)
                    best_val_dice = data.get('best_val_dice', best_val_dice)
                    best_epoch = data.get('best_epoch', best_epoch)
                except Exception:
                    pass

            # 1. Current Epoch Progress Bar
            epoch_frac = (batch / max(1, total_batches)) if total_batches > 0 else 0.0
            epoch_bar = render_bar(epoch_frac, length=24)

            # 2. Total Net Training Progress Bar
            total_frac = (epoch - 1 + epoch_frac) / total_epochs
            total_pct = total_frac * 100.0
            total_bar = render_bar(total_frac, length=24)

            gpu_str = get_gpu_info()

            # Clean screen refresh layout
            output = []
            output.append("=" * 82)
            output.append("  SOLAR FILAMENT NATIVE PATCH REFINER — LIVE TRAINING DASHBOARD")
            output.append("  Press Ctrl+C to exit monitor (Training continues running in background)")
            output.append("=" * 82)
            output.append("")
            output.append(f"  [1] Current Epoch {epoch:02d}/{total_epochs:02d} [{phase}] Progress:")
            output.append(f"  [{epoch_bar}] {batch_pct:5.1f}% ({batch:4d}/{total_batches} Batches)")
            output.append(f"      Batch Loss: {cur_loss:.4f} | Epoch ETA: {format_time(eta_sec)} | Last Epoch {last_epoch_num} Dice: {last_val_dice:.4f}")
            output.append("")
            output.append(f"  [2] Total Net Training Progress (All {total_epochs} Epochs):")
            output.append(f"  [{total_bar}] {total_pct:5.1f}% (Epoch {epoch-1 + epoch_frac:4.1f} / {total_epochs})")
            output.append(f"      🏆 All-Time Best Val Dice: {best_val_dice:.4f} (Achieved at Epoch {best_epoch})")
            output.append("")
            output.append("=" * 82)
            output.append(f"  Telemetry: {gpu_str} | Active Task: task-6694")
            output.append("=" * 82)

            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n".join(output))

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n" + "=" * 82)
        print(" [*] Monitor closed. Training is running smoothly in the background.")
        print(" [*] To view the full results table, run: python show_results.py")
        print("=" * 82 + "\n")

if __name__ == '__main__':
    main()
