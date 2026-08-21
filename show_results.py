"""
Training Results Table Viewer
=============================
Displays a clean, comprehensive tabular summary of all completed training epochs,
metrics, and model performance.

Usage:
    python show_results.py
"""

import os
import sys
import json
import time

def main():
    json_path = "experiments/patch_refiner_training_results.json"
    
    if not os.path.exists(json_path):
        print(f"[!] No training results file found at {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    exp_name = data.get("experiment", "Native Patch Refiner").replace("_", " ").title()
    best_epoch = data.get("best_epoch", 1)
    best_dice = data.get("best_val_dice", 0.0)
    total_time = data.get("total_time_minutes", 0.0)
    history = data.get("history", [])

    print("\n" + "=" * 92)
    print(f"  [*] SOLAR FILAMENT TRAINING SUMMARY TABLE: {exp_name.upper()}")
    print("=" * 92)
    print(f"  [*] Total Epochs Completed : {len(history)} / 35")
    print(f"  [*] Peak Validation Dice   : {best_dice:.4f} (Achieved at Epoch {best_epoch})")
    print(f"  [*] Total Training Time    : {total_time:.1f} minutes ({total_time/60:.2f} hours)")
    print(f"  [*] Best Model Checkpoint  : checkpoints/patch_refiner_best.pth")
    print("-" * 92)
    print(f"| {'Epoch':^9} | {'Train Loss':^11} | {'Val Loss':^10} | {'Val Dice':^10} | {'Val IoU':^9} | {'Precision':^11} | {'Recall':^9} | {'Time':^10} |")
    print("-" * 92)

    for h in history:
        ep = h['epoch']
        is_best = (ep == best_epoch)
        ep_label = f"Epoch {ep} *" if is_best else f"Epoch {ep}"
        
        t_loss = f"{h['train_loss']:.4f}"
        v_loss = f"{h['val_loss']:.4f}"
        v_dice = f"{h['val_dice']:.4f}"
        v_iou = f"{h['val_iou']:.4f}"
        v_prec = f"{h['val_precision']*100:.2f}%"
        v_rec = f"{h['val_recall']*100:.2f}%"
        
        secs = h.get('time_seconds', 0)
        dur_str = f"{int(secs//60)}m {int(secs%60):02d}s"

        print(f"| {ep_label:<9} | {t_loss:^11} | {v_loss:^10} | {v_dice:^10} | {v_iou:^9} | {v_prec:^11} | {v_rec:^9} | {dur_str:^10} |")

    print("-" * 92)
    print("  * Indicates best model checkpoint saved to disk.")
    print("=" * 92 + "\n")

if __name__ == '__main__':
    main()
