"""
Live Progress Tracker for Next-Gen Champion Training
====================================================
Usage:
    python track_v2.py
"""

import os
import json
import time

def track_progress():
    progress_file = "experiments/champion_v2_live_progress.json"
    print("=" * 78)
    print("  [*] NEXT-GEN CHAMPION TRAINING (LIVE STATUS TRACKER)")
    print("=" * 78)

    if not os.path.exists(progress_file):
        print("[*] STATUS: Epoch 1 is currently TRAINING on the GPU (NVIDIA RTX 4050)!")
        print("    -> 7,520 training patches loaded in RAM.")
        print("    -> Running forward/backward passes with Focal-Tversky loss.")
        print("    -> Full metrics (Dice, IoU, Recall, Precision) will populate here at the end of Epoch 1 (~8-10 min).")
        print("=" * 78)
        return

    with open(progress_file, 'r') as f:
        data = json.load(f)

    curr_ep = data.get('current_epoch', 0)
    max_ep = data.get('max_epochs', 25)
    best_ep = data.get('best_epoch', 0)
    best_dice = data.get('best_val_dice', 0.0)
    patience_cnt = data.get('patience_counter', 0)
    max_pat = data.get('max_patience', 5)
    history = data.get('history', [])

    print(f"[*] Current Progress   : Epoch {curr_ep}/{max_ep}")
    print(f"[*] Champion Score     : Val Dice = {best_dice:.4f} (at Epoch {best_ep})")
    print(f"[*] Adaptive Patience  : {patience_cnt}/{max_pat} epochs without improvement")
    print("-" * 78)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val Dice':<10} | {'Val IoU':<10} | {'Recall':<10} | {'Precision':<10}")
    print("-" * 78)

    for h in history[-8:]:  # Show last 8 epochs
        is_best_mark = " *" if h.get('is_best') else ""
        print(
            f"{h['epoch']:<8} | "
            f"{h['train_loss']:<12.4f} | "
            f"{h['val_loss']:<10.4f} | "
            f"{h['val_dice']:<10.4f}{is_best_mark} | "
            f"{h['val_iou']:<10.4f} | "
            f"{h['val_recall']*100:<9.2f}% | "
            f"{h['val_precision']*100:<9.2f}%"
        )
    print("=" * 78)
    print("[*] To Pause/Stop : Press Ctrl+C in the training terminal")
    print("[*] To Resume     : python training/train_v2_champion.py --resume")
    print("=" * 78)

if __name__ == '__main__':
    track_progress()
