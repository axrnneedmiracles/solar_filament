"""
Plot Current Training Results
=============================
Parses the live training logs and plots loss, dice, IoU, precision, recall, and learning rate curves.
"""

import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np

def find_latest_training_log():
    # Search system generated task logs
    task_logs = glob.glob(r"C:\Users\aryan\.gemini\antigravity-ide\brain\*\.system_generated\tasks\task-*.log")
    # Also look in local directory
    local_logs = glob.glob("*.log") + glob.glob("logs/*.log")
    
    candidates = task_logs + local_logs
    valid_logs = []
    for c in candidates:
        try:
            with open(c, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "Starting training for" in content and "Training:" in content:
                    valid_logs.append(c)
        except Exception:
            pass
            
    if not valid_logs:
        return None
    # Sort by modification time
    valid_logs.sort(key=os.path.getmtime, reverse=True)
    return valid_logs[0]

def parse_log(log_path):
    epochs = []
    train_losses = []
    train_dices = []
    val_losses = []
    val_dices = []
    val_ious = []
    val_precs = []
    val_recs = []
    lrs = []

    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+\s+\|\s+Train Loss:\s+([\d\.]+)\s+Dice:\s+([\d\.]+)\s+\|\s+Val Loss:\s+([\d\.]+)\s+Dice:\s+([\d\.]+)\s+IoU:\s+([\d\.]+)\s+P:\s+([\d\.]+)\s+R:\s+([\d\.]+)\s+\|\s+LR:\s+([\d\.eE\-\+]+)"
    )

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ep = int(match.group(1))
                t_loss = float(match.group(2))
                t_dice = float(match.group(3))
                v_loss = float(match.group(4))
                v_dice = float(match.group(5))
                v_iou = float(match.group(6))
                v_prec = float(match.group(7))
                v_rec = float(match.group(8))
                lr = float(match.group(9))

                epochs.append(ep)
                train_losses.append(t_loss)
                train_dices.append(t_dice)
                val_losses.append(v_loss)
                val_dices.append(v_dice)
                val_ious.append(v_iou)
                val_precs.append(v_prec)
                val_recs.append(v_rec)
                lrs.append(lr)

    return {
        "epoch": epochs,
        "train_loss": train_losses,
        "train_dice": train_dices,
        "val_loss": val_losses,
        "val_dice": val_dices,
        "val_iou": val_ious,
        "val_prec": val_precs,
        "val_rec": val_recs,
        "lr": lrs,
    }

def main():
    log_path = find_latest_training_log()
    if not log_path:
        print("[!] No active training log found.")
        return

    print(f"[*] Found active training log: {log_path}")
    data = parse_log(log_path)
    
    n_epochs = len(data["epoch"])
    if n_epochs == 0:
        print("[!] Training has started, but Epoch 1 has not finished logging yet.")
        return

    print("=" * 80)
    print(f"[*] CURRENT TRAINING PROGRESS -- {n_epochs} EPOCHS COMPLETED")
    print("=" * 80)
    print(f"{'Epoch':<7} | {'Train Loss':<10} | {'Train Dice':<10} | {'Val Loss':<10} | {'Val Dice':<10} | {'Val IoU':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 80)
    for i in range(n_epochs):
        print(f"{data['epoch'][i]:<7} | {data['train_loss'][i]:<10.4f} | {data['train_dice'][i]:<10.4f} | {data['val_loss'][i]:<10.4f} | {data['val_dice'][i]:<10.4f} | {data['val_iou'][i]:<10.4f} | {data['val_prec'][i]:<10.4f} | {data['val_rec'][i]:<10.4f}")
    print("-" * 80)
    
    best_idx = np.argmax(data["val_dice"])
    print(f"[+] Best Epoch So Far: Epoch {data['epoch'][best_idx]} (Val Dice: {data['val_dice'][best_idx]:.4f}, IoU: {data['val_iou'][best_idx]:.4f}, Prec: {data['val_prec'][best_idx]:.4f}, Rec: {data['val_rec'][best_idx]:.4f})")
    print("=" * 80)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ep = data["epoch"]

    # 1. Loss
    axes[0, 0].plot(ep, data["train_loss"], "o-", color="#e74c3c", label="Train Loss", linewidth=2)
    axes[0, 0].plot(ep, data["val_loss"], "s--", color="#3498db", label="Val Loss", linewidth=2)
    axes[0, 0].set_title("Loss Curves (Dice + Focal + Boundary)", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch", fontsize=11)
    axes[0, 0].set_ylabel("Loss", fontsize=11)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Dice Score
    axes[0, 1].plot(ep, data["train_dice"], "o-", color="#2ecc71", label="Train Dice", linewidth=2)
    axes[0, 1].plot(ep, data["val_dice"], "s-", color="#9b59b6", label="Val Dice (Current)", linewidth=2.5)
    axes[0, 1].axhline(0.7249, color="#f39c12", linestyle="--", linewidth=2, label="Fixed Baseline (0.7249)")
    axes[0, 1].set_title("Dice Similarity Coefficient (DSC)", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch", fontsize=11)
    axes[0, 1].set_ylabel("Dice Score", fontsize=11)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. IoU (Jaccard Index)
    axes[1, 0].plot(ep, data["val_iou"], "o-", color="#1abc9c", label="Val IoU (Current)", linewidth=2)
    axes[1, 0].axhline(0.5723, color="#e67e22", linestyle="--", linewidth=2, label="Fixed Baseline (0.5723)")
    axes[1, 0].set_title("Validation IoU (Jaccard Index)", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch", fontsize=11)
    axes[1, 0].set_ylabel("IoU Score", fontsize=11)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Precision & Recall Trade-off
    axes[1, 1].plot(ep, data["val_prec"], "o-", color="#34495e", label="Precision", linewidth=2)
    axes[1, 1].plot(ep, data["val_rec"], "s-", color="#e74c3c", label="Recall", linewidth=2)
    axes[1, 1].axhline(0.7238, color="#34495e", linestyle=":", label="Baseline Prec (0.7238)")
    axes[1, 1].axhline(0.7351, color="#e74c3c", linestyle=":", label="Baseline Rec (0.7351)")
    axes[1, 1].set_title("Validation Precision & Recall Dynamics", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch", fontsize=11)
    axes[1, 1].set_ylabel("Score", fontsize=11)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Frangi + Hessian Multi-Channel Mask2Former Live Training Curves", fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()

    out_path = "experiments/current_training_curves.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[*] Plot saved successfully to: {out_path}")

if __name__ == "__main__":
    main()
