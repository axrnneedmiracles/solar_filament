"""
Training Graph & Model Comparison Visualizer
============================================
Generates full-trajectory training and validation curves for each model
and multi-model comparative benchmarks for Dice, IoU, Precision, and Recall.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_model_curves(history, model_id, model_name, output_dir, baseline_dice=0.6990):
    """Plots individual model training curves across all epochs."""
    model_dir = os.path.join(output_dir, model_id)
    ensure_dir(model_dir)

    epochs = [h['epoch'] for h in history]
    train_loss = [h['train']['loss'] for h in history]
    val_loss = [h['val']['loss'] for h in history]
    train_dice = [h['train']['dice'] for h in history]
    val_dice = [h['val']['dice'] for h in history]
    val_iou = [h['val']['iou'] for h in history]
    val_prec = [h['val']['precision'] for h in history]
    val_rec = [h['val']['recall'] for h in history]

    best_val_epoch_idx = int(np.argmax(val_dice))
    best_ep = epochs[best_val_epoch_idx]
    best_d = val_dice[best_val_epoch_idx]
    best_l = val_loss[best_val_epoch_idx]

    # 1. Validation Dice vs Epoch (Key Presentation Graph)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs, val_dice, 's-', color='#8e44ad', linewidth=2.2, label=f'{model_name} (Val Dice)')
    ax.plot(epochs, train_dice, 'o--', color='#27ae60', linewidth=1.5, alpha=0.7, label='Train Dice')
    ax.axhline(baseline_dice, color='#e67e22', linestyle=':', linewidth=2, label=f'Baseline Dice ({baseline_dice:.4f})')
    ax.plot(best_ep, best_d, 'r*', markersize=16, label=f'Best: Epoch {best_ep} (Dice={best_d:.4f})')
    ax.set_title(f'{model_name}\nValidation Dice Similarity Coefficient vs Epoch', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Dice Score', fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'validation_dice_vs_epoch.png'), dpi=200)
    plt.close()

    # 2. Validation Loss vs Epoch
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs, train_loss, 'o-', color='#e74c3c', linewidth=2, label='Train Loss')
    ax.plot(epochs, val_loss, 's--', color='#2980b9', linewidth=2.2, label='Val Loss')
    ax.plot(best_ep, best_l, 'k*', markersize=14, label=f'Best Dice Epoch {best_ep} (Val Loss={best_l:.4f})')
    ax.set_title(f'{model_name}\nLoss Progression across Epochs', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Total Loss', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'validation_loss_vs_epoch.png'), dpi=200)
    plt.close()

    # 3. 4-Panel Overview Dashboard Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    
    # Loss
    axes[0, 0].plot(epochs, train_loss, 'o-', color='#e74c3c', label='Train Loss')
    axes[0, 0].plot(epochs, val_loss, 's--', color='#2980b9', label='Val Loss')
    axes[0, 0].set_title('Loss Curves', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Dice
    axes[0, 1].plot(epochs, train_dice, 'o--', color='#27ae60', label='Train Dice')
    axes[0, 1].plot(epochs, val_dice, 's-', color='#8e44ad', label='Val Dice')
    axes[0, 1].axhline(baseline_dice, color='#e67e22', linestyle=':', label=f'Baseline ({baseline_dice:.4f})')
    axes[0, 1].plot(best_ep, best_d, 'r*', markersize=12)
    axes[0, 1].set_title(f'Dice Score (Best: {best_d:.4f})', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # IoU
    axes[1, 0].plot(epochs, val_iou, 'o-', color='#16a085', label='Val IoU')
    axes[1, 0].set_title('Validation IoU (Jaccard Index)', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Precision & Recall
    axes[1, 1].plot(epochs, val_prec, 'o-', color='#34495e', label='Precision')
    axes[1, 1].plot(epochs, val_rec, 's-', color='#d35400', label='Recall')
    axes[1, 1].set_title('Precision vs Recall Dynamics', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'{model_name} — Complete Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'complete_training_history.png'), dpi=200)
    plt.close()


def generate_comparison_charts(results_json_path, output_dir):
    """Generates cross-model benchmark comparison graphs."""
    comp_dir = os.path.join(output_dir, 'comparison')
    ensure_dir(comp_dir)

    with open(results_json_path, 'r') as f:
        data = json.load(f)

    models = [exp for exp in data['experiments'] if 'metrics' in exp]
    names = [m['name'].replace('Model ', 'M') for m in models]
    dices = [m['metrics']['dice'] for m in models]
    ious = [m['metrics']['iou'] for m in models]
    precs = [m['metrics']['precision'] for m in models]
    recs = [m['metrics']['recall'] for m in models]

    # 1. Dice Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#95a5a6', '#3498db', '#2ecc71', '#e67e22', '#9b59b6'][:len(models)]
    bars = ax.bar(names, dices, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
    ax.axhline(0.6990, color='#e74c3c', linestyle='--', linewidth=1.5, label='Original Baseline (0.6990)')
    ax.set_ylim(0.60, 0.78)
    ax.set_ylabel('Validation Dice Score', fontsize=12, fontweight='bold')
    ax.set_title('Solar Filament Segmentation — Model Architecture & Strategy Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, d in zip(bars, dices):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.004, f'{d:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.legend(fontsize=10)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, 'dice_comparison.png'), dpi=200)
    plt.close()

    # 2. Precision vs Recall Multi-Bar Chart
    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    r1 = ax.bar(x - w/2, precs, w, label='Precision', color='#34495e', edgecolor='black')
    r2 = ax.bar(x + w/2, recs, w, label='Recall', color='#e74c3c', edgecolor='black')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Validation Precision vs. Recall Comparison Across Models', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0.60, 0.82)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(r1, precs):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.003, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(r2, recs):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.003, f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, 'precision_recall_comparison.png'), dpi=200)
    plt.close()

    # 3. IoU (Jaccard Index) Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, ious, color='#16a085', width=0.55, edgecolor='black', linewidth=1.2)
    ax.set_ylim(0.48, 0.65)
    ax.set_ylabel('Validation IoU (Jaccard Index)', fontsize=12, fontweight='bold')
    ax.set_title('Validation IoU (Jaccard Overlap) Progression', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, ious):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.003, f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(comp_dir, 'iou_comparison.png'), dpi=200)
    plt.close()

    print(f"[*] Comparison charts generated in: {comp_dir}")


if __name__ == '__main__':
    res_path = 'experiments/results.json'
    out_dir = 'outputs/training_curves'
    ensure_dir(out_dir)
    if os.path.exists(res_path):
        generate_comparison_charts(res_path, out_dir)
