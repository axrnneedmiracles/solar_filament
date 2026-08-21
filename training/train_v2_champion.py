"""
Next-Generation Champion Model Training Pipeline (Targeting 0.80+ Dice)
======================================================================
Fine-tunes from patch_refiner_best.pth (0.7304 baseline) with:
1. Focal Tversky Loss (alpha=0.35, beta=0.65, gamma=1.33) + Morphological Boundary Loss
2. Sub-Pixel Boundary Refinement Head
3. Copy-Paste / CutMix Chromospheric Augmentations
4. Adaptive Early Stopping (Patience = 5) - stops automatically when metrics stop rising
5. Real-Time Telemetry to experiments/champion_v2_live_progress.json
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mask2former import build_mask2former
from training.losses import FocalTverskyBoundaryLoss
from training.metrics import compute_all_metrics
from training.patch_dataset import get_patch_dataloaders


def train_champion_v2(
    cache_dir: str = "cache_patch_512",
    init_checkpoint: str = "checkpoints/patch_refiner_best.pth",
    batch_size: int = 4,
    grad_accum_steps: int = 2,
    max_epochs: int = 25,
    patience: int = 5,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    results_file: str = "experiments/champion_v2_results.json",
    live_progress_file: str = "experiments/champion_v2_live_progress.json",
    resume: bool = False
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("=" * 78, flush=True)
    print(f"[*] TRAINING NEXT-GEN CHAMPION MODEL (Targeting 0.80+ Dice)", flush=True)
    print(f"[*] Compute Device   : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"[*] Loss Formulation : Focal-Tversky (alpha=0.35, beta=0.65, gamma=1.33) + Boundary Loss", flush=True)
    print(f"[*] Base Checkpoint  : {init_checkpoint}", flush=True)
    print(f"[*] Adaptive Patience: {patience} epochs (Stops if no improvement for {patience} epochs)", flush=True)
    print("=" * 78, flush=True)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # 1. Load Data with RAM caching & Copy-Paste Augmentations
    train_loader, val_loader = get_patch_dataloaders(
        cache_dir=cache_dir,
        batch_size=batch_size,
        preload=True
    )
    print(f"[+] Loaded {len(train_loader.dataset)} training patches & {len(val_loader.dataset)} validation patches.", flush=True)

    # 2. Build Model with Sub-Pixel Boundary Refinement
    model_config = {
        'name': 'mask2former',
        'backbone': 'resnet34',
        'pretrained': True,
        'in_channels': 1,
        'hidden_dim': 128,
        'num_queries': 20,
        'num_decoder_layers': 3,
        'use_boundary_refiner': True
    }
    model = build_mask2former(model_config).to(device)

    # Load starting weights from best checkpoint
    if os.path.exists(init_checkpoint):
        ckpt = torch.load(init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"[+] Successfully loaded baseline weights from {init_checkpoint} (Base Val Dice: {ckpt.get('val_dice', 0.7304):.4f})", flush=True)

    # 3. Loss & Optimizer
    criterion = FocalTverskyBoundaryLoss(
        tversky_weight=0.70,
        boundary_weight=0.30,
        alpha=0.35,
        beta=0.65,
        gamma=1.33
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=1e-7)
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    best_val_dice = 0.7304  # Beat baseline
    best_epoch = 0
    patience_counter = 0
    history = []
    start_epoch = 1

    latest_ckpt_path = os.path.join(checkpoint_dir, "champion_v2_latest.pth")
    best_ckpt_path = os.path.join(checkpoint_dir, "champion_v2_best.pth")

    if resume and os.path.exists(latest_ckpt_path):
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_dice = ckpt.get('best_val_dice', 0.7304)
        best_epoch = ckpt.get('best_epoch', 0)
        patience_counter = ckpt.get('patience_counter', 0)
        history = ckpt.get('history', [])
        print(f"[+] RESUMING TRAINING from Epoch {start_epoch}/{max_epochs} (Best Dice so far: {best_val_dice:.4f} at Epoch {best_epoch})", flush=True)

    for epoch in range(start_epoch, max_epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss_accum = 0.0
        train_batches = 0

        optimizer.zero_grad()

        for step, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                logits = model(images)
                loss = criterion(logits, masks)
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss_accum += loss.item()
            train_batches += 1

            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                elapsed = time.time() - epoch_start_time
                steps_done = step + 1
                total_steps = len(train_loader)
                sec_per_step = elapsed / max(1, steps_done)
                eta_s = (total_steps - steps_done) * sec_per_step
                batch_pct = (steps_done / total_steps) * 100.0
                try:
                    with open("experiments/champion_v2_live_batch.json", 'w') as f_b:
                        json.dump({
                            'epoch': epoch,
                            'max_epochs': max_epochs,
                            'phase': 'Training',
                            'batch': steps_done,
                            'total_batches': total_steps,
                            'percent': round(batch_pct, 1),
                            'running_loss': round(loss.item(), 4),
                            'eta_seconds': round(eta_s, 1),
                            'best_val_dice': round(best_val_dice, 4),
                            'best_epoch': best_epoch,
                            'last_val_dice': round(history[-1]['val_dice'] if history else 0.7304, 4),
                            'last_epoch_num': history[-1]['epoch'] if history else 0,
                            'patience_counter': patience_counter,
                            'max_patience': patience
                        }, f_b)
                except Exception:
                    pass

            if (step + 1) % 400 == 0:
                print(f"  [Epoch {epoch:02d}/{max_epochs:02d}] Step {step+1:04d}/{len(train_loader):04d} | Step Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}", flush=True)

        avg_train_loss = train_loss_accum / max(train_batches, 1)

        # ── VALIDATION PHASE ──
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        all_dices, all_ious, all_precs, all_recs = [], [], [], []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    logits = model(images)
                    loss = criterion(logits, masks)

                val_loss_accum += loss.item()
                val_batches += 1

                probs = torch.sigmoid(logits).cpu().numpy()
                targets = masks.cpu().numpy()

                for b in range(probs.shape[0]):
                    m = compute_all_metrics(probs[b, 0] > 0.5, targets[b, 0] > 0.5)
                    all_dices.append(m['dice'])
                    all_ious.append(m['iou'])
                    all_precs.append(m['precision'])
                    all_recs.append(m['recall'])

        avg_val_loss = val_loss_accum / max(val_batches, 1)
        mean_dice = float(np.mean(all_dices))
        mean_iou = float(np.mean(all_ious))
        mean_prec = float(np.mean(all_precs))
        mean_rec = float(np.mean(all_recs))
        epoch_sec = time.time() - epoch_start_time

        scheduler.step()

        # Check for improvement
        is_best = mean_dice > best_val_dice
        if is_best:
            best_val_dice = mean_dice
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': mean_dice,
                'val_iou': mean_iou,
                'val_recall': mean_rec,
                'val_precision': mean_prec,
                'config': {
                    'model': model_config,
                    'data': {'image_size': 512}
                }
            }, best_ckpt_path)
            status_tag = f"[+] NEW BEST! (Dice: {mean_dice:.4f} | IoU: {mean_iou:.4f} | Rec: {mean_rec*100:.2f}% | Prec: {mean_prec*100:.2f}%)"
        else:
            patience_counter += 1
            status_tag = f"Patience: {patience_counter}/{patience} (Best: {best_val_dice:.4f} at Epoch {best_epoch})"

        # Always save latest state on every epoch for instant pause & resume
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_dice': best_val_dice,
            'best_epoch': best_epoch,
            'patience_counter': patience_counter,
            'history': history,
            'val_dice': mean_dice,
            'val_iou': mean_iou,
            'val_recall': mean_rec,
            'val_precision': mean_prec,
            'config': {'model': model_config, 'data': {'image_size': 512}}
        }, latest_ckpt_path)

        print(
            f"--> Epoch {epoch:02d}/{max_epochs:02d} [{epoch_sec/60:.1f}m] | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val Dice: {mean_dice:.4f} | Val IoU: {mean_iou:.4f} | "
            f"Val Rec: {mean_rec*100:.2f}% | Val Prec: {mean_prec*100:.2f}% | {status_tag}",
            flush=True
        )

        # Log epoch record
        epoch_record = {
            'epoch': epoch,
            'train_loss': round(avg_train_loss, 4),
            'val_loss': round(avg_val_loss, 4),
            'val_dice': round(mean_dice, 4),
            'val_iou': round(mean_iou, 4),
            'val_recall': round(mean_rec, 4),
            'val_precision': round(mean_prec, 4),
            'is_best': is_best,
            'epoch_time_seconds': round(epoch_sec, 1)
        }
        history.append(epoch_record)

        # Write live telemetry
        live_telemetry = {
            'current_epoch': epoch,
            'max_epochs': max_epochs,
            'best_epoch': best_epoch,
            'best_val_dice': round(best_val_dice, 4),
            'patience_counter': patience_counter,
            'max_patience': patience,
            'latest_dice': round(mean_dice, 4),
            'latest_iou': round(mean_iou, 4),
            'latest_recall': round(mean_rec, 4),
            'latest_precision': round(mean_prec, 4),
            'history': history
        }
        with open(live_progress_file, 'w') as f:
            json.dump(live_telemetry, f, indent=2)

        # Early stopping check
        if patience_counter >= patience:
            print(f"\n[!] Early stopping triggered! Validation Dice did not improve for {patience} consecutive epochs.", flush=True)
            print(f"[+] Peak Performance preserved at Epoch {best_epoch} (Val Dice: {best_val_dice:.4f}).", flush=True)
            break

    # Save final results
    final_summary = {
        'training_completed': True,
        'best_epoch': best_epoch,
        'best_val_dice': best_val_dice,
        'history': history,
        'checkpoint_path': best_ckpt_path
    }
    with open(results_file, 'w') as f:
        json.dump(final_summary, f, indent=2)

    print("=" * 78, flush=True)
    print(f"[+] Champion Model Training Finished! Best Val Dice: {best_val_dice:.4f} (Saved to {best_ckpt_path})", flush=True)
    print("=" * 78, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train or Resume Next-Gen Champion Model")
    parser.add_argument('--resume', action='store_true', help="Resume training from champion_v2_latest.pth")
    parser.add_argument('--epochs', type=int, default=25, help="Maximum epochs to train")
    parser.add_argument('--patience', type=int, default=5, help="Adaptive patience for early stopping")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train_champion_v2(
        cache_dir="cache_patch_512",
        init_checkpoint="checkpoints/patch_refiner_best.pth",
        batch_size=4,
        grad_accum_steps=2,
        max_epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        resume=args.resume
    )
