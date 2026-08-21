"""
High-Precision Native Patch Refiner Training Pipeline (VRAM Safe & Ultra-Fast)
==============================================================================
Trains Mask2Former (ResNet-34) on 512x512 native-resolution solar patches.
Features:
- In-memory RAM cached tensors
- PyTorch AMP (Automatic Mixed Precision fp16)
- Hybrid Loss (Soft Dice 40% + Focal 30% + Boundary 30%)
- Batch Size 4 + Gradient Accumulation 2 (Effective Batch Size 8)
- Cosine Annealing Learning Rate Schedule
- Real-time progress telemetry and checkpoint saving
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
from training.losses import DiceFocalBoundaryLoss
from training.metrics import compute_all_metrics
from training.patch_dataset import get_patch_dataloaders


def train_patch_refiner(
    cache_dir: str = "cache_patch_512",
    batch_size: int = 4,
    grad_accum_steps: int = 2,
    epochs: int = 35,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    patience: int = 10,
    results_file: str = "experiments/patch_refiner_training_results.json",
    resume: bool = False
):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("=" * 75, flush=True)
    print(f"[*] TRAINING NATIVE-RESOLUTION SOLAR FILAMENT PATCH REFINER", flush=True)
    print(f"[*] Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"[*] Micro-Batch Size: {batch_size} | Grad Accum: {grad_accum_steps} (Effective: {batch_size * grad_accum_steps}) | Epochs: {epochs}", flush=True)
    print("=" * 75, flush=True)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    # 1. Load Data with RAM caching
    train_loader, val_loader = get_patch_dataloaders(
        cache_dir=cache_dir,
        batch_size=batch_size,
        preload=True
    )
    print(f"[+] Data loaded: {len(train_loader.dataset)} train patches, {len(val_loader.dataset)} val patches.", flush=True)

    # 2. Build Model
    model_config = {
        'name': 'mask2former',
        'backbone': 'resnet34',
        'pretrained': True,
        'in_channels': 1,
        'hidden_dim': 128,
        'num_queries': 20,
        'num_decoder_layers': 3,
        'dropout': 0.1
    }
    model = build_mask2former(model_config).to(device)

    # 3. Loss & Optimizer
    criterion = DiceFocalBoundaryLoss(
        dice_weight=0.4,
        focal_weight=0.3,
        boundary_weight=0.3,
        alpha=0.75,
        gamma=2.0
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    best_val_dice = 0.0
    best_epoch = 0
    patience_counter = 0
    start_epoch = 1
    history = []

    # Resume from checkpoint if requested or latest checkpoint exists
    latest_ckpt_path = os.path.join(checkpoint_dir, "patch_refiner_latest.pth")
    live_state_file = "experiments/patch_refiner_live_state.json"

    if resume and os.path.exists(latest_ckpt_path):
        print(f"[*] Resuming training from: {latest_ckpt_path}", flush=True)
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        
        # Load existing history JSON to determine exact resume epoch & best record
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    jdata = json.load(f)
                history = jdata.get('history', [])
                best_val_dice = jdata.get('best_val_dice', ckpt.get('val_dice', 0.0))
                best_epoch = jdata.get('best_epoch', ckpt.get('epoch', 0))
                start_epoch = jdata.get('total_epochs_completed', ckpt.get('epoch', 0)) + 1
            except Exception:
                start_epoch = ckpt.get('epoch', 0) + 1
                best_val_dice = ckpt.get('val_dice', 0.0)
                best_epoch = ckpt.get('epoch', 0)
        else:
            start_epoch = ckpt.get('epoch', 0) + 1
            best_val_dice = ckpt.get('val_dice', 0.0)
            best_epoch = ckpt.get('epoch', 0)

        # Advance scheduler to current epoch
        for _ in range(start_epoch - 1):
            scheduler.step()

        print(f"[+] Successfully resumed at Epoch {start_epoch} | Best Val Dice: {best_val_dice:.4f} (Epoch {best_epoch})", flush=True)

    total_start_time = time.time()

    print("\n[*] Commencing Training Loop...\n", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # ── Training Phase ──
        model.train()
        train_loss = 0.0
        num_train_samples = 0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                logits = model(images)
                loss = criterion(logits, masks)
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            bs = images.size(0)
            train_loss += loss.item() * bs
            num_train_samples += bs

            # Update live state every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                elapsed_sec = time.time() - epoch_start
                batches_done = batch_idx + 1
                total_b = len(train_loader)
                pct = (batches_done / total_b) * 100.0
                cur_loss = train_loss / max(1, num_train_samples)
                sec_per_batch = elapsed_sec / batches_done
                eta_sec = (total_b - batches_done) * sec_per_batch

                try:
                    with open(live_state_file, 'w') as f:
                        json.dump({
                            'epoch': epoch,
                            'total_epochs': epochs,
                            'phase': 'Training',
                            'batch': batches_done,
                            'total_batches': total_b,
                            'percent': round(pct, 1),
                            'running_loss': round(cur_loss, 4),
                            'best_val_dice': round(best_val_dice, 4),
                            'best_epoch': best_epoch,
                            'elapsed_seconds': round(elapsed_sec, 1),
                            'eta_seconds': round(eta_sec, 1),
                            'timestamp': time.time()
                        }, f)
                except Exception:
                    pass

        train_loss /= num_train_samples

        # ── Validation Phase ──
        model.eval()
        val_loss = 0.0
        val_dices, val_ious, val_precs, val_recs = [], [], [], []
        num_val_samples = 0

        with torch.no_grad():
            for v_idx, (images, masks) in enumerate(val_loader):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    logits = model(images)
                    loss = criterion(logits, masks)

                bs = images.size(0)
                val_loss += loss.item() * bs
                num_val_samples += bs

                m = compute_all_metrics(logits, masks)
                val_dices.append(m['dice'])
                val_ious.append(m['iou'])
                val_precs.append(m['precision'])
                val_recs.append(m['recall'])

                # Update live state during validation
                if (v_idx + 1) % 10 == 0 or (v_idx + 1) == len(val_loader):
                    try:
                        with open(live_state_file, 'w') as f:
                            json.dump({
                                'epoch': epoch,
                                'total_epochs': epochs,
                                'phase': 'Validating',
                                'batch': v_idx + 1,
                                'total_batches': len(val_loader),
                                'percent': round(((v_idx + 1) / len(val_loader)) * 100.0, 1),
                                'running_loss': round(train_loss, 4),
                                'best_val_dice': round(best_val_dice, 4),
                                'best_epoch': best_epoch,
                                'elapsed_seconds': round(time.time() - epoch_start, 1),
                                'eta_seconds': 0,
                                'timestamp': time.time()
                            }, f)
                    except Exception:
                        pass

        val_loss /= num_val_samples
        val_dice = float(sum(val_dices) / len(val_dices))
        val_iou = float(sum(val_ious) / len(val_ious))
        val_prec = float(sum(val_precs) / len(val_precs))
        val_rec = float(sum(val_recs) / len(val_recs))

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start

        # Check for improvement
        improved = val_dice > best_val_dice
        if improved:
            best_val_dice = val_dice
            best_epoch = epoch
            patience_counter = 0

            # Save best checkpoint
            ckpt_path = os.path.join(checkpoint_dir, "patch_refiner_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
                'val_precision': val_prec,
                'val_recall': val_rec,
                'config': {
                    'model': model_config,
                    'patch_size': 512,
                    'epochs': epochs,
                    'lr': lr
                }
            }, ckpt_path)
            status_flag = f"[*] [BEST: {val_dice:.4f}] Checkpoint Saved!"
        else:
            patience_counter += 1
            status_flag = f"[patience: {patience_counter}/{patience}]"

        print(
            f"Epoch {epoch:2d}/{epochs:2d} ({epoch_time:4.1f}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Val Rec: {val_rec:.4f} | "
            f"Val Prec: {val_prec:.4f} | "
            f"{status_flag}",
            flush=True
        )

        # Record history
        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4),
            'val_dice': round(val_dice, 4),
            'val_iou': round(val_iou, 4),
            'val_precision': round(val_prec, 4),
            'val_recall': round(val_rec, 4),
            'lr': current_lr,
            'time_seconds': round(epoch_time, 1)
        })

        # Save latest checkpoint on every epoch for continuous resume capability
        latest_ckpt = os.path.join(checkpoint_dir, "patch_refiner_latest.pth")
        torch.save({
            'epoch': epoch,
            'best_epoch': best_epoch,
            'best_val_dice': best_val_dice,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_dice': val_dice,
            'val_iou': val_iou,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'config': {
                'model': model_config,
                'patch_size': 512,
                'epochs': epochs,
                'lr': lr
            }
        }, latest_ckpt)

        # Save live results JSON
        with open(results_file, 'w') as f:
            json.dump({
                'experiment': 'native_patch_refiner',
                'best_epoch': best_epoch,
                'best_val_dice': round(best_val_dice, 4),
                'total_epochs_completed': epoch,
                'total_time_minutes': round((time.time() - total_start_time) / 60.0, 2),
                'history': history
            }, f, indent=2)

        if patience_counter >= patience:
            print(f"\n[*] Early stopping triggered after {patience} epochs without improvement.", flush=True)
            break

    total_time_min = (time.time() - total_start_time) / 60.0
    print("\n" + "=" * 75, flush=True)
    print(f"[+] TRAINING COMPLETED in {total_time_min:.1f} minutes", flush=True)
    print(f"[+] Best Validation Dice: {best_val_dice:.4f} achieved at Epoch {best_epoch}", flush=True)
    print("=" * 75, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train Native Resolution Solar Filament Patch Refiner")
    parser.add_argument('--resume', action='store_true', help="Resume training from checkpoints/patch_refiner_latest.pth")
    parser.add_argument('--epochs', type=int, default=35, help="Total number of epochs")
    parser.add_argument('--batch-size', type=int, default=4, help="Micro-batch size")
    args = parser.parse_args()

    train_patch_refiner(resume=args.resume, epochs=args.epochs, batch_size=args.batch_size)
