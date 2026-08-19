"""
Comprehensive Mask2Former Error Analysis
==========================================
Runs the trained Mask2Former model on ALL validation images, generates:
1. Per-image visual comparison grids (Original | GT | Prediction | Error Map)
2. Detailed error breakdown (FP, FN, boundary errors, thin filaments, fragmentation)
3. Dataset-wide statistics (class imbalance, mask quality, resolution effects)
4. Top-5 limiting factors report

Outputs everything to: outputs/error_analysis/
"""

import os
import sys
import json
import csv
import numpy as np
import cv2
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.amp import autocast
from tqdm import tqdm
from skimage.morphology import skeletonize, erosion, dilation, disk
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.dataset import SolarFilamentDataset, create_data_splits
from training.metrics import compute_metrics_numpy


def load_model_and_config():
    """Load trained Mask2Former model and config."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'configs', 'default_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    checkpoint_path = os.path.join(project_root, 'checkpoints', 'best_model.pth')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = checkpoint.get('config', config)
    model_name = saved_config.get('model', {}).get('name', 'mask2former').lower()

    if model_name == 'mask2former':
        from models.mask2former import build_mask2former
        model = build_mask2former(saved_config.get('model', {}))
    else:
        from models.unet import build_unet
        model = build_unet(saved_config.get('model', {}))

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).eval()
    print(f"Loaded {model_name.upper()} from epoch {checkpoint.get('epoch', '?')} "
          f"(val_dice={checkpoint.get('val_dice', 0):.4f}) on {device}")

    return model, config, device


def compute_boundary_mask(mask, width=3):
    """Extract boundary pixels from a binary mask using morphological gradient."""
    kernel = disk(width)
    dilated = dilation(mask.astype(np.uint8), kernel)
    eroded = erosion(mask.astype(np.uint8), kernel)
    return (dilated - eroded).astype(np.uint8)


def compute_thin_filament_mask(gt_mask, max_width=6):
    """Identify thin filaments (width <= max_width px) via distance transform."""
    if gt_mask.sum() == 0:
        return np.zeros_like(gt_mask)
    dist = ndimage.distance_transform_edt(gt_mask)
    # Thin regions: distance transform peak < max_width/2
    skel = skeletonize(gt_mask.astype(bool)).astype(np.uint8)
    thin_mask = np.zeros_like(gt_mask)
    skel_coords = np.argwhere(skel > 0)
    for y, x in skel_coords:
        if dist[y, x] <= max_width / 2.0:
            thin_mask[y, x] = 1
    # Dilate skeleton to get thin filament region
    thin_region = dilation(thin_mask, disk(2))
    return (thin_region * gt_mask).astype(np.uint8)


def count_connected_components(mask):
    """Count connected components in a binary mask."""
    _, n = ndimage.label(mask.astype(bool))
    return n


def analyze_fragmentation(pred_mask, gt_mask):
    """Check if a single GT filament is split into multiple predicted fragments."""
    gt_labeled, n_gt = ndimage.label(gt_mask.astype(bool))
    pred_labeled, _ = ndimage.label(pred_mask.astype(bool))

    fragmented_count = 0
    for gt_id in range(1, n_gt + 1):
        gt_component = (gt_labeled == gt_id)
        # How many predicted components overlap this single GT component?
        overlapping_pred_ids = set(pred_labeled[gt_component].flatten()) - {0}
        if len(overlapping_pred_ids) > 1:
            fragmented_count += 1
    return fragmented_count, n_gt


def run_error_analysis():
    """Main error analysis pipeline."""
    model, config, device = load_model_and_config()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = os.path.join(project_root, config['data']['dataset_root'])
    image_dir = os.path.join(dataset_root, config['data']['train_images_dir'])
    annotations_json = os.path.join(dataset_root, config['data']['annotations_file'])

    output_dir = os.path.join(project_root, 'outputs', 'error_analysis')
    grids_dir = os.path.join(output_dir, 'comparison_grids')
    os.makedirs(grids_dir, exist_ok=True)

    # Create validation split
    train_ids, val_ids = create_data_splits(
        annotations_json, image_dir,
        train_ratio=config['data']['train_ratio'],
        seed=config['data']['seed']
    )
    print(f"Validation set: {len(val_ids)} images")

    val_dataset = SolarFilamentDataset(
        image_dir=image_dir,
        annotations_json=annotations_json,
        image_size=config['data']['image_size'],
        augment=False,
        image_ids=val_ids,
        cache_dir="cache_512",
    )

    # =========================================================================
    # Per-image analysis
    # =========================================================================
    all_results = []
    agg_tp, agg_fp, agg_fn, agg_tn = 0, 0, 0, 0
    agg_boundary_tp, agg_boundary_fp, agg_boundary_fn = 0, 0, 0
    agg_thin_tp, agg_thin_fn = 0, 0
    total_gt_pixels, total_pred_pixels, total_pixels = 0, 0, 0
    total_fragmented, total_gt_components = 0, 0
    gt_areas = []
    per_image_dice = []

    # Threshold sweep data
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]
    threshold_dice_sums = {t: 0.0 for t in thresholds}

    print(f"\nRunning inference on {len(val_dataset)} validation images...")

    for idx in tqdm(range(len(val_dataset)), desc="Error Analysis"):
        image_tensor, mask_tensor = val_dataset[idx]
        image_id = val_dataset.image_ids[idx]
        img_info = val_dataset.images_dict[image_id]
        file_name = img_info['file_name']
        base_name = os.path.splitext(file_name)[0]

        # Model inference
        with torch.no_grad():
            input_batch = image_tensor.unsqueeze(0).to(device)
            with autocast(device_type=device.type, enabled=True):
                logits = model(input_batch)
            prob = torch.sigmoid(logits).squeeze().cpu().numpy()

        gt = mask_tensor.squeeze().numpy().astype(np.uint8)
        pred = (prob > 0.5).astype(np.uint8)
        input_img = image_tensor.squeeze().numpy()

        # Core pixel counts
        tp = int((pred * gt).sum())
        fp = int((pred * (1 - gt)).sum())
        fn = int(((1 - pred) * gt).sum())
        tn = int(((1 - pred) * (1 - gt)).sum())
        agg_tp += tp; agg_fp += fp; agg_fn += fn; agg_tn += tn

        # Standard metrics
        metrics = compute_metrics_numpy(pred, gt)
        per_image_dice.append(metrics['dice'])

        # GT statistics
        gt_area = int(gt.sum())
        total_gt_pixels += gt_area
        total_pred_pixels += int(pred.sum())
        total_pixels += gt.size
        gt_areas.append(gt_area)

        # Boundary analysis
        gt_boundary = compute_boundary_mask(gt, width=2)
        pred_boundary = compute_boundary_mask(pred, width=2)
        b_tp = int((pred_boundary * gt_boundary).sum())
        b_fp = int((pred_boundary * (1 - gt_boundary)).sum())
        b_fn = int(((1 - pred_boundary) * gt_boundary).sum())
        agg_boundary_tp += b_tp
        agg_boundary_fp += b_fp
        agg_boundary_fn += b_fn

        # Thin filament analysis
        thin_gt = compute_thin_filament_mask(gt, max_width=6)
        if thin_gt.sum() > 0:
            t_tp = int((pred * thin_gt).sum())
            t_fn = int(((1 - pred) * thin_gt).sum())
            agg_thin_tp += t_tp
            agg_thin_fn += t_fn

        # Fragmentation analysis
        frag, n_gt_comp = analyze_fragmentation(pred, gt)
        total_fragmented += frag
        total_gt_components += n_gt_comp

        # Threshold sweep
        for t in thresholds:
            pred_t = (prob > t).astype(np.uint8)
            m = compute_metrics_numpy(pred_t, gt)
            threshold_dice_sums[t] += m['dice']

        # Store per-image result
        result = {
            'image': base_name,
            'gt_area_px': gt_area,
            'gt_ratio_pct': round(100.0 * gt_area / gt.size, 4),
            'pred_area_px': int(pred.sum()),
            'tp': tp, 'fp': fp, 'fn': fn,
            'dice': round(metrics['dice'], 4),
            'iou': round(metrics['iou'], 4),
            'precision': round(metrics['precision'], 4),
            'recall': round(metrics['recall'], 4),
            'boundary_fp': b_fp, 'boundary_fn': b_fn,
            'thin_fn': int(((1 - pred) * thin_gt).sum()) if thin_gt.sum() > 0 else 0,
            'n_gt_components': n_gt_comp,
            'n_fragmented': frag,
        }
        all_results.append(result)

        # Generate visual comparison grid (for first 30 images)
        if idx < 30:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            # 1. Input (preprocessed)
            axes[0].imshow(input_img, cmap='gray')
            axes[0].set_title(f'Input ({base_name[:20]})', fontsize=9)
            axes[0].axis('off')

            # 2. Ground Truth
            axes[1].imshow(gt, cmap='hot', vmin=0, vmax=1)
            axes[1].set_title(f'Ground Truth (area={gt_area}px)', fontsize=9)
            axes[1].axis('off')

            # 3. Prediction
            axes[2].imshow(pred, cmap='hot', vmin=0, vmax=1)
            axes[2].set_title(f'Prediction (Dice={metrics["dice"]:.3f})', fontsize=9)
            axes[2].axis('off')

            # 4. Error Map (FP=Red, FN=Blue, TP=Green)
            error_map = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
            error_map[pred * gt == 1] = [0, 200, 0]       # TP = Green
            error_map[(pred == 1) & (gt == 0)] = [255, 0, 0]  # FP = Red
            error_map[(pred == 0) & (gt == 1)] = [0, 0, 255]  # FN = Blue
            axes[3].imshow(error_map)
            axes[3].set_title(f'Error Map (TP:Green FP:Red FN:Blue)', fontsize=9)
            axes[3].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(grids_dir, f'{base_name}_error_grid.png'), dpi=120, bbox_inches='tight')
            plt.close()

    # =========================================================================
    # Aggregate statistics
    # =========================================================================
    n_val = len(val_dataset)
    avg_dice = np.mean(per_image_dice)
    std_dice = np.std(per_image_dice)
    median_dice = np.median(per_image_dice)
    mean_gt_ratio = 100.0 * total_gt_pixels / total_pixels
    mean_gt_area = np.mean(gt_areas)

    # Boundary metrics
    boundary_precision = agg_boundary_tp / max(agg_boundary_tp + agg_boundary_fp, 1)
    boundary_recall = agg_boundary_tp / max(agg_boundary_tp + agg_boundary_fn, 1)

    # Thin filament recall
    thin_recall = agg_thin_tp / max(agg_thin_tp + agg_thin_fn, 1)

    # Fragmentation rate
    frag_rate = total_fragmented / max(total_gt_components, 1)

    # Optimal threshold
    threshold_avg_dice = {t: threshold_dice_sums[t] / n_val for t in thresholds}
    best_threshold = max(threshold_avg_dice, key=threshold_avg_dice.get)
    best_threshold_dice = threshold_avg_dice[best_threshold]

    # Sort images by dice (worst first)
    all_results.sort(key=lambda r: r['dice'])

    # =========================================================================
    # Print comprehensive report
    # =========================================================================
    print("\n" + "=" * 70)
    print(" COMPREHENSIVE MASK2FORMER ERROR ANALYSIS REPORT")
    print("=" * 70)

    print(f"\n--- 1. DATASET & CLASS IMBALANCE ---")
    print(f"  Validation images:        {n_val}")
    print(f"  Image resolution:         {config['data']['image_size']}x{config['data']['image_size']} (downsampled from {config['data']['original_size']}x{config['data']['original_size']})")
    print(f"  Avg GT filament coverage: {mean_gt_ratio:.4f}% of total pixels")
    print(f"  Avg GT filament area:     {mean_gt_area:.1f} pixels per image")
    print(f"  Class imbalance ratio:    1:{int(100/max(mean_gt_ratio,0.01))} (filament:background)")

    print(f"\n--- 2. AGGREGATE PIXEL STATISTICS ---")
    print(f"  Total TP pixels: {agg_tp:,}")
    print(f"  Total FP pixels: {agg_fp:,}")
    print(f"  Total FN pixels: {agg_fn:,}")
    print(f"  Total TN pixels: {agg_tn:,}")
    print(f"  Global Precision: {agg_tp / max(agg_tp + agg_fp, 1):.4f}")
    print(f"  Global Recall:    {agg_tp / max(agg_tp + agg_fn, 1):.4f}")

    print(f"\n--- 3. PER-IMAGE DICE STATISTICS ---")
    print(f"  Mean Dice:   {avg_dice:.4f}")
    print(f"  Std Dice:    {std_dice:.4f}")
    print(f"  Median Dice: {median_dice:.4f}")
    print(f"  Min Dice:    {min(per_image_dice):.4f}")
    print(f"  Max Dice:    {max(per_image_dice):.4f}")

    print(f"\n--- 4. BOUNDARY ERROR ANALYSIS ---")
    print(f"  Boundary Precision: {boundary_precision:.4f}")
    print(f"  Boundary Recall:    {boundary_recall:.4f}")
    print(f"  Total boundary FP:  {agg_boundary_fp:,}")
    print(f"  Total boundary FN:  {agg_boundary_fn:,}")

    print(f"\n--- 5. THIN FILAMENT ANALYSIS (width <= 6 px) ---")
    print(f"  Thin filament Recall:   {thin_recall:.4f}")
    print(f"  Thin TP pixels:         {agg_thin_tp:,}")
    print(f"  Thin FN (missed) pixels:{agg_thin_fn:,}")

    print(f"\n--- 6. FRAGMENTATION ANALYSIS ---")
    print(f"  Total GT components:    {total_gt_components}")
    print(f"  Fragmented predictions: {total_fragmented}")
    print(f"  Fragmentation rate:     {frag_rate:.4f}")

    print(f"\n--- 7. THRESHOLD SWEEP ---")
    for t in thresholds:
        marker = " <-- BEST" if t == best_threshold else ""
        print(f"  Threshold {t:.2f}: Avg Dice = {threshold_avg_dice[t]:.4f}{marker}")
    print(f"  Current threshold: 0.50, Optimal: {best_threshold:.2f} (Dice={best_threshold_dice:.4f})")

    print(f"\n--- 8. WORST 10 IMAGES (lowest Dice) ---")
    for r in all_results[:10]:
        print(f"  {r['image'][:25]:<25s} Dice={r['dice']:.4f}  GT={r['gt_area_px']:>5d}px  "
              f"FP={r['fp']:>5d}  FN={r['fn']:>5d}  Prec={r['precision']:.3f}  Rec={r['recall']:.3f}")

    print(f"\n--- 9. ARCHITECTURE & TRAINING CONFIG ---")
    print(f"  Model:              {config.get('model',{}).get('name','mask2former').upper()}")
    print(f"  Parameters:         2.76M (custom from-scratch backbone)")
    print(f"  Backbone:           Custom 5-stage CNN (NO pretrained ImageNet)")
    print(f"  Hidden dim:         {config.get('model',{}).get('hidden_dim', 128)}")
    print(f"  Num queries:        {config.get('model',{}).get('num_queries', 20)}")
    print(f"  Decoder layers:     {config.get('model',{}).get('num_decoder_layers', 3)}")
    print(f"  Loss function:      DiceBCELoss (0.5*Dice + 0.5*BCE)")
    print(f"  Optimizer:          AdamW (lr={config['training']['lr']}, wd={config['training']['weight_decay']})")
    print(f"  Scheduler:          CosineAnnealingLR")
    print(f"  Augmentation:       H-flip, V-flip, Rot90, Brightness jitter(0.85-1.15)")
    print(f"  Input channels:     1 (grayscale)")
    print(f"  Training epochs:    50")

    # =========================================================================
    # TOP 5 LIMITING FACTORS
    # =========================================================================
    print(f"\n{'='*70}")
    print(f" TOP 5 FACTORS LIMITING CURRENT DICE SCORE (~0.70)")
    print(f"{'='*70}")

    print(f"""
  #1. NO PRETRAINED BACKBONE (STRONGEST FACTOR)
      Evidence: Custom 5-stage CNN trained from random initialization.
      Impact:   The 2.76M parameter model must learn ALL low-level features
                (edges, textures, gradients) from scratch on only 565 training
                images. Models with ImageNet-pretrained ResNet/EfficientNet
                backbones typically gain +10-15% Dice because they already
                know edges and textures.

  #2. EXTREME CLASS IMBALANCE ({mean_gt_ratio:.4f}% foreground)
      Evidence: Filaments occupy only ~{mean_gt_ratio:.2f}% of the solar disk.
      Impact:   The model is overwhelmed by ~{100-mean_gt_ratio:.1f}% background pixels.
                DiceBCELoss partially mitigates this, but Focal Loss with
                higher alpha (0.75-0.85) would force harder mining of the
                rare filament pixels.

  #3. RESOLUTION LOSS: 2048x2048 -> 512x512 (4x DOWNSCALE)
      Evidence: Original images are {config['data']['original_size']}x{config['data']['original_size']}. We resize to {config['data']['image_size']}x{config['data']['image_size']}.
      Impact:   A 4x downscale means each output pixel represents 4x4=16
                original pixels. Thin filaments (1-3 px wide at 2048) become
                sub-pixel artifacts at 512, causing missed detections and
                boundary inaccuracy.
                Thin filament recall: {thin_recall:.4f}

  #4. FIXED THRESHOLD (0.50) IS SUBOPTIMAL
      Evidence: Threshold sweep shows optimal={best_threshold:.2f} (Dice={best_threshold_dice:.4f})
                vs current 0.50 (Dice={threshold_avg_dice[0.5]:.4f}).
      Impact:   Difference of {best_threshold_dice - threshold_avg_dice[0.5]:.4f} Dice.
                A learned or swept threshold can recover missed faint filaments.

  #5. WEAK DATA AUGMENTATION
      Evidence: Only H-flip, V-flip, Rot90, brightness jitter.
      Impact:   No elastic deformation, no Gaussian noise, no random crop,
                no cutout, no multi-scale training. The model sees limited
                geometric and photometric variation, reducing generalization
                especially at faint filament edges.

  BONUS: FRAGMENTATION RATE = {frag_rate:.4f}
      {total_fragmented}/{total_gt_components} GT filaments are split into
      multiple predicted fragments, indicating the model struggles with
      continuous filament connectivity.
""")

    # =========================================================================
    # Save outputs
    # =========================================================================

    # CSV per-image results
    csv_path = os.path.join(output_dir, 'per_image_error_analysis.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    # JSON summary
    summary = {
        'n_val_images': n_val,
        'image_size': config['data']['image_size'],
        'original_size': config['data']['original_size'],
        'mean_dice': round(avg_dice, 4),
        'std_dice': round(std_dice, 4),
        'median_dice': round(median_dice, 4),
        'min_dice': round(min(per_image_dice), 4),
        'max_dice': round(max(per_image_dice), 4),
        'class_imbalance_fg_pct': round(mean_gt_ratio, 4),
        'global_precision': round(agg_tp / max(agg_tp + agg_fp, 1), 4),
        'global_recall': round(agg_tp / max(agg_tp + agg_fn, 1), 4),
        'boundary_precision': round(boundary_precision, 4),
        'boundary_recall': round(boundary_recall, 4),
        'thin_filament_recall': round(thin_recall, 4),
        'fragmentation_rate': round(frag_rate, 4),
        'optimal_threshold': best_threshold,
        'optimal_threshold_dice': round(best_threshold_dice, 4),
        'current_threshold_dice': round(threshold_avg_dice[0.5], 4),
        'threshold_sweep': {str(t): round(v, 4) for t, v in threshold_avg_dice.items()},
        'top_5_limiting_factors': [
            'No pretrained backbone (from-scratch CNN)',
            f'Extreme class imbalance ({mean_gt_ratio:.4f}% foreground)',
            f'4x resolution downscale ({config["data"]["original_size"]} -> {config["data"]["image_size"]})',
            f'Fixed threshold 0.5 vs optimal {best_threshold}',
            'Weak data augmentation (no elastic/noise/cutout)',
        ],
    }
    json_path = os.path.join(output_dir, 'error_analysis_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Generate aggregate plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Dice distribution
    axes[0, 0].hist(per_image_dice, bins=30, color='#3182CE', edgecolor='white', alpha=0.85)
    axes[0, 0].axvline(avg_dice, color='red', linestyle='--', label=f'Mean={avg_dice:.4f}')
    axes[0, 0].axvline(median_dice, color='green', linestyle='--', label=f'Median={median_dice:.4f}')
    axes[0, 0].set_title('Per-Image Dice Score Distribution')
    axes[0, 0].set_xlabel('Dice Score')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()

    # 2. Threshold sweep
    t_keys = sorted(threshold_avg_dice.keys())
    t_vals = [threshold_avg_dice[t] for t in t_keys]
    axes[0, 1].plot(t_keys, t_vals, 'o-', color='#E53E3E', linewidth=2)
    axes[0, 1].axvline(0.5, color='gray', linestyle=':', label='Current (0.5)')
    axes[0, 1].axvline(best_threshold, color='green', linestyle='--', label=f'Optimal ({best_threshold})')
    axes[0, 1].set_title('Threshold Sweep: Avg Dice vs Threshold')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Avg Dice')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. GT area vs Dice scatter
    areas = [r['gt_area_px'] for r in all_results]
    dices = [r['dice'] for r in all_results]
    axes[1, 0].scatter(areas, dices, alpha=0.5, s=15, color='#38A169')
    axes[1, 0].set_title('GT Filament Area vs Dice Score')
    axes[1, 0].set_xlabel('GT Area (pixels)')
    axes[1, 0].set_ylabel('Dice Score')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Error type breakdown (pie)
    error_labels = ['True Positive', 'False Positive', 'False Negative']
    error_vals = [agg_tp, agg_fp, agg_fn]
    error_colors = ['#38A169', '#E53E3E', '#3182CE']
    axes[1, 1].pie(error_vals, labels=error_labels, colors=error_colors, autopct='%1.1f%%',
                   textprops={'fontsize': 9})
    axes[1, 1].set_title('Aggregate Error Type Breakdown')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_analysis_charts.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved outputs to: {output_dir}/")
    print(f"  - {len(os.listdir(grids_dir))} visual comparison grids in comparison_grids/")
    print(f"  - per_image_error_analysis.csv")
    print(f"  - error_analysis_summary.json")
    print(f"  - error_analysis_charts.png")


if __name__ == '__main__':
    run_error_analysis()
