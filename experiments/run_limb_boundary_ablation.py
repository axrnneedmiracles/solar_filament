"""
Solar Disk Boundary Erosion & Limb Filament Diagnostic Experiment
===================================================================
Investigates the impact of solar disk radius erosion (0.93 vs 0.97 vs 1.00)
on filament segmentation accuracy, particularly near the solar limb.

Computes:
1. Whole-Disk & Limb-Region (outer 10% radius) Dice, IoU, Precision, Recall.
2. Ground-Truth filament pixel count inside the 7% eroded annulus.
3. 6-panel visual comparison grids for 20+ limb-filament validation samples.
4. Dedicated diagnostic analysis of target image 20140519195834Ch.
5. Statistical distribution and causal analysis (limb darkening, projection, etc.).

Saves all artifacts to: outputs/limb_boundary_ablation/
"""

import os
import sys
import json
import csv
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mask2former import build_mask2former
from preprocessing.solar_preprocessor import SolarPreprocessor
from preprocessing.dataset import load_coco_annotations, create_data_splits, coco_poly_to_mask


def detect_true_solar_disk(gray: np.ndarray) -> Tuple[int, int, int]:
    """Detects true un-eroded solar disk center and radius."""
    _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = gray.shape
        return w // 2, h // 2, min(w, h) // 2 - 10
    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    return int(cx), int(cy), int(radius)


def preprocess_with_custom_radius_scale(raw_gray: np.ndarray, radius_scale: float = 0.93, target_size: int = 512) -> Tuple[np.ndarray, Tuple[int, int, int], np.ndarray]:
    """
    Preprocesses solar image with a custom solar disk radius scaling factor.
    radius_scale = 0.93: Current 7% boundary reduction (0.7% radius reduction)
    radius_scale = 0.97: 3% boundary reduction
    radius_scale = 1.00: Full detected solar disk without erosion
    """
    orig_cx, orig_cy, orig_radius = detect_true_solar_disk(raw_gray)
    safe_radius = int(orig_radius * radius_scale)
    h, w = raw_gray.shape

    # Disk mask
    disk_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(disk_mask, (orig_cx, orig_cy), safe_radius, 255, -1)

    # Limb darkening correction
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - orig_cx) ** 2 + (y - orig_cy) ** 2) / max(safe_radius, 1)
    r = np.clip(r, 0, 1)
    mu = np.sqrt(np.maximum(1 - r ** 2, 0))
    u = 0.6
    correction = 1.0 / (1.0 - u * (1.0 - mu) + 1e-8)

    corrected = raw_gray.astype(np.float64)
    corrected[disk_mask > 0] = corrected[disk_mask > 0] * correction[disk_mask > 0]
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    # Normalize within disk
    masked = corrected[disk_mask > 0]
    if len(masked) > 0:
        vmin, vmax = np.percentile(masked, [1, 99])
        if vmax - vmin >= 1:
            corrected = (corrected - vmin) / (vmax - vmin) * 255
            corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    corrected[disk_mask == 0] = 0

    # Denoise
    denoised = cv2.GaussianBlur(corrected, (5, 5), 1.0)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    enhanced[disk_mask == 0] = 0

    # Resize to target
    resized_input = cv2.resize(enhanced, (target_size, target_size), interpolation=cv2.INTER_AREA)
    norm_tensor = resized_input.astype(np.float32) / 255.0

    return norm_tensor, (orig_cx, orig_cy, orig_radius), disk_mask


def compute_binary_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray, region_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Computes Dice, IoU, Precision, Recall on a binary mask, optionally within a region."""
    if region_mask is not None:
        p = pred_bin[region_mask > 0]
        g = gt_bin[region_mask > 0]
    else:
        p = pred_bin.flatten()
        g = gt_bin.flatten()

    tp = np.sum((p == 1) & (g == 1))
    fp = np.sum((p == 1) & (g == 0))
    fn = np.sum((p == 0) & (g == 1))
    tn = np.sum((p == 0) & (g == 0))

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8) if (2 * tp + fp + fn) > 0 else 1.0 if (np.sum(g) == 0 and np.sum(p) == 0) else 0.0
    iou = tp / (tp + fp + fn + 1e-8) if (tp + fp + fn) > 0 else 1.0 if (np.sum(g) == 0 and np.sum(p) == 0) else 0.0
    precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0

    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'gt_pixels': int(np.sum(g)),
        'pred_pixels': int(np.sum(p))
    }


def run_ablation_study():
    output_dir = "outputs/limb_boundary_ablation"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)

    print("=" * 80)
    print("SOLAR DISK BOUNDARY EROSION & LIMB FILAMENT ABLATION EXPERIMENT")
    print("=" * 80)

    # 1. Load Model (Use CPU to ensure zero VRAM collision with active background training)
    device = torch.device('cpu')
    ckpt_path = "checkpoints/phase2_hybrid_loss_dice0.7249.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/phase3_768res_dice0.7207.pth"
    print(f"[*] Loading Model from: {ckpt_path} on {device} (Zero VRAM impact)")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg = checkpoint.get('config', {}).get('model', {})
    target_size = checkpoint.get('config', {}).get('data', {}).get('image_size', 512)

    model = build_mask2former(saved_cfg).to(device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Load Dataset Split (reproducible seed 42)
    img_dir = "images/MAGFiLO_1.0_Kaggle_2026/train/train_images"
    ann_file = "images/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    images_dict, annotations_by_image, _ = load_coco_annotations(ann_file)
    train_ids, val_ids = create_data_splits(ann_file, img_dir, train_ratio=0.8, seed=42)
    print(f"[*] Total Validation Images: {len(val_ids)}")

    # 3. Analyze all validation images for limb filament presence
    print("[*] Scanning validation set for limb filaments (r > 0.85 R_disk)...")
    
    val_records = []
    total_gt_in_eroded_annulus_all = 0
    total_gt_all = 0

    radius_scales = [0.93, 0.97, 1.00]
    results_by_scale = {0.93: [], 0.97: [], 1.00: []}

    for idx, img_id in enumerate(val_ids):
        info = images_dict[img_id]
        fn = info['file_name']
        fp = os.path.join(img_dir, fn)
        raw = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue

        h, w = raw.shape
        # Create full ground-truth mask at original resolution
        gt_raw = np.zeros((h, w), dtype=np.uint8)
        for ann in annotations_by_image.get(img_id, []):
            seg = ann.get('segmentation', [])
            if seg:
                gt_raw = np.maximum(gt_raw, coco_poly_to_mask(seg, h, w))

        gt_512 = cv2.resize(gt_raw, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        # Detect true solar disk
        cx, cy, r_full = detect_true_solar_disk(raw)
        
        # Build geometric masks at target_size
        scale_x = target_size / w
        scale_y = target_size / h
        cx_512 = int(cx * scale_x)
        cy_512 = int(cy * scale_y)
        r_512 = int(r_full * scale_x)

        # Create radial distance map
        y_grid, x_grid = np.ogrid[:target_size, :target_size]
        dist_map = np.sqrt((x_grid - cx_512)**2 + (y_grid - cy_512)**2) / max(r_512, 1)

        # Full disk: dist <= 1.00
        full_disk_mask = (dist_map <= 1.00).astype(np.uint8)
        # Limb annulus: 0.90 <= dist <= 1.00
        limb_annulus_mask = ((dist_map >= 0.90) & (dist_map <= 1.00)).astype(np.uint8)
        # 7% Eroded boundary annulus: 0.93 < dist <= 1.00
        eroded_annulus_mask = ((dist_map > 0.93) & (dist_map <= 1.00)).astype(np.uint8)

        gt_total_px = int(np.sum(gt_512 * full_disk_mask))
        gt_limb_px = int(np.sum(gt_512 * limb_annulus_mask))
        gt_eroded_px = int(np.sum(gt_512 * eroded_annulus_mask))

        total_gt_all += gt_total_px
        total_gt_in_eroded_annulus_all += gt_eroded_px

        record = {
            'image_id': img_id,
            'file_name': fn,
            'gt_total_px': gt_total_px,
            'gt_limb_px': gt_limb_px,
            'gt_eroded_px': gt_eroded_px,
            'has_limb_filaments': gt_limb_px > 10,
            'disk_params': (cx_512, cy_512, r_512),
            'predictions': {}
        }

        # Run inference for all 3 radius scale variants
        for scale in radius_scales:
            norm_in, _, _ = preprocess_with_custom_radius_scale(raw, radius_scale=scale, target_size=target_size)
            tensor_in = torch.from_numpy(norm_in).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor_in)
                probs = torch.sigmoid(logits).cpu().squeeze().numpy()

            pred_bin = (probs > 0.5).astype(np.uint8)

            # Whole-disk metrics
            whole_metrics = compute_binary_metrics(pred_bin, gt_512, region_mask=full_disk_mask)
            # Limb-region metrics
            limb_metrics = compute_binary_metrics(pred_bin, gt_512, region_mask=limb_annulus_mask)

            results_by_scale[scale].append({
                'image_id': img_id,
                'file_name': fn,
                'whole_dice': whole_metrics['dice'],
                'whole_iou': whole_metrics['iou'],
                'whole_prec': whole_metrics['precision'],
                'whole_rec': whole_metrics['recall'],
                'limb_dice': limb_metrics['dice'],
                'limb_iou': limb_metrics['iou'],
                'limb_prec': limb_metrics['precision'],
                'limb_rec': limb_metrics['recall'],
                'gt_limb_px': gt_limb_px,
                'gt_eroded_px': gt_eroded_px,
                'pred_bin': pred_bin,
                'probs': probs,
                'norm_in': norm_in,
            })

            record['predictions'][scale] = {
                'pred_bin': pred_bin,
                'whole_dice': whole_metrics['dice'],
                'whole_rec': whole_metrics['recall'],
                'limb_dice': limb_metrics['dice'],
                'limb_rec': limb_metrics['recall']
            }

        val_records.append(record)

    # 4. Filter validation images containing significant limb filaments
    limb_records = [r for r in val_records if r['has_limb_filaments']]
    limb_records.sort(key=lambda r: r['gt_limb_px'], reverse=True)
    print(f"[+] Total Validation Images with Limb Filaments: {len(limb_records)} (Selected Top {min(len(limb_records), 25)} for Visual Grid)")

    # 5. Generate Side-by-Side 6-Panel Comparisons for at least 20 limb validation images
    print("[*] Generating side-by-side diagnostic figures...")
    top_limb_samples = limb_records[:25]

    for i, r in enumerate(top_limb_samples):
        fn = r['file_name']
        base_id = os.path.splitext(fn)[0]
        raw_img = cv2.imread(os.path.join(img_dir, fn), cv2.IMREAD_GRAYSCALE)
        h, w = raw_img.shape
        raw_512 = cv2.resize(raw_img, (target_size, target_size))

        gt_raw = np.zeros((h, w), dtype=np.uint8)
        for ann in annotations_by_image.get(r['image_id'], []):
            seg = ann.get('segmentation', [])
            if seg:
                gt_raw = np.maximum(gt_raw, coco_poly_to_mask(seg, h, w))
        gt_512 = cv2.resize(gt_raw, (target_size, target_size), interpolation=cv2.INTER_NEAREST)

        pred_093 = r['predictions'][0.93]['pred_bin']
        pred_100 = r['predictions'][1.00]['pred_bin']

        cx, cy, radius = r['disk_params']

        fig, axes = plt.subplots(2, 3, figsize=(16, 11), facecolor='#0B0F19')
        fig.suptitle(f"Limb Boundary Ablation: {base_id} | GT Limb Px: {r['gt_limb_px']} px (Er: {r['gt_eroded_px']} px)",
                     fontsize=14, color='white', fontweight='bold', y=0.98)

        # 1. Raw Solar Image with Limb Annulus
        vis_raw = cv2.cvtColor(raw_512, cv2.COLOR_GRAY2RGB)
        cv2.circle(vis_raw, (cx, cy), radius, (0, 255, 255), 2)  # Full limb (Yellow)
        cv2.circle(vis_raw, (cx, cy), int(radius * 0.93), (255, 0, 0), 1)  # 0.93 Boundary (Red)
        cv2.circle(vis_raw, (cx, cy), int(radius * 0.90), (0, 255, 0), 1)  # 0.90 Limb Start (Green)
        axes[0, 0].imshow(vis_raw)
        axes[0, 0].set_title("1. Raw Solar Disk (Yellow=1.00r, Red=0.93r)", color='white', fontsize=10)
        axes[0, 0].axis('off')

        # 2. Ground Truth Mask
        vis_gt = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        vis_gt[gt_512 > 0] = [255, 255, 255]
        cv2.circle(vis_gt, (cx, cy), radius, (0, 255, 255), 1)
        cv2.circle(vis_gt, (cx, cy), int(radius * 0.93), (255, 0, 0), 1)
        axes[0, 1].imshow(vis_gt)
        axes[0, 1].set_title(f"2. Ground Truth Mask (Limb GT: {r['gt_limb_px']} px)", color='white', fontsize=10)
        axes[0, 1].axis('off')

        # 3. Prediction with 0.7% Boundary Reduction (0.93 radius)
        vis_093 = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        vis_093[pred_093 > 0] = [0, 255, 255]  # Cyan
        cv2.circle(vis_093, (cx, cy), int(radius * 0.93), (255, 0, 0), 1)
        axes[0, 2].imshow(vis_093)
        axes[0, 2].set_title(f"3. Pred: 7% Erosion (0.93r)\nLimb Rec: {r['predictions'][0.93]['limb_rec']:.1%}, Dice: {r['predictions'][0.93]['limb_dice']:.3f}", color='white', fontsize=10)
        axes[0, 2].axis('off')

        # 4. Prediction with 0% Erosion (1.00 radius full disk)
        vis_100 = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        vis_100[pred_100 > 0] = [0, 255, 0]  # Green
        cv2.circle(vis_100, (cx, cy), radius, (0, 255, 255), 1)
        axes[1, 0].imshow(vis_100)
        axes[1, 0].set_title(f"4. Pred: 0% Erosion (1.00r Full Disk)\nLimb Rec: {r['predictions'][1.00]['limb_rec']:.1%}, Dice: {r['predictions'][1.00]['limb_dice']:.3f}", color='white', fontsize=10)
        axes[1, 0].axis('off')

        # 5. Error Map (0% Erosion vs Ground Truth)
        # TP=Green, FP=Red, FN=Blue
        error_map = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        error_map[(pred_100 == 1) & (gt_512 == 1)] = [0, 255, 0]     # TP = Green
        error_map[(pred_100 == 1) & (gt_512 == 0)] = [255, 50, 50]   # FP = Red
        error_map[(pred_100 == 0) & (gt_512 == 1)] = [50, 100, 255]  # FN = Blue
        cv2.circle(error_map, (cx, cy), radius, (200, 200, 200), 1)
        axes[1, 1].imshow(error_map)
        axes[1, 1].set_title("5. Error Map (Green=TP, Red=FP, Blue=FN)", color='white', fontsize=10)
        axes[1, 1].axis('off')

        # 6. Zoomed Inset of the Prominent Limb Section
        # Find centroid of GT limb pixels if present, otherwise limb center
        y_gt_limb, x_gt_limb = np.where((gt_512 > 0) & (np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2) >= 0.85 * radius))
        if len(x_gt_limb) > 0:
            zc_x, zc_y = int(np.mean(x_gt_limb)), int(np.mean(y_gt_limb))
        else:
            zc_x, zc_y = cx, cy - radius

        hw = 64
        zx1, zy1 = max(0, zc_x - hw), max(0, zc_y - hw)
        zx2, zy2 = min(target_size, zc_x + hw), min(target_size, zc_y + hw)

        zoom_vis = np.zeros((zy2 - zy1, zx2 - zx1, 3), dtype=np.uint8)
        sub_raw = raw_512[zy1:zy2, zx1:zx2]
        sub_gt = gt_512[zy1:zy2, zx1:zx2]
        sub_pred100 = pred_100[zy1:zy2, zx1:zx2]

        zoom_vis = cv2.cvtColor(sub_raw, cv2.COLOR_GRAY2RGB)
        zoom_vis[sub_gt > 0] = [0, 255, 0]  # GT overlay in Green
        zoom_vis[sub_pred100 > 0] = [255, 0, 255]  # Pred overlay in Magenta
        zoom_vis[(sub_gt > 0) & (sub_pred100 > 0)] = [0, 255, 255]  # Overlap Cyan

        axes[1, 2].imshow(zoom_vis)
        axes[1, 2].set_title("6. Limb Zoom Inset (Cyan=Match, Green=GT, Mag=Pred)", color='white', fontsize=10)
        axes[1, 2].axis('off')

        plt.tight_layout()
        save_fig_path = os.path.join(output_dir, "comparisons", f"sample_{i+1:02d}_{base_id}_ablation.png")
        plt.savefig(save_fig_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()

    # 6. Special Diagnostic on Requested Target Image: 20140519195834Ch
    print("\n" + "=" * 80)
    print("DIAGNOSTIC TARGET IMAGE ANALYSIS: 20140519195834Ch")
    print("=" * 80)
    target_img_path = "images/MAGFiLO_1.0_Kaggle_2026/test/test_images/20140519195834Ch.jpeg"
    target_results = {}

    if os.path.exists(target_img_path):
        raw_target = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
        t_h, t_w = raw_target.shape
        t_cx, t_cy, t_rad = detect_true_solar_disk(raw_target)
        print(f"[+] Target Image Loaded: Shape ({t_h}, {t_w}), Solar Disk: Center ({t_cx}, {t_cy}), True Radius: {t_rad} px")

        fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor='#0B0F19')
        fig.suptitle("Target Image 20140519195834Ch: Boundary Erosion Ablation", fontsize=15, color='white', fontweight='bold')

        # Plot 1: Raw with boundary circles
        vis_t_raw = cv2.cvtColor(cv2.resize(raw_target, (target_size, target_size)), cv2.COLOR_GRAY2RGB)
        cv2.circle(vis_t_raw, (target_size//2, target_size//2), int(t_rad * (target_size/t_w)), (0, 255, 255), 2)
        cv2.circle(vis_t_raw, (target_size//2, target_size//2), int(t_rad * 0.93 * (target_size/t_w)), (255, 0, 0), 2)
        axes[0].imshow(vis_t_raw)
        axes[0].set_title("1. Raw Image (Yellow=1.00r, Red=0.93r)", color='white', fontsize=11)
        axes[0].axis('off')

        for s_idx, scale in enumerate([0.93, 0.97, 1.00]):
            norm_t, _, _ = preprocess_with_custom_radius_scale(raw_target, radius_scale=scale, target_size=target_size)
            tensor_t = torch.from_numpy(norm_t).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                t_logits = model(tensor_t)
                t_prob = torch.sigmoid(t_logits).cpu().squeeze().numpy()

            t_pred = (t_prob > 0.5).astype(np.uint8)
            num_fils, _, t_stats, _ = cv2.connectedComponentsWithStats(t_pred, connectivity=8)
            fil_px = np.sum(t_pred)

            target_results[scale] = {
                'detected_filaments': num_fils - 1,
                'total_filament_pixels': int(fil_px),
                'pred_mask': t_pred,
                'prob_map': t_prob
            }

            vis_t_pred = cv2.cvtColor(cv2.resize(raw_target, (target_size, target_size)), cv2.COLOR_GRAY2RGB)
            # Overlay prediction in Cyan/Lime
            vis_t_pred[t_pred > 0] = [0, 255, 255]
            cv2.circle(vis_t_pred, (target_size//2, target_size//2), int(t_rad * scale * (target_size/t_w)), (255, 255, 255), 1)

            label_name = f"Erosion: {100*(1-scale):.1f}% (Radius: {scale:.2f}r)"
            axes[s_idx + 1].imshow(vis_t_pred)
            axes[s_idx + 1].set_title(f"{label_name}\nDetected: {num_fils-1} fils, {fil_px} px", color='white', fontsize=11)
            axes[s_idx + 1].axis('off')

            print(f"  -> Scale {scale:.2f}r ({100*(1-scale):.1f}% erosion): {num_fils - 1} filament(s) detected, {fil_px} total pixels")

        plt.tight_layout()
        target_fig_path = os.path.join(output_dir, "target_image_20140519195834Ch_ablation.png")
        plt.savefig(target_fig_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        print(f"[+] Saved Target Image Visualization: {target_fig_path}")

    # 7. Compute Global Validation Statistics Across All Variants
    print("\n" + "=" * 80)
    print("GLOBAL NUMERICAL RESULTS (VALIDATION BENCHMARK)")
    print("=" * 80)

    summary_stats = {}
    for scale in radius_scales:
        scale_records = results_by_scale[scale]
        whole_dices = [r['whole_dice'] for r in scale_records]
        whole_ious = [r['whole_iou'] for r in scale_records]
        whole_precs = [r['whole_prec'] for r in scale_records]
        whole_recs = [r['whole_rec'] for r in scale_records]

        # Filter for images with ground-truth limb filaments
        limb_dices = [r['limb_dice'] for r in scale_records if r['gt_limb_px'] > 0]
        limb_ious = [r['limb_iou'] for r in scale_records if r['gt_limb_px'] > 0]
        limb_precs = [r['limb_prec'] for r in scale_records if r['gt_limb_px'] > 0]
        limb_recs = [r['limb_rec'] for r in scale_records if r['gt_limb_px'] > 0]

        summary_stats[scale] = {
            'radius_scale': scale,
            'erosion_pct': round((1.0 - scale) * 100, 1),
            'whole_disk': {
                'dice': float(np.mean(whole_dices)),
                'iou': float(np.mean(whole_ious)),
                'precision': float(np.mean(whole_precs)),
                'recall': float(np.mean(whole_recs))
            },
            'limb_region': {
                'dice': float(np.mean(limb_dices)) if limb_dices else 0.0,
                'iou': float(np.mean(limb_ious)) if limb_ious else 0.0,
                'precision': float(np.mean(limb_precs)) if limb_precs else 0.0,
                'recall': float(np.mean(limb_recs)) if limb_recs else 0.0
            }
        }

        print(f"\n--- Variant {100*(1-scale):.1f}% Erosion (Radius Scale: {scale:.2f}r) ---")
        print(f"  Whole-Disk:   Dice={summary_stats[scale]['whole_disk']['dice']:.4f} | IoU={summary_stats[scale]['whole_disk']['iou']:.4f} | Prec={summary_stats[scale]['whole_disk']['precision']:.4f} | Rec={summary_stats[scale]['whole_disk']['recall']:.4f}")
        print(f"  Limb-Region:  Dice={summary_stats[scale]['limb_region']['dice']:.4f} | IoU={summary_stats[scale]['limb_region']['iou']:.4f} | Prec={summary_stats[scale]['limb_region']['precision']:.4f} | Rec={summary_stats[scale]['limb_region']['recall']:.4f}")

    # Ground truth pixel count in the 7% boundary-removed region
    gt_eroded_annulus_pct = (total_gt_in_eroded_annulus_all / max(total_gt_all, 1)) * 100
    print(f"\n[*] Total Ground-Truth Filament Pixels in Dataset: {total_gt_all:,} px")
    print(f"[*] Ground-Truth Pixels in 7% Boundary-Removed Region [0.93r, 1.00r]: {total_gt_in_eroded_annulus_all:,} px ({gt_eroded_annulus_pct:.2f}%)")

    # 8. Save Metrics CSV & Summary JSON
    csv_path = os.path.join(output_dir, "per_image_limb_ablation_metrics.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Image_ID", "File_Name", "GT_Total_Px", "GT_Limb_Px", "GT_Eroded_Annulus_Px",
            "Dice_093_Whole", "Rec_093_Whole", "Dice_093_Limb", "Rec_093_Limb",
            "Dice_097_Whole", "Rec_097_Whole", "Dice_097_Limb", "Rec_097_Limb",
            "Dice_100_Whole", "Rec_100_Whole", "Dice_100_Limb", "Rec_100_Limb"
        ])
        for idx in range(len(val_ids)):
            r93 = results_by_scale[0.93][idx]
            r97 = results_by_scale[0.97][idx]
            r100 = results_by_scale[1.00][idx]
            writer.writerow([
                r93['image_id'], r93['file_name'], val_records[idx]['gt_total_px'], r93['gt_limb_px'], r93['gt_eroded_px'],
                f"{r93['whole_dice']:.4f}", f"{r93['whole_rec']:.4f}", f"{r93['limb_dice']:.4f}", f"{r93['limb_rec']:.4f}",
                f"{r97['whole_dice']:.4f}", f"{r97['whole_rec']:.4f}", f"{r97['limb_dice']:.4f}", f"{r97['limb_rec']:.4f}",
                f"{r100['whole_dice']:.4f}", f"{r100['whole_rec']:.4f}", f"{r100['limb_dice']:.4f}", f"{r100['limb_rec']:.4f}",
            ])
    print(f"[+] Saved Per-Image Metrics CSV: {csv_path}")

    # Summary JSON
    json_path = os.path.join(output_dir, "limb_ablation_summary.json")
    summary_data = {
        'total_validation_images': len(val_ids),
        'images_with_limb_filaments': len(limb_records),
        'total_gt_filament_pixels': total_gt_all,
        'gt_pixels_in_093_eroded_region': total_gt_in_eroded_annulus_all,
        'gt_pixels_in_093_eroded_region_pct': float(gt_eroded_annulus_pct),
        'variants': summary_stats,
        'target_image_20140519195834Ch': {
            'detected_093_filaments': target_results.get(0.93, {}).get('detected_filaments', 0),
            'detected_093_pixels': target_results.get(0.93, {}).get('total_filament_pixels', 0),
            'detected_097_filaments': target_results.get(0.97, {}).get('detected_filaments', 0),
            'detected_097_pixels': target_results.get(0.97, {}).get('total_filament_pixels', 0),
            'detected_100_filaments': target_results.get(1.00, {}).get('detected_filaments', 0),
            'detected_100_pixels': target_results.get(1.00, {}).get('total_filament_pixels', 0),
        }
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    print(f"[+] Saved Summary JSON: {json_path}")

    # 9. Master Comparison Bar Chart
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0B0F19')

    scales_labels = ['7% Erosion\n(0.93r)', '3% Erosion\n(0.97r)', '0% Erosion\n(1.00r Full Disk)']
    x = np.arange(len(scales_labels))
    width = 0.35

    # Subplot 1: Whole-Disk Metrics
    whole_d = [summary_stats[s]['whole_disk']['dice'] for s in radius_scales]
    whole_r = [summary_stats[s]['whole_disk']['recall'] for s in radius_scales]
    rects1 = ax[0].bar(x - width/2, whole_d, width, label='Whole-Disk Dice', color='#00d2ff', alpha=0.9)
    rects2 = ax[0].bar(x + width/2, whole_r, width, label='Whole-Disk Recall', color='#00f2fe', alpha=0.9)
    ax[0].set_ylabel('Score', color='white', fontsize=12)
    ax[0].set_title('Whole-Disk Segmentation Performance', color='white', fontsize=13, fontweight='bold')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(scales_labels, color='white', fontsize=11)
    ax[0].set_ylim(0.0, 1.0)
    ax[0].legend(loc='lower right')
    ax[0].grid(axis='y', linestyle='--', alpha=0.2)

    for r in rects1:
        h = r.get_height()
        ax[0].text(r.get_x() + r.get_width()/2., h + 0.02, f"{h:.3f}", ha='center', va='bottom', color='white', fontsize=10)
    for r in rects2:
        h = r.get_height()
        ax[0].text(r.get_x() + r.get_width()/2., h + 0.02, f"{h:.3f}", ha='center', va='bottom', color='white', fontsize=10)

    # Subplot 2: Limb-Region Metrics (Outer 10%)
    limb_d = [summary_stats[s]['limb_region']['dice'] for s in radius_scales]
    limb_r = [summary_stats[s]['limb_region']['recall'] for s in radius_scales]
    rects3 = ax[1].bar(x - width/2, limb_d, width, label='Limb-Region Dice (r > 0.90)', color='#ff007f', alpha=0.9)
    rects4 = ax[1].bar(x + width/2, limb_r, width, label='Limb-Region Recall (r > 0.90)', color='#ff758c', alpha=0.9)
    ax[1].set_ylabel('Score', color='white', fontsize=12)
    ax[1].set_title('Limb-Region (Outer 10% Radius) Performance', color='white', fontsize=13, fontweight='bold')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(scales_labels, color='white', fontsize=11)
    ax[1].set_ylim(0.0, 1.0)
    ax[1].legend(loc='lower right')
    ax[1].grid(axis='y', linestyle='--', alpha=0.2)

    for r in rects3:
        h = r.get_height()
        ax[1].text(r.get_x() + r.get_width()/2., h + 0.02, f"{h:.3f}", ha='center', va='bottom', color='white', fontsize=10)
    for r in rects4:
        h = r.get_height()
        ax[1].text(r.get_x() + r.get_width()/2., h + 0.02, f"{h:.3f}", ha='center', va='bottom', color='white', fontsize=10)

    plt.tight_layout()
    bar_chart_path = os.path.join(output_dir, "limb_boundary_ablation_barchart.png")
    plt.savefig(bar_chart_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Master Ablation Chart: {bar_chart_path}")

    print("\n" + "=" * 80)
    print("[+] ABLATION STUDY COMPLETE! All artifacts saved to outputs/limb_boundary_ablation/")
    print("=" * 80)


if __name__ == '__main__':
    run_ablation_study()
