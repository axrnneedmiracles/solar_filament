"""
Frangi & Hessian Parameter Study for Solar Filaments
=====================================================
Determines optimal scale ranges for Frangi vesselness and Hessian ridge
filters on H-alpha solar images BEFORE using them as model input channels.

Key concern: Blood-vessel Frangi defaults may over-enhance granulation,
sunspots, and limb artifacts on solar images. This script performs a
systematic sweep and visualizes results.
"""

import os
import sys
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.solar_preprocessor import SolarPreprocessor
from preprocessing.dataset import load_coco_annotations, create_data_splits


# ─────────────────────────────────────────────────────────────
# Core Frangi / Hessian computations (self-contained)
# ─────────────────────────────────────────────────────────────

def frangi_vesselness(image_float: np.ndarray,
                      scales: list,
                      beta: float = 0.5,
                      c_frac: float = 0.5) -> np.ndarray:
    """
    Multi-scale Frangi vesselness for DARK ridges on a bright background.
    Input: float64 image normalized to [0, 1].
    Returns: float64 vesselness map in [0, 1].
    """
    # Invert so dark filaments become bright ridges
    img = 1.0 - image_float
    vesselness = np.zeros_like(img)

    for sigma in scales:
        smoothed = gaussian_filter(img, sigma=sigma)
        Hyy = gaussian_filter(smoothed, sigma=sigma, order=[2, 0]) * (sigma ** 2)
        Hxx = gaussian_filter(smoothed, sigma=sigma, order=[0, 2]) * (sigma ** 2)
        Hxy = gaussian_filter(smoothed, sigma=sigma, order=[1, 1]) * (sigma ** 2)

        trace = Hxx + Hyy
        det = Hxx * Hyy - Hxy ** 2
        disc = np.sqrt(np.maximum(trace ** 2 - 4 * det, 0))

        lambda1 = (trace - disc) / 2
        lambda2 = (trace + disc) / 2

        # Sort by absolute value
        abs1, abs2 = np.abs(lambda1), np.abs(lambda2)
        swap = abs1 > abs2
        l1 = np.where(swap, lambda2, lambda1)
        l2 = np.where(swap, lambda1, lambda2)

        # For bright ridges in inverted image: l2 < 0
        valid = l2 < 0
        Rb = np.zeros_like(l1)
        Rb[valid] = (l1[valid] / (l2[valid] + 1e-10)) ** 2
        S2 = l1 ** 2 + l2 ** 2
        c = c_frac * np.max(np.sqrt(S2)) + 1e-7

        V = np.zeros_like(img)
        V[valid] = np.exp(-Rb[valid] / (2 * beta ** 2)) * (1 - np.exp(-S2[valid] / (2 * c ** 2)))
        vesselness = np.maximum(vesselness, V)

    v_max = vesselness.max()
    if v_max > 0:
        vesselness /= v_max
    return vesselness


def hessian_ridge_response(image_float: np.ndarray,
                           scales: list) -> np.ndarray:
    """
    Multi-scale Hessian ridge response for DARK ridges.
    Returns the maximum ridge response across all scales, normalized to [0, 1].
    """
    responses = []
    for sigma in scales:
        smoothed = gaussian_filter(image_float, sigma=sigma)
        Hyy = gaussian_filter(smoothed, sigma=sigma, order=[2, 0]) * (sigma ** 2)
        Hxx = gaussian_filter(smoothed, sigma=sigma, order=[0, 2]) * (sigma ** 2)
        Hxy = gaussian_filter(smoothed, sigma=sigma, order=[1, 1]) * (sigma ** 2)

        trace = Hxx + Hyy
        det = Hxx * Hyy - Hxy ** 2
        disc = np.sqrt(np.maximum(trace ** 2 - 4 * det, 0))

        lambda2 = (trace + disc) / 2   # Larger eigenvalue
        # Dark ridges: lambda2 > 0 means concave (dark valley)
        response = np.maximum(lambda2, 0)
        responses.append(response)

    combined = np.max(responses, axis=0)
    c_max = combined.max()
    if c_max > 0:
        combined /= c_max
    return combined


# ─────────────────────────────────────────────────────────────
# Parameter Study
# ─────────────────────────────────────────────────────────────

def run_parameter_study():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, 'outputs', 'frangi_study')
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    dataset_root = os.path.join(project_root, 'images', 'MAGFiLO_1.0_Kaggle_2026')
    image_dir = os.path.join(dataset_root, 'train', 'train_images')
    annotations_json = os.path.join(dataset_root, 'train', 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')

    _, val_ids = create_data_splits(annotations_json, image_dir, train_ratio=0.8, seed=42)
    images_dict, annotations_by_image, _ = load_coco_annotations(annotations_json)

    # Pick 5 diverse images (evenly spaced from validation set)
    step = max(1, len(val_ids) // 5)
    sample_ids = [val_ids[i * step] for i in range(5) if i * step < len(val_ids)]

    preprocessor = SolarPreprocessor(target_size=512)

    # Frangi scale sets to test
    frangi_scale_sets = {
        'fine_only':    [0.5, 1.0, 1.5, 2.0],
        'medium':       [1.0, 2.0, 3.0, 5.0],
        'coarse':       [2.0, 4.0, 6.0, 8.0],
        'wide_range':   [0.5, 1.0, 2.0, 3.0, 5.0, 7.0],
        'solar_tuned':  [0.8, 1.5, 2.5, 4.0, 6.0],
    }

    # Hessian scale sets to test
    hessian_scale_sets = {
        'fine':     [0.5, 1.0, 1.5],
        'medium':   [1.0, 2.0, 3.0],
        'coarse':   [2.0, 4.0, 6.0],
        'wide':     [0.5, 1.0, 2.0, 4.0],
        'solar':    [0.8, 1.5, 3.0, 5.0],
    }

    print("=" * 70)
    print("FRANGI & HESSIAN PARAMETER STUDY FOR SOLAR FILAMENTS")
    print("=" * 70)

    # Store IoU-like overlap scores with GT for each configuration
    frangi_scores = {name: [] for name in frangi_scale_sets}
    hessian_scores = {name: [] for name in hessian_scale_sets}

    for img_idx, image_id in enumerate(sample_ids):
        img_info = images_dict[image_id]
        file_name = img_info['file_name']
        img_path = os.path.join(image_dir, file_name)
        base_name = os.path.splitext(file_name)[0]

        print(f"\n[{img_idx+1}/5] Processing: {file_name}")

        raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print(f"  SKIP: Could not load {img_path}")
            continue

        orig_h, orig_w = raw.shape

        # Generate ground truth mask
        from preprocessing.dataset import coco_poly_to_mask
        gt_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in annotations_by_image.get(image_id, []):
            seg = ann.get('segmentation', [])
            if seg:
                ann_mask = coco_poly_to_mask(seg, orig_h, orig_w)
                gt_mask = np.maximum(gt_mask, ann_mask)
        gt_mask_512 = cv2.resize(gt_mask, (512, 512), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        # Preprocess
        preprocessed = preprocessor.preprocess_for_model(raw)  # float32 [0, 1], shape (512, 512)

        # ── Frangi sweep ──
        for name, scales in frangi_scale_sets.items():
            vesselness = frangi_vesselness(preprocessed.astype(np.float64), scales)
            # Compute overlap: threshold at 0.15 and compute Dice with GT
            binary = (vesselness > 0.15).astype(np.float32)
            intersection = (binary * gt_mask_512).sum()
            dice = (2 * intersection + 1) / (binary.sum() + gt_mask_512.sum() + 1)
            frangi_scores[name].append(dice)

        # ── Hessian sweep ──
        for name, scales in hessian_scale_sets.items():
            ridge = hessian_ridge_response(preprocessed.astype(np.float64), scales)
            binary = (ridge > 0.15).astype(np.float32)
            intersection = (binary * gt_mask_512).sum()
            dice = (2 * intersection + 1) / (binary.sum() + gt_mask_512.sum() + 1)
            hessian_scores[name].append(dice)

        # ── Generate visual grid for this image ──
        # Use the best Frangi and Hessian configs based on accumulated scores so far
        best_frangi_name = max(frangi_scores, key=lambda k: np.mean(frangi_scores[k]))
        best_hessian_name = max(hessian_scores, key=lambda k: np.mean(hessian_scores[k]))

        best_frangi = frangi_vesselness(preprocessed.astype(np.float64), frangi_scale_sets[best_frangi_name])
        best_hessian = hessian_ridge_response(preprocessed.astype(np.float64), hessian_scale_sets[best_hessian_name])

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(preprocessed, cmap='gray')
        axes[0].set_title('Preprocessed H-alpha', fontsize=11)
        axes[0].axis('off')

        axes[1].imshow(best_frangi, cmap='hot')
        axes[1].set_title(f'Frangi [{best_frangi_name}]', fontsize=11)
        axes[1].axis('off')

        axes[2].imshow(best_hessian, cmap='hot')
        axes[2].set_title(f'Hessian [{best_hessian_name}]', fontsize=11)
        axes[2].axis('off')

        axes[3].imshow(gt_mask_512, cmap='gray')
        axes[3].set_title('Ground-Truth Mask', fontsize=11)
        axes[3].axis('off')

        plt.suptitle(f'Frangi / Hessian Parameter Study — {base_name}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{base_name}_param_study.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {base_name}_param_study.png")

    # ── Summary: Print mean Dice for each configuration ──
    print("\n" + "=" * 70)
    print("FRANGI VESSELNESS — PARAMETER SWEEP RESULTS (Mean Dice vs GT)")
    print("=" * 70)
    for name, scores in sorted(frangi_scores.items(), key=lambda x: -np.mean(x[1])):
        print(f"  {name:15s}: Mean Dice = {np.mean(scores):.4f}  Scales = {frangi_scale_sets[name]}")

    print("\nHESSIAN RIDGE — PARAMETER SWEEP RESULTS (Mean Dice vs GT)")
    print("=" * 70)
    for name, scores in sorted(hessian_scores.items(), key=lambda x: -np.mean(x[1])):
        print(f"  {name:15s}: Mean Dice = {np.mean(scores):.4f}  Scales = {hessian_scale_sets[name]}")

    best_frangi_name = max(frangi_scores, key=lambda k: np.mean(frangi_scores[k]))
    best_hessian_name = max(hessian_scores, key=lambda k: np.mean(hessian_scores[k]))

    print(f"\n>>> BEST FRANGI CONFIG:  '{best_frangi_name}' — Scales = {frangi_scale_sets[best_frangi_name]}")
    print(f">>> BEST HESSIAN CONFIG: '{best_hessian_name}' — Scales = {hessian_scale_sets[best_hessian_name]}")

    # Save results JSON
    results = {
        'frangi_scores': {k: [float(v) for v in vals] for k, vals in frangi_scores.items()},
        'hessian_scores': {k: [float(v) for v in vals] for k, vals in hessian_scores.items()},
        'best_frangi': best_frangi_name,
        'best_frangi_scales': frangi_scale_sets[best_frangi_name],
        'best_frangi_mean_dice': float(np.mean(frangi_scores[best_frangi_name])),
        'best_hessian': best_hessian_name,
        'best_hessian_scales': hessian_scale_sets[best_hessian_name],
        'best_hessian_mean_dice': float(np.mean(hessian_scores[best_hessian_name])),
    }
    with open(os.path.join(output_dir, 'parameter_study_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/parameter_study_results.json")
    return results


if __name__ == '__main__':
    run_parameter_study()
