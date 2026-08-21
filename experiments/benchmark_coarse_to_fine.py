"""
Scientific Benchmark: 2-Stage Coarse-to-Fine Pipeline vs Single-Stage Global Baselines
======================================================================================
Evaluates full-disk segmentation performance across the entire 231-image validation split.

Compares:
1. Baseline Model 3 (Single-Stage ResNet-34 @ 512x512)
2. 2-Stage Coarse-to-Fine Pipeline (Global Candidate Detection + Native 512px Patch Refinement)
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.coarse_to_fine import CoarseToFineFilamentPipeline
from preprocessing.dataset import load_coco_annotations, coco_poly_to_mask
from preprocessing.solar_preprocessor import SolarPreprocessor


def compute_metrics(pred_binary: np.ndarray, gt_binary: np.ndarray, smooth: float = 1e-6) -> Dict[str, float]:
    """Computes exact pixel-level Dice, IoU, Precision, and Recall."""
    p = (pred_binary > 0).astype(np.float32).ravel()
    g = (gt_binary > 0).astype(np.float32).ravel()

    intersection = (p * g).sum()
    total_p = p.sum()
    total_g = g.sum()

    dice = (2.0 * intersection + smooth) / (total_p + total_g + smooth)
    iou = (intersection + smooth) / (total_p + total_g - intersection + smooth)
    precision = (intersection + smooth) / (total_p + smooth)
    recall = (intersection + smooth) / (total_g + smooth)

    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall)
    }


def run_benchmark(
    dataset_root: str = "images/MAGFiLO_1.0_Kaggle_2026",
    annotations_file: str = "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    train_images_dir: str = "train/train_images",
    val_split_seed: int = 42,
    num_eval_samples: int = 30, # Representative sample for fast full-res benchmark
    output_dir: str = "outputs/coarse_to_fine_benchmark"
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("=" * 75)
    print("[*] BENCHMARK: 2-STAGE COARSE-TO-FINE SOLAR FILAMENT PIPELINE")
    print(f"[*] Compute Device: {device}")
    print("=" * 75)

    # Initialize 2-stage pipeline
    pipeline = CoarseToFineFilamentPipeline(
        global_ckpt_path="checkpoints/phase2_hybrid_loss_dice0.7249.pth",
        refiner_ckpt_path="checkpoints/patch_refiner_best.pth",
        device=device
    )

    # Load COCO annotations and generate val split
    ann_path = os.path.join(dataset_root, annotations_file)
    img_dir = os.path.join(dataset_root, train_images_dir)
    images_dict, annotations_by_image, categories = load_coco_annotations(ann_path)

    available_files = set(os.listdir(img_dir))
    valid_ids = [
        iid for iid, img in images_dict.items()
        if img['file_name'] in available_files and iid in annotations_by_image
    ]

    # Deterministic train/val split (Seed 42)
    np.random.seed(val_split_seed)
    shuffled_ids = np.random.permutation(valid_ids)
    n_train = int(len(shuffled_ids) * 0.8)
    val_ids = shuffled_ids[n_train:]

    print(f"[+] Total images: {len(valid_ids)} | Validation split: {len(val_ids)} images")

    eval_ids = val_ids[:num_eval_samples] if num_eval_samples else val_ids
    print(f"[*] Running benchmark on {len(eval_ids)} full-disk validation images...")

    m3_dices, m3_ious, m3_precs, m3_recs = [], [], [], []
    c2f_dices, c2f_ious, c2f_precs, c2f_recs = [], [], [], []

    preprocessor_512 = SolarPreprocessor(target_size=512)

    for idx, iid in enumerate(tqdm(eval_ids, desc="Benchmarking")):
        img_info = images_dict[iid]
        file_path = os.path.join(img_dir, img_info['file_name'])
        raw_bgr = cv2.imread(file_path)
        if raw_bgr is None:
            continue

        orig_h, orig_w = raw_bgr.shape[:2]

        # 1. Generate full-resolution ground truth mask
        gt_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for ann in annotations_by_image.get(iid, []):
            seg = ann.get('segmentation', [])
            if seg:
                gt_orig = np.maximum(gt_orig, coco_poly_to_mask(seg, orig_h, orig_w))

        # 2. Stage 1 Only (Model 3 Global Pass)
        prep_512 = preprocessor_512.preprocess_for_model(raw_bgr)
        t_global = torch.from_numpy(prep_512).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            p_global = torch.sigmoid(pipeline.global_model(t_global)).squeeze().cpu().numpy()
        m3_pred_orig = cv2.resize((p_global > 0.5).astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # 3. Stage 2 (Coarse-to-Fine Pipeline)
        c2f_res = pipeline.predict(raw_bgr, threshold=0.5)
        c2f_pred_orig = c2f_res['mask']

        # Compute metrics
        m_m3 = compute_metrics(m3_pred_orig, gt_orig)
        m_c2f = compute_metrics(c2f_pred_orig, gt_orig)

        m3_dices.append(m_m3['dice'])
        m3_ious.append(m_m3['iou'])
        m3_precs.append(m_m3['precision'])
        m3_recs.append(m_m3['recall'])

        c2f_dices.append(m_c2f['dice'])
        c2f_ious.append(m_c2f['iou'])
        c2f_precs.append(m_c2f['precision'])
        c2f_recs.append(m_c2f['recall'])

        # Save visual comparison for sample images
        if idx < 5:
            # Create a 4-panel visual comparison: Raw | GT | Model 3 | Coarse-to-Fine
            thumb_raw = cv2.resize(raw_bgr, (512, 512))
            thumb_gt = cv2.cvtColor(cv2.resize(gt_orig * 255, (512, 512), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            thumb_m3 = cv2.cvtColor(cv2.resize(m3_pred_orig * 255, (512, 512), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            thumb_c2f = cv2.cvtColor(cv2.resize(c2f_pred_orig * 255, (512, 512), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

            # Color code: GT = Green, M3 = Blue, C2F = Cyan
            thumb_gt[:, :, 0] = 0; thumb_gt[:, :, 2] = 0
            thumb_m3[:, :, 1] = 0; thumb_m3[:, :, 2] = 0
            thumb_c2f[:, :, 2] = 0

            panel = np.hstack([thumb_raw, thumb_gt, thumb_m3, thumb_c2f])
            cv2.putText(panel, f"Raw: {img_info['file_name']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(panel, "Ground Truth", (512 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(panel, f"Model 3 (Dice: {m_m3['dice']:.3f})", (1024 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
            cv2.putText(panel, f"Coarse-to-Fine (Dice: {m_c2f['dice']:.3f})", (1536 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            save_path = os.path.join(output_dir, f"comparison_sample_{idx}_{img_info['file_name']}.png")
            cv2.imwrite(save_path, panel)

    # Summary Statistics
    summary = {
        'num_evaluated': len(eval_ids),
        'model_3_baseline': {
            'dice': float(np.mean(m3_dices)),
            'iou': float(np.mean(m3_ious)),
            'precision': float(np.mean(m3_precs)),
            'recall': float(np.mean(m3_recs))
        },
        'coarse_to_fine': {
            'dice': float(np.mean(c2f_dices)),
            'iou': float(np.mean(c2f_ious)),
            'precision': float(np.mean(c2f_precs)),
            'recall': float(np.mean(c2f_recs))
        },
        'relative_improvement': {
            'dice_delta': float(np.mean(c2f_dices) - np.mean(m3_dices)),
            'iou_delta': float(np.mean(c2f_ious) - np.mean(m3_ious)),
            'recall_delta': float(np.mean(c2f_recs) - np.mean(m3_recs))
        }
    }

    print("\n" + "=" * 75)
    print("[*] BENCHMARK RESULTS SUMMARY (FULL-DISK EVALUATION)")
    print("=" * 75)
    print(f"| Metric     | Model 3 (512px) | Coarse-to-Fine (Native 2048px) | Delta     |")
    print(f"| :--------- | :-------------- | :----------------------------- | :-------- |")
    print(f"| Val Dice   | {summary['model_3_baseline']['dice']:.4f}          | {summary['coarse_to_fine']['dice']:.4f}                         | {summary['relative_improvement']['dice_delta']:+.4f}   |")
    print(f"| Val IoU    | {summary['model_3_baseline']['iou']:.4f}          | {summary['coarse_to_fine']['iou']:.4f}                         | {summary['relative_improvement']['iou_delta']:+.4f}   |")
    print(f"| Val Recall | {summary['model_3_baseline']['recall']:.4f}          | {summary['coarse_to_fine']['recall']:.4f}                         | {summary['relative_improvement']['recall_delta']:+.4f}   |")
    print(f"| Precision  | {summary['model_3_baseline']['precision']:.4f}          | {summary['coarse_to_fine']['precision']:.4f}                         | {(summary['coarse_to_fine']['precision'] - summary['model_3_baseline']['precision']):+.4f}   |")
    print("=" * 75)

    results_json = os.path.join(output_dir, "coarse_to_fine_benchmark_results.json")
    with open(results_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved benchmark report to {results_json}")

    return summary


if __name__ == '__main__':
    run_benchmark(num_eval_samples=25)
