"""
Limb Error Visualizations Generator
====================================
Generates at least 10 high-resolution limb filament error visualizations comparing:
1. Original H-alpha Image (with solar disk boundary)
2. Ground Truth Mask (with limb annulus)
3. Model Prediction Mask
4. Ground Truth + Prediction Overlay (Cyan/Green/Magenta)
5. Color-Coded Error Map (TP=Green, FP=Red, FN=Blue)
6. Zoomed Limb Region Detail

Saves all figures to: reports/model_comparison/limb_analysis/
"""

import os
import sys
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mask2former import build_mask2former
from preprocessing.dataset import load_coco_annotations, create_data_splits, coco_poly_to_mask
from preprocessing.solar_preprocessor import SolarPreprocessor

os.makedirs("reports/model_comparison/limb_analysis", exist_ok=True)

def generate_limb_visualizations():
    plt.style.use('dark_background')
    device = torch.device('cpu')
    ckpt_path = "checkpoints/phase2_hybrid_loss_dice0.7249.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/phase3_768res_dice0.7207.pth"

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg = checkpoint.get('config', {}).get('model', {})
    target_size = 512

    model = build_mask2former(saved_cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    img_dir = "images/MAGFiLO_1.0_Kaggle_2026/train/train_images"
    ann_file = "images/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    images_dict, annotations_by_image, _ = load_coco_annotations(ann_file)
    train_ids, val_ids = create_data_splits(ann_file, img_dir, train_ratio=0.8, seed=42)

    preprocessor = SolarPreprocessor(target_size=target_size)

    # Find validation samples with near-limb filaments or outer region structures
    count = 0
    for idx, img_id in enumerate(val_ids):
        info = images_dict[img_id]
        fn = info['file_name']
        fp = os.path.join(img_dir, fn)
        raw = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue

        h, w = raw.shape
        gt_raw = np.zeros((h, w), dtype=np.uint8)
        for ann in annotations_by_image.get(img_id, []):
            seg = ann.get('segmentation', [])
            if seg:
                gt_raw = np.maximum(gt_raw, coco_poly_to_mask(seg, h, w))

        gt_512 = cv2.resize(gt_raw, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        if np.sum(gt_512) < 20:
            continue

        # Detect solar disk
        cx_raw, cy_raw, r_raw = preprocessor.detect_solar_disk(raw)
        scale_x = target_size / w
        cx = int(cx_raw * scale_x)
        cy = int(cy_raw * scale_x)
        radius = int(r_raw * scale_x)

        # Run inference
        norm_in = preprocessor.preprocess_for_model(raw)
        tensor_in = torch.from_numpy(norm_in).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor_in)
            probs = torch.sigmoid(logits).cpu().squeeze().numpy()

        pred_bin = (probs > 0.5).astype(np.uint8)

        # Create 5-panel figure
        fig, axes = plt.subplots(1, 5, figsize=(22, 5), facecolor='#0B0F19')
        base_id = os.path.splitext(fn)[0]
        fig.suptitle(f"Limb Region Diagnostic Overlay: {base_id}", fontsize=14, color='white', fontweight='bold')

        raw_512 = cv2.resize(raw, (target_size, target_size))
        vis_raw = cv2.cvtColor(raw_512, cv2.COLOR_GRAY2RGB)
        cv2.circle(vis_raw, (cx, cy), int(radius / 0.93), (0, 255, 255), 2)  # Full limb (Yellow)
        cv2.circle(vis_raw, (cx, cy), radius, (255, 0, 0), 1)  # 0.93 Boundary (Red)
        axes[0].imshow(vis_raw)
        axes[0].set_title("1. Original H-alpha Disk\n(Yellow=True Limb, Red=0.93r)", color='white', fontsize=10)
        axes[0].axis('off')

        # 2. Ground Truth Mask
        vis_gt = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        vis_gt[gt_512 > 0] = [255, 255, 255]
        cv2.circle(vis_gt, (cx, cy), radius, (255, 0, 0), 1)
        axes[1].imshow(vis_gt)
        axes[1].set_title(f"2. Ground Truth Mask\n({np.sum(gt_512)} filament px)", color='white', fontsize=10)
        axes[1].axis('off')

        # 3. Model Prediction
        vis_pred = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        vis_pred[pred_bin > 0] = [0, 255, 255]
        cv2.circle(vis_pred, (cx, cy), radius, (255, 0, 0), 1)
        axes[2].imshow(vis_pred)
        axes[2].set_title(f"3. Mask2Former Prediction\n({np.sum(pred_bin)} pred px)", color='white', fontsize=10)
        axes[2].axis('off')

        # 4. Ground Truth + Prediction Overlay
        vis_overlay = cv2.cvtColor(raw_512, cv2.COLOR_GRAY2RGB)
        vis_overlay[gt_512 > 0] = [0, 255, 0]        # GT = Green
        vis_overlay[pred_bin > 0] = [255, 0, 255]    # Pred = Magenta
        vis_overlay[(gt_512 > 0) & (pred_bin > 0)] = [0, 255, 255]  # Overlap = Cyan
        cv2.circle(vis_overlay, (cx, cy), radius, (255, 255, 255), 1)
        axes[3].imshow(vis_overlay)
        axes[3].set_title("4. Overlay\n(Cyan=Match, Green=GT, Mag=Pred)", color='white', fontsize=10)
        axes[3].axis('off')

        # 5. Error Map (TP=Green, FP=Red, FN=Blue)
        error_map = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        error_map[(pred_bin == 1) & (gt_512 == 1)] = [0, 255, 0]     # TP = Green
        error_map[(pred_bin == 1) & (gt_512 == 0)] = [255, 50, 50]   # FP = Red
        error_map[(pred_bin == 0) & (gt_512 == 1)] = [50, 100, 255]  # FN = Blue
        cv2.circle(error_map, (cx, cy), radius, (200, 200, 200), 1)
        axes[4].imshow(error_map)
        axes[4].set_title("5. Error Map\n(Green=TP, Red=FP, Blue=FN)", color='white', fontsize=10)
        axes[4].axis('off')

        plt.tight_layout()
        save_path = f"reports/model_comparison/limb_analysis/limb_case_{count+1:02d}_{base_id}.png"
        plt.savefig(save_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()

        count += 1
        if count >= 12:
            break

    print(f"[+] Generated {count} limb diagnostic error visualizations in reports/model_comparison/limb_analysis/")

if __name__ == '__main__':
    generate_limb_visualizations()
