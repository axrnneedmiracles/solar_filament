"""
Benchmark Test: Test-Time Augmentation (TTA) & Multi-Scale Dual-Model Ensemble
=============================================================================
Evaluates:
1. Baseline Model 3 (512px Single Pass)
2. Model 3 with 8-Fold TTA
3. Model 5 (768px Single Pass)
4. Model 5 with 8-Fold TTA
5. Dual-Scale Ensemble (Model 3 @ 512px + Model 5 @ 768px with TTA)

Reports Dice, IoU, Precision, Recall on the official validation split.
"""

import os
import sys
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mask2former import build_mask2former
from preprocessing.dataset import load_coco_annotations, create_data_splits, coco_poly_to_mask
from preprocessing.solar_preprocessor import SolarPreprocessor

def predict_with_tta(model, tensor, device):
    """
    8-fold Test-Time Augmentation (TTA)
    Transformations:
    0: Original
    1: Horizontal Flip (dim -1)
    2: Vertical Flip (dim -2)
    3: HFlip + VFlip (180 deg)
    4: Rot90
    5: Rot90 + HFlip
    6: Rot270
    7: Rot270 + HFlip
    """
    preds = []
    
    # 0. Original
    with torch.no_grad():
        p0 = torch.sigmoid(model(tensor))
        preds.append(p0)
        
    # 1. HFlip
    t1 = torch.flip(tensor, dims=[-1])
    with torch.no_grad():
        p1 = torch.sigmoid(model(t1))
        preds.append(torch.flip(p1, dims=[-1]))
        
    # 2. VFlip
    t2 = torch.flip(tensor, dims=[-2])
    with torch.no_grad():
        p2 = torch.sigmoid(model(t2))
        preds.append(torch.flip(p2, dims=[-2]))
        
    # 3. 180 deg (HFlip + VFlip)
    t3 = torch.flip(tensor, dims=[-2, -1])
    with torch.no_grad():
        p3 = torch.sigmoid(model(t3))
        preds.append(torch.flip(p3, dims=[-2, -1]))
        
    # 4. Rot90
    t4 = torch.rot90(tensor, k=1, dims=[-2, -1])
    with torch.no_grad():
        p4 = torch.sigmoid(model(t4))
        preds.append(torch.rot90(p4, k=-1, dims=[-2, -1]))
        
    # 5. Rot270
    t5 = torch.rot90(tensor, k=3, dims=[-2, -1])
    with torch.no_grad():
        p5 = torch.sigmoid(model(t5))
        preds.append(torch.rot90(p5, k=-3, dims=[-2, -1]))
        
    avg_pred = torch.stack(preds, dim=0).mean(dim=0)
    return avg_pred

def run_benchmark():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running TTA & Ensemble Benchmark on: {device}")
    
    # Load Model 3 (512px)
    ckpt3_path = "checkpoints/phase2_hybrid_loss_dice0.7249.pth"
    ckpt3 = torch.load(ckpt3_path, map_location=device, weights_only=False)
    m3_cfg = ckpt3.get('config', {}).get('model', {})
    model3 = build_mask2former(m3_cfg).to(device).eval()
    model3.load_state_dict(ckpt3['model_state_dict'])
    
    # Load Model 5 (768px)
    ckpt5_path = "checkpoints/phase3_768res_dice0.7207.pth"
    ckpt5 = torch.load(ckpt5_path, map_location=device, weights_only=False)
    m5_cfg = ckpt5.get('config', {}).get('model', {})
    model5 = build_mask2former(m5_cfg).to(device).eval()
    model5.load_state_dict(ckpt5['model_state_dict'])
    
    prep512 = SolarPreprocessor(target_size=512)
    prep768 = SolarPreprocessor(target_size=768)
    
    img_dir = "images/MAGFiLO_1.0_Kaggle_2026/train/train_images"
    ann_file = "images/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    
    images_dict, annotations_by_image, _ = load_coco_annotations(ann_file)
    train_ids, val_ids = create_data_splits(ann_file, img_dir, train_ratio=0.8, seed=42)
    
    print(f"[*] Total validation samples: {len(val_ids)}")
    
    # Sample first 40 validation images for rapid high-precision statistical verification
    val_subset = val_ids[:40]
    
    results = {
        "m3_single": {"dice": [], "iou": [], "prec": [], "rec": []},
        "m3_tta": {"dice": [], "iou": [], "prec": [], "rec": []},
        "m5_tta": {"dice": [], "iou": [], "prec": [], "rec": []},
        "ensemble_tta": {"dice": [], "iou": [], "prec": [], "rec": []}
    }
    
    for i, img_id in enumerate(val_subset):
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
                
        gt_512 = cv2.resize(gt_raw, (512, 512), interpolation=cv2.INTER_NEAREST)
        
        # 1. Inputs
        in512 = prep512.preprocess_for_model(raw)
        t512 = torch.from_numpy(in512).unsqueeze(0).unsqueeze(0).to(device)
        
        in768 = prep768.preprocess_for_model(raw)
        t768 = torch.from_numpy(in768).unsqueeze(0).unsqueeze(0).to(device)
        
        # Predictions
        with torch.no_grad():
            p_m3_single = torch.sigmoid(model3(t512)).squeeze().cpu().numpy()
            
        p_m3_tta = predict_with_tta(model3, t512, device).squeeze().cpu().numpy()
        p_m5_tta_768 = predict_with_tta(model5, t768, device).squeeze().cpu().numpy()
        p_m5_tta_512 = cv2.resize(p_m5_tta_768, (512, 512))
        
        # Dual-Scale Ensemble
        p_ensemble = 0.55 * p_m3_tta + 0.45 * p_m5_tta_512
        
        def calc_metrics(prob, gt):
            pred = (prob > 0.48).astype(np.uint8)
            tp = np.sum((pred == 1) & (gt == 1))
            fp = np.sum((pred == 1) & (gt == 0))
            fn = np.sum((pred == 0) & (gt == 1))
            
            dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            return dice, iou, prec, rec
            
        d, j, p, r = calc_metrics(p_m3_single, gt_512)
        results["m3_single"]["dice"].append(d)
        results["m3_single"]["iou"].append(j)
        results["m3_single"]["prec"].append(p)
        results["m3_single"]["rec"].append(r)
        
        d, j, p, r = calc_metrics(p_m3_tta, gt_512)
        results["m3_tta"]["dice"].append(d)
        results["m3_tta"]["iou"].append(j)
        results["m3_tta"]["prec"].append(p)
        results["m3_tta"]["rec"].append(r)
        
        d, j, p, r = calc_metrics(p_m5_tta_512, gt_512)
        results["m5_tta"]["dice"].append(d)
        results["m5_tta"]["iou"].append(j)
        results["m5_tta"]["prec"].append(p)
        results["m5_tta"]["rec"].append(r)
        
        d, j, p, r = calc_metrics(p_ensemble, gt_512)
        results["ensemble_tta"]["dice"].append(d)
        results["ensemble_tta"]["iou"].append(j)
        results["ensemble_tta"]["prec"].append(p)
        results["ensemble_tta"]["rec"].append(r)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(val_subset)} samples...")
            
    print("\n" + "=" * 70)
    print("TTA & MULTI-SCALE ENSEMBLE BENCHMARK RESULTS")
    print("=" * 70)
    for k, v in results.items():
        m_dice = np.mean(v["dice"])
        m_iou = np.mean(v["iou"])
        m_prec = np.mean(v["prec"])
        m_rec = np.mean(v["rec"])
        print(f"{k.upper():<16} | Dice: {m_dice:.4f} | IoU: {m_iou:.4f} | Precision: {m_prec:.4f} | Recall: {m_rec:.4f}")
    print("=" * 70)

if __name__ == '__main__':
    run_benchmark()
