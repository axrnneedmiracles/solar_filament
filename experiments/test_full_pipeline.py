"""
Verify Full Pipeline Inference Test
===================================
Tests end-to-end execution of:
Preprocessing -> Segmentation -> Scoring -> Zoom Crop -> Super-Resolution -> Colormap
"""

import os
import sys
import glob
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predict import SolarFilamentPredictor

def test_pipeline():
    print("[*] Initializing SolarFilamentPredictor...")
    predictor = SolarFilamentPredictor()

    # Find a test image
    img_candidates = glob.glob("images/MAGFiLO_1.0_Kaggle_2026/train/train_images/*.png") + \
                     glob.glob("images/MAGFiLO_1.0_Kaggle_2026/train/train_images/*.jpg") + \
                     glob.glob("images/sample_*.png")
    
    if not img_candidates:
        print("[!] No sample images found, generating synthetic test image...")
        test_img = np.full((512, 512, 3), 160, dtype=np.uint8)
        # Draw curved dark filament
        pts = np.array([[120, 200], [200, 250], [280, 240], [360, 310]], np.int32)
        cv2.polylines(test_img, [pts], False, (40, 40, 40), 8)
    else:
        sample_path = img_candidates[0]
        print(f"[*] Testing on image: {sample_path}")
        test_img = cv2.imread(sample_path)

    print("[*] Running full prediction...")
    res = predictor.predict(test_img, colormap_name="halpha_gold")

    print("\n" + "=" * 60)
    print("[+] FULL PIPELINE TEST RESULTS")
    print("=" * 60)
    print(f"* Preprocessed Image Shape:       {res['preprocessed'].shape}")
    print(f"* Predicted Mask Shape:           {res['final_mask'].shape}")
    print(f"* Filament Structural Score:      {res['structural_score']}/100")
    print(f"* Detected Components:            {res['score_metrics']['num_filaments']}")
    print(f"* Zoomed Filament Crop Shape:     {res['zoomed_filament_crop'].shape}")
    print(f"* 2x Super-Resolution Shape:      {res['super_resolution_2x'].shape}")
    print(f"* 4x Super-Resolution Shape:      {res['super_resolution_4x'].shape}")
    print(f"* False-Color Solar Image Shape:  {res['colored_solar_image'].shape}")
    print(f"* Inference Latency:              {res['inference_time']*1000:.1f} ms")
    print("=" * 60)
    print("\n" + res['score_breakdown'])

if __name__ == '__main__':
    test_pipeline()
