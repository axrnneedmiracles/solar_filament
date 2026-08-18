"""
Test Single Solar Image CLI
===========================
Run filament segmentation & morphology analysis on any solar test image.
Works with classical Frangi/Hessian immediately, and automatically utilizes
U-Net & Hybrid fusion when trained model checkpoints are available.

Usage:
    python test_single_image.py --image "images/MAGFiLO_1.0_Kaggle_2026/test/test_images/20110120105534Ch.jpeg"
    python test_single_image.py --random-test
"""

import os
import sys
import argparse
import random
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.solar_preprocessor import SolarPreprocessor
from classical.frangi import FrangiPipeline
from analysis.filament_morphology import analyze_filaments, generate_morphology_report, draw_morphology_overlay
from visualization.viz import create_filament_overlay, create_comparison_grid, probability_to_heatmap


def run_pipeline(image_path: str, output_dir: str = "outputs/predictions", method: str = "auto"):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print("=" * 65)
    print(f" SOLAR FILAMENT SEGMENTATION & MORPHOLOGY ANALYSIS")
    print("=" * 65)
    print(f"Target Image: {image_path}")

    # 1. Load image
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print(f"Error: Could not open or read image file: {image_path}")
        return

    orig_h, orig_w = raw_img.shape[:2]
    print(f"Dimensions:   {orig_w}x{orig_h} px")

    # 2. Preprocessor & Classical Pipeline
    preprocessor = SolarPreprocessor(target_size=512)
    frangi_pipe = FrangiPipeline(
        scales=[1, 2, 3, 5, 8],
        alpha=0.5,
        beta=0.5,
        gamma=15.0,
        threshold=0.15,
        min_area=80,
        max_area=50000,
        target_size=512
    )

    t0 = time.time()
    frangi_res = frangi_pipe.process_resized(raw_img)
    frangi_time = time.time() - t0
    print(f"Classical CV (Frangi + Hessian) runtime: {frangi_time*1000:.1f} ms")

    # 3. Check for PyTorch U-Net checkpoint
    unet_available = False
    unet_prob = None
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "best_model.pth")

    try:
        import torch
        from models.unet import build_unet
        if os.path.exists(checkpoint_path):
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            model = build_unet()
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device).eval()

            preproc_model = preprocessor.preprocess_for_model(raw_img)
            tensor = torch.from_numpy(preproc_model).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                unet_prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            unet_available = True
            print(f"U-Net DL Model loaded & evaluated on {device}")
    except Exception as e:
        unet_available = False

    # 4. Determine final mask
    frangi_mask = frangi_res['filament_mask']
    frangi_prob = frangi_res['frangi_probability']

    if unet_available and method in ['auto', 'hybrid']:
        from hybrid.fusion import fuse_predictions
        final_prob = fuse_predictions(unet_prob, frangi_prob, alpha=0.6)
        final_mask = (final_prob > 0.45).astype(np.uint8)
        selected_method = "Hybrid (U-Net + Frangi)"
    elif unet_available and method == 'unet':
        final_prob = unet_prob
        final_mask = (final_prob > 0.5).astype(np.uint8)
        selected_method = "U-Net Deep Learning"
    else:
        final_prob = frangi_prob
        final_mask = frangi_mask
        selected_method = "Frangi + Hessian Ridge Detection"

    print(f"Active Mode:  {selected_method}")

    # 5. Morphology & Analysis
    filaments = analyze_filaments(final_mask, final_prob, min_area=40)
    report_text = generate_morphology_report(filaments)
    print("\n" + report_text)

    # 6. Save visualizations
    img_512 = cv2.resize(raw_img, (512, 512))
    overlay = create_filament_overlay(img_512, final_mask, color=(0, 0, 255), alpha=0.45)
    annotated_overlay = draw_morphology_overlay(overlay, final_mask, filaments)

    # Save output artifacts
    out_overlay_path = os.path.join(output_dir, f"{base_name}_detected_overlay.png")
    out_mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    out_report_path = os.path.join(output_dir, f"{base_name}_morphology.txt")
    out_grid_path = os.path.join(output_dir, f"{base_name}_comparison_grid.png")

    cv2.imwrite(out_overlay_path, annotated_overlay)
    cv2.imwrite(out_mask_path, (final_mask * 255).astype(np.uint8))

    grid_dict = {
        'original': img_512,
        'preprocessed': frangi_res['preprocessed'],
        'frangi_response': (frangi_res['frangi_response'] * 255).astype(np.uint8),
        'unet_probability': probability_to_heatmap(final_prob),
        'final_mask': (final_mask * 255).astype(np.uint8),
        'overlay': annotated_overlay
    }
    grid = create_comparison_grid(grid_dict, target_size=512)
    cv2.imwrite(out_grid_path, grid)

    with open(out_report_path, "w") as f:
        f.write(report_text)

    print("-" * 65)
    print(f"Generated Visual Results in [{output_dir}]:")
    print(f"  * Overlay with BBoxes & Skeletons: {out_overlay_path}")
    print(f"  * Binary Filament Mask:           {out_mask_path}")
    print(f"  * Full Multi-Stage Grid:          {out_grid_path}")
    print(f"  * Detailed Geometry Report:       {out_report_path}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Solar Filament Segmentation Inference")
    parser.add_argument("--image", type=str, help="Path to input solar image (JPEG/PNG)")
    parser.add_argument("--random-test", action="store_true", help="Pick a random image from the test set")
    parser.add_argument("--method", type=str, default="auto", choices=["auto", "hybrid", "unet", "frangi"])
    parser.add_argument("--output-dir", type=str, default="outputs/predictions")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(project_root, "images", "MAGFiLO_1.0_Kaggle_2026", "test", "test_images")

    if args.random_test or not args.image:
        if not os.path.exists(test_dir):
            print(f"Test directory not found: {test_dir}")
            return
        test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not test_files:
            print("No test images found.")
            return
        selected = random.choice(test_files)
        image_path = os.path.join(test_dir, selected)
        print(f"Selected random unseen test image: {selected}")
    else:
        image_path = args.image

    run_pipeline(image_path, args.output_dir, args.method)


if __name__ == "__main__":
    main()
