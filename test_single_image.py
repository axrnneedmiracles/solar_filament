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
from typing import Optional
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference.predict import SolarFilamentPredictor
from analysis.filament_morphology import analyze_filaments, generate_morphology_report, draw_morphology_overlay
from visualization.viz import create_filament_overlay, create_comparison_grid, probability_to_heatmap


def run_pipeline(image_path: str, output_dir: str = "outputs/predictions", method: str = "auto", checkpoint_path: Optional[str] = None):
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

    # 2. Predictor
    predictor = SolarFilamentPredictor(checkpoint_path=checkpoint_path)
    target_size = predictor.image_size

    # Run inference
    t0 = time.time()
    results = predictor.predict(raw_img, method=method.lower(), fusion_alpha=0.6)
    total_time = time.time() - t0

    dl_available = predictor.model is not None
    if dl_available and method.lower() in ['auto', 'hybrid']:
        selected_method = f"Hybrid (Mask2Former + Frangi) [{target_size}x{target_size}]"
    elif dl_available and method.lower() in ['unet', 'mask2former', 'deeplearning']:
        selected_method = f"Mask2Former Deep Learning [{target_size}x{target_size}]"
    else:
        selected_method = f"Frangi + Hessian Ridge Detection [{target_size}x{target_size}]"

    print(f"Active Model: {selected_method}")
    print(f"Inference Time: {results.get('inference_time', 0.0)*1000:.1f} ms (Total Pipeline: {total_time*1000:.1f} ms)")

    final_mask = results['final_mask']
    final_prob = results['final_probability']

    # 3. Morphology & Analysis
    filaments = analyze_filaments(final_mask, final_prob, min_area=40)
    report_text = generate_morphology_report(filaments)
    print("\n" + report_text)

    # 4. Visualizations
    img_resized = cv2.resize(raw_img, (target_size, target_size))
    overlay = create_filament_overlay(img_resized, final_mask, color=(0, 0, 255), alpha=0.45)
    annotated_overlay = draw_morphology_overlay(overlay, final_mask, filaments)

    # Save output artifacts
    out_overlay_path = os.path.join(output_dir, f"{base_name}_morphology_overlay.png")
    out_mask_path = os.path.join(output_dir, f"{base_name}_filament_mask.png")
    out_grid_path = os.path.join(output_dir, f"{base_name}_analysis_grid.png")
    out_report_path = os.path.join(output_dir, f"{base_name}_morphology_report.txt")
    out_csv_path = os.path.join(output_dir, f"{base_name}_filaments.csv")
    out_json_path = os.path.join(output_dir, f"{base_name}_filaments.json")

    cv2.imwrite(out_overlay_path, annotated_overlay)
    cv2.imwrite(out_mask_path, (final_mask * 255).astype(np.uint8))

    preproc_bgr = cv2.cvtColor(results['preprocessed'], cv2.COLOR_GRAY2BGR) if len(results['preprocessed'].shape) == 2 else results['preprocessed']
    grid_dict = {
        "1. Original H-Alpha": img_resized,
        "2. Solar Preprocessed": preproc_bgr,
        "3. Frangi Ridge Map": probability_to_heatmap(cv2.resize(results['frangi_response'], (target_size, target_size)).astype(np.float32)),
        "4. Model Probability": probability_to_heatmap(final_prob.astype(np.float32)),
        "5. Binary Mask": cv2.cvtColor((final_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
        "6. Annotated Detection": annotated_overlay,
    }
    grid = create_comparison_grid(grid_dict, target_size=target_size)
    cv2.imwrite(out_grid_path, grid)

    with open(out_report_path, "w") as f:
        f.write(report_text)

    from analysis.filament_morphology import export_morphology_csv, export_morphology_json
    export_morphology_csv(filaments, out_csv_path)
    export_morphology_json(filaments, out_json_path)

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
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PyTorch checkpoint (.pth)")
    parser.add_argument("--random-test", action="store_true", help="Pick a random image from the test set")
    parser.add_argument("--method", type=str, default="auto", choices=["auto", "hybrid", "unet", "mask2former", "frangi"])
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

    run_pipeline(image_path, args.output_dir, args.method, args.checkpoint)


if __name__ == "__main__":
    main()
