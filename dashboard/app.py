"""
Solar Filament Segmentation Dashboard
======================================
Gradio web interface for interactive solar filament detection,
visualization, and morphology analysis.
"""

import os
import sys
import numpy as np
import cv2
import gradio as gr
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predict import SolarFilamentPredictor
from analysis.filament_morphology import analyze_filaments, generate_morphology_report
from visualization.viz import probability_to_heatmap

# Global predictor
predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = SolarFilamentPredictor()
    return predictor


def process_image(image: np.ndarray, method: str, fusion_alpha: float) -> Tuple:
    """Process a solar image and return all visualizations."""
    p = get_predictor()

    if image is None:
        blank = np.zeros((512, 512, 3), dtype=np.uint8)
        return blank, blank, blank, blank, blank, blank, "Please upload a solar image."

    # Convert RGB (Gradio) to BGR (OpenCV)
    if len(image.shape) == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image

    # Run prediction
    results = p.predict(image_bgr, method=method.lower(), fusion_alpha=fusion_alpha)
    target_size = p.image_size

    # 1. Original
    original = cv2.resize(image, (target_size, target_size))

    # 2. Preprocessed
    preproc = results.get('preprocessed', np.zeros((target_size, target_size), dtype=np.uint8))
    preproc_rgb = cv2.cvtColor(preproc, cv2.COLOR_GRAY2RGB) if len(preproc.shape) == 2 else preproc

    # 3. Frangi response
    frangi_resp = results.get('frangi_response', np.zeros((target_size, target_size)))
    frangi_resp_resized = cv2.resize(frangi_resp, (target_size, target_size))
    frangi_heatmap = probability_to_heatmap(frangi_resp_resized.astype(np.float32))
    frangi_heatmap = cv2.cvtColor(frangi_heatmap, cv2.COLOR_BGR2RGB)

    # 4. U-Net probability
    unet_prob = results.get('unet_probability', frangi_resp_resized)
    unet_heatmap = probability_to_heatmap(unet_prob.astype(np.float32))
    unet_heatmap = cv2.cvtColor(unet_heatmap, cv2.COLOR_BGR2RGB)

    # 5. Final mask
    final_mask = results.get('final_mask', np.zeros((target_size, target_size), dtype=np.uint8))
    mask_vis = (final_mask * 255).astype(np.uint8)
    mask_vis_rgb = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2RGB)

    # 6. Overlay
    overlay = results.get('overlay', original)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB) if len(overlay.shape) == 3 else original

    # Morphology analysis
    prob_map = results.get('final_probability', frangi_resp_resized)
    filaments = analyze_filaments(final_mask, prob_map, min_area=30)
    report = generate_morphology_report(filaments)

    inf_time = results.get('inference_time', 0.0)
    report += f"\nInference time: {inf_time * 1000:.1f} ms"
    report += f"\nActive Mode: {method}"
    report += f"\nFusion alpha: {fusion_alpha:.2f}"
    if p.model is None:
        report += "\n[Note: U-Net checkpoint not yet loaded. Running active Frangi + Hessian Ridge detector.]"

    return original, preproc_rgb, frangi_heatmap, unet_heatmap, mask_vis_rgb, overlay_rgb, report


def create_dashboard():
    """Create and return the Gradio interface."""
    with gr.Blocks(
        title="Solar Filament Segmentation System",
    ) as demo:
        gr.Markdown("""
        # ☀️ Solar Filament Segmentation & Space Weather Intelligence System
        ### Classical Computer Vision (Frangi / Hessian) + Deep Learning (Mask2Former / U-Net) Hybrid Platform

        Upload any full-disk H-alpha solar image to detect, segment, and quantitatively analyze solar filaments in real time.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Upload Solar Observation Image (H-alpha)", type="numpy")
                with gr.Row():
                    method = gr.Radio(
                        choices=["Hybrid", "Mask2Former", "UNet", "Frangi"],
                        value="Hybrid",
                        label="Segmentation Pipeline"
                    )
                fusion_alpha = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                    label="Fusion Weight α (0.0 = Pure Frangi, 1.0 = Pure Deep Learning)"
                )
                process_btn = gr.Button("🔍 Segment & Analyze Filaments", variant="primary", size="lg")

        with gr.Row():
            output_original = gr.Image(label="1. Original Observation")
            output_preprocessed = gr.Image(label="2. Preprocessed (Limb Corrected + CLAHE)")
            output_frangi = gr.Image(label="3. Frangi Ridge Response")

        with gr.Row():
            output_unet = gr.Image(label="4. Mask2Former / Deep Learning Probability")
            output_mask = gr.Image(label="5. Cleaned Filament Mask")
            output_overlay = gr.Image(label="6. Detected Filament Overlay")

        with gr.Row():
            morphology_report = gr.Textbox(
                label="📊 Quantitative Filament Morphology Report",
                lines=15,
                max_lines=30,
            )

        process_btn.click(
            fn=process_image,
            inputs=[input_image, method, fusion_alpha],
            outputs=[
                output_original, output_preprocessed, output_frangi,
                output_unet, output_mask, output_overlay,
                morphology_report
            ],
        )

        gr.Markdown("""
        ---
        **System Specs**: NVIDIA CUDA RTX 4050 GPU Support | MAGFiLO 1.0 Dataset | Mask2Former Transformer Architecture
        """)

    return demo


if __name__ == '__main__':
    demo = create_dashboard()
    try:
        demo.launch(share=False, server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
    except OSError:
        print("Port 7860 is busy. Launching on an available port...")
        demo.launch(share=False, server_name="127.0.0.1", theme=gr.themes.Soft())
