"""
Streamlit Web Dashboard for Solar Filament Segmentation
=========================================================
100% Free Hosting on Streamlit Community Cloud (share.streamlit.io)
"""

import streamlit as st
import numpy as np
import cv2
import PIL.Image
from inference.predict import SolarFilamentPredictor

st.set_page_config(
    page_title="Solar Filament Intelligence System",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ Solar Filament Segmentation & Space Weather Intelligence System")
st.markdown("### Deep Learning (Mask2Former / U-Net) + Classical Computer Vision (Frangi / Hessian) Hybrid Platform")

@st.cache_resource
def get_predictor():
    return SolarFilamentPredictor(config_path="configs/default_config.yaml")

predictor = get_predictor()

col1, col2 = st.sidebar, st.container()

with st.sidebar:
    st.header("⚙️ Settings")
    method = st.selectbox(
        "Segmentation Pipeline",
        ["Hybrid", "Mask2Former", "UNet", "Frangi"]
    )
    fusion_alpha = st.slider(
        "Fusion Weight α",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="0.0 = Pure Frangi, 1.0 = Pure Deep Learning"
    )
    uploaded_file = st.file_uploader("Upload H-alpha Solar Image", type=["jpeg", "jpg", "png", "fts", "fits"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)

    st.subheader("🔍 Processing Results")

    with st.spinner("Running solar limb correction & neural segmentation..."):
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        results = predictor.predict(image_bgr, method=method.lower(), fusion_alpha=fusion_alpha)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(image_np, caption="1. Original Observation", use_container_width=True)
    with c2:
        st.image(results['preprocessed'], caption="2. Preprocessed (Limb Corrected + CLAHE)", use_container_width=True)
    with c3:
        st.image(results['frangi_response'], caption="3. Frangi Ridge Response", use_container_width=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.image(results.get('unet_probability', results.get('final_probability')), caption="4. Neural Probability Map", use_container_width=True)
    with c5:
        st.image(results['final_mask'] * 255, caption="5. Cleaned Filament Mask", use_container_width=True)
    with c6:
        st.image(cv2.cvtColor(results['overlay'], cv2.COLOR_BGR2RGB), caption="6. Detected Filament Overlay", use_container_width=True)

    st.subheader("📊 Filament Morphology Metrics & Space Weather Analysis")
    from analysis.filament_morphology import analyze_filaments, generate_morphology_report
    filaments = analyze_filaments(results['final_mask'], results.get('final_probability'), min_area=40)
    report_text = generate_morphology_report(filaments)
    st.code(report_text, language="text")

else:
    st.info("👈 Please upload a solar H-alpha image from the left sidebar to start analysis.")
