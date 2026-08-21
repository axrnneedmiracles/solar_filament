"""
Filament Structural Scoring Module
==================================
Computes a transparent, explainable "Filament Structural Score" (0 - 100)
based purely on geometric and morphological segmentation properties.

IMPORTANT SCIENTIFIC NOTE:
This score describes the physical prominence, coherence, and geometric certainty
of detected solar filaments. It is NOT a flare probability or eruption forecast.
"""

import numpy as np
import cv2
from typing import Dict, Any, List


def calculate_filament_structural_score(
    mask: np.ndarray,
    probability_map: np.ndarray = None,
    original_image: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Computes the Filament Structural Score and its component breakdown.

    Parameters:
    -----------
    mask : np.ndarray
        Binary filament segmentation mask (0 or 1, or 0 or 255).
    probability_map : np.ndarray, optional
        Predicted sigmoid/softmax probability map [0.0, 1.0].
    original_image : np.ndarray, optional
        Preprocessed or raw H-alpha solar image.

    Returns:
    --------
    dict containing:
        - total_score: float (0.0 to 100.0)
        - components: dict of normalized 0-100 scores
        - metrics: dict of raw physical measurements
        - breakdown_text: formatted explanation string
    """
    # Normalize mask to binary 0/1
    if mask.max() > 1:
        bin_mask = (mask > 127).astype(np.uint8)
    else:
        bin_mask = (mask > 0.5).astype(np.uint8)

    total_pixels = bin_mask.size
    filament_pixels = int(np.sum(bin_mask))

    if filament_pixels == 0:
        return {
            "total_score": 0.0,
            "components": {
                "area_extent": 0.0,
                "length_continuity": 0.0,
                "aspect_ratio": 0.0,
                "absorption_contrast": 0.0,
                "segmentation_confidence": 0.0,
                "morphological_coherence": 0.0,
            },
            "metrics": {
                "total_area_px": 0,
                "num_filaments": 0,
                "max_length_px": 0.0,
                "mean_aspect_ratio": 0.0,
                "mean_contrast": 0.0,
                "mean_confidence": 0.0,
            },
            "breakdown_text": "No filaments detected in observation field.",
        }

    # 1. Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    # Filter background (label 0)
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    num_components = len(component_areas)

    # 2. Area Extent Score (0 - 100)
    # Typical major filament covers 0.05% to 2.0% of solar disk area
    area_fraction = filament_pixels / float(total_pixels)
    area_score = float(np.clip((area_fraction / 0.015) * 100.0, 0.0, 100.0))

    # 3. Length & Spine Continuity Score (0 - 100)
    # Use bounding box diagonals and skeleton approximations
    max_length = 0.0
    aspect_ratios = []
    
    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        diag = np.sqrt(w ** 2 + h ** 2)
        if diag > max_length:
            max_length = diag
            
        # Elongation approximation: diag^2 / area
        ar = (diag ** 2) / max(area, 1)
        aspect_ratios.append(ar)

    # Long filaments can span 100 to 400 pixels at 512x512
    length_score = float(np.clip((max_length / 250.0) * 100.0, 0.0, 100.0))

    # 4. Aspect Ratio / Elongation Score (0 - 100)
    # Filaments are highly curvilinear (high aspect ratio) vs round sunspots
    mean_ar = float(np.mean(aspect_ratios)) if aspect_ratios else 1.0
    aspect_ratio_score = float(np.clip((mean_ar / 8.0) * 100.0, 0.0, 100.0))

    # 5. Absorption Contrast Score (0 - 100)
    if original_image is not None:
        if len(original_image.shape) == 3:
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = original_image.copy()

        # Match dimensions if needed (e.g. 768px model vs 512px preprocessed image)
        if gray.shape[:2] != bin_mask.shape[:2]:
            gray = cv2.resize(gray, (bin_mask.shape[1], bin_mask.shape[0]))
            
        bg_mask = (bin_mask == 0)
        bg_mean = float(np.mean(gray[bg_mask])) if np.sum(bg_mask) > 0 else 128.0
        fg_mean = float(np.mean(gray[bin_mask == 1]))
        
        # Contrast = (Background - Filament) / Background (filaments are dark absorption features)
        contrast = max(0.0, (bg_mean - fg_mean) / max(bg_mean, 1e-5))
        contrast_score = float(np.clip((contrast / 0.40) * 100.0, 0.0, 100.0))
    else:
        contrast = 0.5
        contrast_score = 65.0

    # 6. Segmentation Confidence Score (0 - 100)
    if probability_map is not None:
        conf_values = probability_map[bin_mask == 1]
        mean_conf = float(np.mean(conf_values)) if len(conf_values) > 0 else 0.5
        confidence_score = float(np.clip(mean_conf * 100.0, 0.0, 100.0))
    else:
        mean_conf = 0.85
        confidence_score = 85.0

    # 7. Morphological Coherence Score (0 - 100)
    # Non-fragmentation index: penalize if mask is broken into 50 tiny fragments
    coherence_score = float(np.clip(100.0 - (num_components - 1) * 4.0, 15.0, 100.0))

    # Weighted Combined Score
    # Weights sum to 1.0
    weights = {
        "area_extent": 0.20,
        "length_continuity": 0.25,
        "aspect_ratio": 0.15,
        "absorption_contrast": 0.15,
        "segmentation_confidence": 0.15,
        "morphological_coherence": 0.10,
    }

    total_score = (
        weights["area_extent"] * area_score +
        weights["length_continuity"] * length_score +
        weights["aspect_ratio"] * aspect_ratio_score +
        weights["absorption_contrast"] * contrast_score +
        weights["segmentation_confidence"] * confidence_score +
        weights["morphological_coherence"] * coherence_score
    )
    total_score = round(float(np.clip(total_score, 0.0, 100.0)), 1)

    components = {
        "area_extent": round(area_score, 1),
        "length_continuity": round(length_score, 1),
        "aspect_ratio": round(aspect_ratio_score, 1),
        "absorption_contrast": round(contrast_score, 1),
        "segmentation_confidence": round(confidence_score, 1),
        "morphological_coherence": round(coherence_score, 1),
    }

    metrics = {
        "total_area_px": filament_pixels,
        "num_filaments": num_components,
        "max_length_px": round(max_length, 1),
        "mean_aspect_ratio": round(mean_ar, 2),
        "mean_contrast": round(contrast, 3),
        "mean_confidence": round(mean_conf, 3),
    }

    breakdown_text = (
        f"Filament Structural Score: {total_score}/100\n"
        f"-----------------------------------------\n"
        f"• Length & Spine Continuity:  {components['length_continuity']}/100 (Max span: {metrics['max_length_px']} px)\n"
        f"• Area Extent:                {components['area_extent']}/100 ({metrics['total_area_px']} px)\n"
        f"• Absorption Contrast:        {components['absorption_contrast']}/100 (Rel. Dip: {metrics['mean_contrast']})\n"
        f"• Curvilinear Aspect Ratio:   {components['aspect_ratio']}/100 (Ratio: {metrics['mean_aspect_ratio']})\n"
        f"• Segmentation Confidence:    {components['segmentation_confidence']}/100 (P: {metrics['mean_confidence']:.2f})\n"
        f"• Morphological Coherence:    {components['morphological_coherence']}/100 ({metrics['num_filaments']} active component(s))\n"
    )

    return {
        "total_score": total_score,
        "components": components,
        "metrics": metrics,
        "breakdown_text": breakdown_text,
    }
