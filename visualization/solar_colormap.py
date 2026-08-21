"""
Solar Color Representation & Multi-Wavelength Visualizer
========================================================
Applies calibrated solar false-color palettes to monochromatic H-alpha solar observations
for enhanced human visual inspection, feature tracking, and multi-band alignment.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Dict, Any


def apply_solar_colormap(
    gray_image: np.ndarray,
    palette_name: str = "halpha_gold",
) -> np.ndarray:
    """
    Applies calibrated solar color lookup tables to a grayscale H-alpha image.

    Palettes:
    ---------
    - 'halpha_gold': Classic GONG/BBSO warm golden-orange H-alpha palette.
    - 'aia_304': SDO AIA 304 Angstrom Chromosphere / Transition Region red-orange.
    - 'aia_171': SDO AIA 171 Angstrom Quiet Corona gold-yellow.
    - 'inferno': High-contrast perceptual colormap for filament absorption tracking.
    """
    if gray_image is None or gray_image.size == 0:
        return gray_image

    if len(gray_image.shape) == 3:
        gray = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_image.copy()

    # Normalize to [0, 255]
    if gray.dtype != np.uint8:
        gray_norm = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    else:
        gray_norm = gray

    name = palette_name.lower().strip()

    if name in ("halpha_gold", "gold", "halpha"):
        # Custom Solar Gold/Orange LUT
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            r = int(np.clip(i * 1.05, 0, 255))
            g = int(np.clip((i ** 1.15) * 0.72, 0, 255))
            b = int(np.clip((i ** 1.35) * 0.28, 0, 255))
            lut[i, 0] = [b, g, r]  # BGR
        colored = cv2.LUT(cv2.cvtColor(gray_norm, cv2.COLOR_GRAY2BGR), lut)

    elif name in ("aia_304", "sdo_304", "red_orange"):
        # SDO 304 Angstrom He II palette (deep red-orange)
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            r = int(np.clip(i * 1.10, 0, 255))
            g = int(np.clip((i ** 1.25) * 0.45, 0, 255))
            b = int(np.clip((i ** 1.40) * 0.15, 0, 255))
            lut[i, 0] = [b, g, r]
        colored = cv2.LUT(cv2.cvtColor(gray_norm, cv2.COLOR_GRAY2BGR), lut)

    elif name in ("aia_171", "sdo_171", "golden"):
        # SDO 171 Angstrom Fe IX palette (bright golden-yellow)
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for i in range(256):
            r = int(np.clip(i * 1.05, 0, 255))
            g = int(np.clip(i * 0.92, 0, 255))
            b = int(np.clip((i ** 1.30) * 0.35, 0, 255))
            lut[i, 0] = [b, g, r]
        colored = cv2.LUT(cv2.cvtColor(gray_norm, cv2.COLOR_GRAY2BGR), lut)

    elif name in ("inferno", "heat"):
        colored = cv2.applyColorMap(gray_norm, cv2.COLORMAP_INFERNO)

    else:
        colored = cv2.applyColorMap(gray_norm, cv2.COLORMAP_HOT)

    return colored
