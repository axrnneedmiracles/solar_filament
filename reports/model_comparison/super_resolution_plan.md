# High-Resolution Filament Zoom & Two-Stage Super-Resolution Architecture
**Architecture Plan:** Coarse-to-Fine Solar Filament Segmentation & AI-Enhanced Visualization  

---

## 1. End-to-End System Pipeline

```
Full-Disk Solar Image (2048x2048)
        ↓
Solar Preprocessor (Limb Darkening + CLAHE)
        ↓
Stage 1: Coarse Mask2Former Segmentation (Whole Disk @ 768px)
        ↓
Multi-Filament Region Locator (Connected Components Ranking)
        ↓
Bounding Box Extraction & Context Margin Cropping (256x256 tiles)
        ↓
Stage 2: Super-Resolution & Sub-Pixel Refinement
        ├── Path A: AI-Enhanced Visualization (2x / 4x Lanczos + Unsharp Masking)
        └── Path B: High-Res Sub-Pixel Spine Refinement (Fine Boundary Extraction)
        ↓
Coordinate Remapping to Original Solar Ephemeris Grid
```

---

## 2. Scientific Disclaimers & Integrity Constraints

1. **Visualization vs. Physics Recovery:** Super-resolution enhances human visual interpretability; it does **not** recover sub-diffraction telescope optics.
2. **Deterministic Processing:** All super-resolution filters are strictly deterministic without generative hallucination.

---

## 3. Implementation Status in Workspace

* **File:** [`inference/super_resolution.py`](file:///c:/Users/aryan/OneDrive/Desktop/solarf/inference/super_resolution.py) (Implemented & Verified)
* **File:** [`analysis/filament_cropper.py`](file:///c:/Users/aryan/OneDrive/Desktop/solarf/analysis/filament_cropper.py) (Implemented & Verified)
* **Dashboard Integration:** Active in [`dashboard/app.py`](file:///c:/Users/aryan/OneDrive/Desktop/solarf/dashboard/app.py) allowing users to select and upscale any individual filament.
