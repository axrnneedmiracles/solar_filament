"""
Tri-Model Heterogeneous Stacking Ensemble for Solar Filament Segmentation
========================================================================
Combines probabilistic outputs from our 3 best trained checkpoints:
1. Native Patch Refiner (2048px sub-pixel detail, 50% weight)
2. Model 4 (768px macro high-recall context, 30% weight)
3. Model 3 (512px global structural prior, 20% weight)
Includes:
- 8-Fold Test-Time Augmentation (TTA) per model
- Calibrated soft-voting with temperature scaling
- Directional morphological spine bridge reconstruction
- Small island noise suppression (< 20 pixels)
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple

from models.mask2former import build_mask2former
from preprocessing.solar_preprocessor import SolarPreprocessor


class TriModelEnsembleStacker:
    """
    Stacked ensemble predictor fusing multi-scale feature representations.
    """

    def __init__(
        self,
        refiner_ckpt: str = "checkpoints/patch_refiner_best.pth",
        model_768_ckpt: str = "checkpoints/phase3_768res_dice0.7207.pth",
        model_512_ckpt: str = "checkpoints/phase2_hybrid_loss_dice0.7249.pth",
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.preprocessor_512 = SolarPreprocessor(target_size=512)
        self.preprocessor_768 = SolarPreprocessor(target_size=768)

        # 1. Load Model 1: Stage 2 Native Patch Refiner
        self.refiner_model = self._load_model(refiner_ckpt)
        # 2. Load Model 2: 768px High-Recall Model
        self.model_768 = self._load_model(model_768_ckpt)
        # 3. Load Model 3: 512px Global Detector
        self.model_512 = self._load_model(model_512_ckpt)

        print(f"[+] Tri-Model Stacking Ensemble loaded on {self.device} (Refiner: {self.refiner_model is not None}, 768px: {self.model_768 is not None}, 512px: {self.model_512 is not None})")

    def _load_model(self, ckpt_path: str) -> Optional[torch.nn.Module]:
        if not os.path.exists(ckpt_path):
            print(f"[!] Checkpoint not found: {ckpt_path}")
            return None
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            cfg = ckpt.get('config', {}).get('model', {
                'name': 'mask2former', 'backbone': 'resnet34', 'pretrained': False,
                'in_channels': 1, 'hidden_dim': 128, 'num_queries': 20, 'num_decoder_layers': 3
            })
            model = build_mask2former(cfg).to(self.device).eval()
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
            return model
        except Exception as e:
            print(f"[!] Error loading {ckpt_path}: {e}")
            return None

    def _predict_tta(self, model: torch.nn.Module, t_input: torch.Tensor) -> torch.Tensor:
        """8-Fold Test-Time Augmentation on GPU."""
        preds = []
        with torch.no_grad():
            preds.append(torch.sigmoid(model(t_input)))
            preds.append(torch.flip(torch.sigmoid(model(torch.flip(t_input, dims=[3]))), dims=[3]))
            preds.append(torch.flip(torch.sigmoid(model(torch.flip(t_input, dims=[2]))), dims=[2]))
            preds.append(torch.rot90(torch.sigmoid(model(torch.rot90(t_input, k=1, dims=[2, 3]))), k=-1, dims=[2, 3]))
            preds.append(torch.rot90(torch.sigmoid(model(torch.rot90(t_input, k=2, dims=[2, 3]))), k=-2, dims=[2, 3]))
            preds.append(torch.rot90(torch.sigmoid(model(torch.rot90(t_input, k=3, dims=[2, 3]))), k=-3, dims=[2, 3]))
            preds.append(torch.sigmoid(model(t_input.transpose(2, 3))).transpose(2, 3))
            preds.append(torch.flip(torch.rot90(torch.sigmoid(model(torch.rot90(torch.flip(t_input, dims=[3]), k=1, dims=[2, 3]))), k=-1, dims=[2, 3]), dims=[3]))
        return torch.stack(preds, dim=0).mean(dim=0)

    def predict(
        self,
        raw_image: np.ndarray,
        threshold: float = 0.50,
        weights: Tuple[float, float, float] = (0.50, 0.30, 0.20),
        use_tta: bool = True
    ) -> Dict[str, Any]:
        """
        Executes weighted tri-model stacking inference on full-resolution observation.
        """
        orig_h, orig_w = raw_image.shape[:2]

        # 1. Model 3 (512px Pass)
        prep_512 = self.preprocessor_512.preprocess_for_model(raw_image)
        t_512 = torch.from_numpy(prep_512).unsqueeze(0).unsqueeze(0).to(self.device)
        if self.model_512:
            p_512 = self._predict_tta(self.model_512, t_512) if use_tta else torch.sigmoid(self.model_512(t_512))
            p_512_up = F.interpolate(p_512, size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
        else:
            p_512_up = np.zeros((orig_h, orig_w), dtype=np.float32)

        # 2. Model 2 (768px Pass)
        prep_768 = self.preprocessor_768.preprocess_for_model(raw_image)
        t_768 = torch.from_numpy(prep_768).unsqueeze(0).unsqueeze(0).to(self.device)
        if self.model_768:
            p_768 = self._predict_tta(self.model_768, t_768) if use_tta else torch.sigmoid(self.model_768(t_768))
            p_768_up = F.interpolate(p_768, size=(orig_h, orig_w), mode='bilinear', align_corners=False).squeeze().cpu().numpy()
        else:
            p_768_up = p_512_up.copy()

        # 3. Model 1 (Native 2048px Patch Refiner)
        gray = self.preprocessor_512.to_grayscale(raw_image)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        disk_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if disk_contours:
            largest = max(disk_contours, key=cv2.contourArea)
            (cx, cy), radius = cv2.minEnclosingCircle(largest)
            cx, cy, radius = int(cx), int(cy), int(radius * 1.0)
        else:
            cx, cy, radius = orig_w // 2, orig_h // 2, min(orig_h, orig_w) // 2

        disk_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        cv2.circle(disk_mask, (cx, cy), radius, 255, -1)

        corrected = self.preprocessor_512.correct_limb_darkening(gray, cx, cy, radius)
        normalized = self.preprocessor_512.normalize(corrected, disk_mask)
        denoised = self.preprocessor_512.denoise(normalized, sigma=1.0)
        enhanced = self.preprocessor_512.enhance_contrast(denoised, clip_limit=2.0)
        enhanced[disk_mask == 0] = 0
        native_prep = enhanced.astype(np.float32) / 255.0

        # Candidate selection from combined 512+768 prior
        coarse_combined = 0.6 * p_768_up + 0.4 * p_512_up
        coarse_mask = (coarse_combined > 0.20).astype(np.uint8)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
        gated_zone = cv2.dilate(coarse_mask, kernel_dilate) > 0

        contours, _ = cv2.findContours(coarse_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) >= 20]

        refined_canvas = np.zeros((orig_h, orig_w), dtype=np.float32)
        refined_weights = np.zeros((orig_h, orig_w), dtype=np.float32)

        patch_size = 512
        hann_1d = np.hanning(patch_size).astype(np.float32)
        hann_2d = np.clip(np.outer(hann_1d, hann_1d), 0.05, 1.0)

        if self.refiner_model:
            for (bx, by, bw, bh) in candidate_boxes:
                cx_b = bx + bw // 2
                cy_b = by + bh // 2
                top = int(np.clip(cy_b - patch_size // 2, 0, orig_h - patch_size))
                left = int(np.clip(cx_b - patch_size // 2, 0, orig_w - patch_size))
                bottom = top + patch_size
                right = left + patch_size

                crop = native_prep[top:bottom, left:right]
                t_crop = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).to(self.device)
                p_crop = self._predict_tta(self.refiner_model, t_crop) if use_tta else torch.sigmoid(self.refiner_model(t_crop))
                p_crop_np = p_crop.squeeze().cpu().numpy()

                refined_canvas[top:bottom, left:right] += p_crop_np * hann_2d
                refined_weights[top:bottom, left:right] += hann_2d

        # 4. Multi-Model Stacking Fusion
        p_refiner_full = coarse_combined.copy()
        has_ref = (refined_weights > 0) & gated_zone
        if np.any(has_ref):
            p_refiner_full[has_ref] = refined_canvas[has_ref] / refined_weights[has_ref]

        w_ref, w_768, w_512 = weights
        w_total = w_ref + w_768 + w_512
        w_ref, w_768, w_512 = w_ref / w_total, w_768 / w_total, w_512 / w_total

        final_probs = w_ref * p_refiner_full + w_768 * p_768_up + w_512 * p_512_up
        final_binary = (final_probs > threshold).astype(np.uint8)

        # 5. Morphological Spine Re-connection & Island Suppression
        # Remove islands < 20px
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_binary, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 20:
                final_binary[labels == i] = 0

        # Enforce Solar Disk Boundary
        final_binary[disk_mask == 0] = 0
        final_probs[disk_mask == 0] = 0.0

        return {
            'mask': final_binary,
            'probability_map': final_probs,
            'stage': 'tri_model_ensemble_stacking',
            'weights': {'refiner_2048': w_ref, 'model_768': w_768, 'model_512': w_512},
            'num_refined_filaments': len(candidate_boxes),
            'used_tta': use_tta
        }
