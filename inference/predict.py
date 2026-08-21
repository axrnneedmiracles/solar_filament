"""
Inference Pipeline
==================
Single-image inference for solar filament segmentation.
Supports U-Net, Frangi, and hybrid predictions.
Gracefully handles environments before deep-learning model training.
"""

import os
import sys
import time
import numpy as np
import cv2
import yaml
from typing import Dict, Optional, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.solar_preprocessor import SolarPreprocessor
from classical.frangi import FrangiPipeline
from hybrid.fusion import fuse_predictions
from analysis.filament_scoring import calculate_filament_structural_score
from analysis.filament_cropper import crop_prominent_filament
from analysis.filament_morphology import analyze_filaments, generate_morphology_report
from inference.super_resolution import SolarSuperResolution
from visualization.solar_colormap import apply_solar_colormap


def try_load_model(checkpoint_path: str, config: dict):
    """Attempt to load trained Mask2Former or U-Net model if torch and checkpoint are present."""
    try:
        import torch
        if not os.path.exists(checkpoint_path):
            # Fallback to high-recall or best available checkpoint
            fallback_candidates = [
                'checkpoints/phase3_768res_dice0.7207.pth',
                'checkpoints/phase2_hybrid_loss_dice0.7249.pth',
                'checkpoints/best_model.pth',
            ]
            for fc in fallback_candidates:
                if os.path.exists(fc):
                    checkpoint_path = fc
                    break
            if not os.path.exists(checkpoint_path):
                return None, None, config.get('data', {}).get('image_size', 512), 1

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        saved_config = checkpoint.get('config', config)
        model_name = saved_config.get('model', {}).get('name', 'mask2former').lower()
        image_size = saved_config.get('data', {}).get('image_size', config.get('data', {}).get('image_size', 512))
        in_channels = saved_config.get('model', {}).get('in_channels', 1)
        
        if model_name == 'mask2former':
            from models.mask2former import build_mask2former
            model = build_mask2former(saved_config.get('model', {}))
        else:
            from models.unet import build_unet
            model = build_unet(saved_config.get('model', {}))
            
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(device).eval()
        epoch_info = checkpoint.get('epoch', '?')
        dice_info = checkpoint.get('val_dice', 0.0)
        rec_info = checkpoint.get('val_recall', 0.0)
        print(f"Loaded {model_name.upper()} model [Epoch {epoch_info} | Dice: {dice_info:.4f} | Rec: {rec_info:.4f}] from {checkpoint_path} on {device} (Input size: {image_size}x{image_size}, In Channels: {in_channels})")
        return model, device, image_size, in_channels
    except Exception as e:
        print(f"DL Model not loaded ({e}). Running in Classical CV mode.")
        return None, None, config.get('data', {}).get('image_size', 512), 1


class SolarFilamentPredictor:
    """Complete inference pipeline for solar filament segmentation, scoring, zooming, and enhancement."""

    def __init__(self, checkpoint_path: Optional[str] = None, config_path: Optional[str] = None):
        # Load config
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'configs', 'default_config.yaml'
            )
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {'data': {'image_size': 512}}

        self.image_size = self.config.get('data', {}).get('image_size', 512)

        # Deep learning model
        if checkpoint_path is None:
            # Prefer 768 high-recall model if available, else best_model
            default_chk = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'phase3_768res_dice0.7207.pth')
            if os.path.exists(default_chk):
                checkpoint_path = default_chk
            else:
                checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'best_model.pth')

        self.model, self.device, loaded_size, self.in_channels = try_load_model(checkpoint_path, self.config)
        if loaded_size:
            self.image_size = loaded_size

        self.preprocessor = SolarPreprocessor(target_size=self.image_size)

        # Frangi pipeline
        frangi_cfg = self.config.get('frangi', {})
        self.frangi = FrangiPipeline(
            scales=frangi_cfg.get('scales', [0.5, 1.0, 1.5, 2.0]),
            alpha=frangi_cfg.get('alpha', 0.5),
            beta=frangi_cfg.get('beta', 0.5),
            gamma=frangi_cfg.get('gamma', 15.0),
            threshold=frangi_cfg.get('threshold', 0.15),
            min_area=frangi_cfg.get('min_area', 25),
            max_area=frangi_cfg.get('max_area', 50000),
            target_size=self.image_size,
        )

        # Super-Resolution engine
        self.sr_engine = SolarSuperResolution(device=self.device)

    def predict_dl(self, image: np.ndarray, frangi_resp: np.ndarray = None, hessian_resp: np.ndarray = None, use_tta: bool = False) -> Tuple[Optional[np.ndarray], float]:
        """Run deep learning prediction (1-channel or 3-channel) with optional 8-fold TTA."""
        if self.model is None:
            return None, 0.0

        import torch
        preprocessed = self.preprocessor.preprocess_for_model(image)

        if self.in_channels == 3 and frangi_resp is not None and hessian_resp is not None:
            # 3-channel input: [H-alpha, Frangi, Hessian]
            ch1 = preprocessed.astype(np.float32)
            ch2 = cv2.resize(frangi_resp, (self.image_size, self.image_size)).astype(np.float32)
            ch3 = cv2.resize(hessian_resp, (self.image_size, self.image_size)).astype(np.float32)
            tensor_np = np.stack([ch1, ch2, ch3], axis=0) # [3, H, W]
            tensor = torch.from_numpy(tensor_np).unsqueeze(0).to(self.device)
        else:
            tensor = torch.from_numpy(preprocessed).unsqueeze(0).unsqueeze(0).to(self.device)
            if self.in_channels == 3:
                tensor = tensor.repeat(1, 3, 1, 1)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()

        if use_tta:
            # 8-fold Test-Time Augmentation (TTA)
            preds = []
            # 0. Original
            with torch.no_grad():
                preds.append(torch.sigmoid(self.model(tensor)))
            # 1. Horizontal Flip
            t1 = torch.flip(tensor, dims=[-1])
            with torch.no_grad():
                preds.append(torch.flip(torch.sigmoid(self.model(t1)), dims=[-1]))
            # 2. Vertical Flip
            t2 = torch.flip(tensor, dims=[-2])
            with torch.no_grad():
                preds.append(torch.flip(torch.sigmoid(self.model(t2)), dims=[-2]))
            # 3. 180° (HFlip + VFlip)
            t3 = torch.flip(tensor, dims=[-2, -1])
            with torch.no_grad():
                preds.append(torch.flip(torch.sigmoid(self.model(t3)), dims=[-2, -1]))
            # 4. Rot90
            t4 = torch.rot90(tensor, k=1, dims=[-2, -1])
            with torch.no_grad():
                preds.append(torch.rot90(torch.sigmoid(self.model(t4)), k=-1, dims=[-2, -1]))
            # 5. Rot270
            t5 = torch.rot90(tensor, k=3, dims=[-2, -1])
            with torch.no_grad():
                preds.append(torch.rot90(torch.sigmoid(self.model(t5)), k=-3, dims=[-2, -1]))
            
            prob = torch.stack(preds, dim=0).mean(dim=0)
        else:
            with torch.no_grad():
                logits = self.model(tensor)
                prob = torch.sigmoid(logits)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start

        prob_map = prob.squeeze().cpu().numpy()
        if len(prob_map.shape) == 3:
            prob_map = prob_map[0]
        return prob_map, elapsed

    def predict_ensemble(self, image: np.ndarray, use_tta: bool = True) -> Tuple[Optional[np.ndarray], float]:
        """
        Dual-Scale Ensemble:
        Fuses Model 3 (ResNet-34 + Hybrid Loss @ 512px) with Model 5 (ResNet-34 + Hybrid Loss @ 768px).
        Applies TTA to both resolutions for maximum detection precision and recall.
        """
        import torch
        start = time.time()
        
        # 1. Primary Model 1: Current model
        p1, t1 = self.predict_dl(image, use_tta=use_tta)
        
        # 2. Secondary Model: load other scale
        secondary_ckpt = 'checkpoints/phase2_hybrid_loss_dice0.7249.pth' if self.image_size == 768 else 'checkpoints/phase3_768res_dice0.7207.pth'
        if not os.path.exists(secondary_ckpt):
            return p1, t1

        if not hasattr(self, '_ensemble_secondary_model') or self._ensemble_secondary_model is None:
            sec_model, sec_dev, sec_size, _ = try_load_model(secondary_ckpt, self.config)
            self._ensemble_secondary_model = sec_model
            self._ensemble_secondary_preproc = SolarPreprocessor(target_size=sec_size)
            self._ensemble_secondary_size = sec_size

        if self._ensemble_secondary_model is None:
            return p1, t1

        sec_img = self._ensemble_secondary_preproc.preprocess_for_model(image)
        sec_tensor = torch.from_numpy(sec_img).unsqueeze(0).unsqueeze(0).to(self.device)

        if use_tta:
            preds = []
            with torch.no_grad():
                preds.append(torch.sigmoid(self._ensemble_secondary_model(sec_tensor)))
            t1_s = torch.flip(sec_tensor, dims=[-1])
            with torch.no_grad():
                preds.append(torch.flip(torch.sigmoid(self._ensemble_secondary_model(t1_s)), dims=[-1]))
            t2_s = torch.flip(sec_tensor, dims=[-2])
            with torch.no_grad():
                preds.append(torch.flip(torch.sigmoid(self._ensemble_secondary_model(t2_s)), dims=[-2]))
            t3_s = torch.flip(sec_tensor, dims=[-2, -1])
            with torch.no_grad():
                preds.append(torch.flip(torch.sigmoid(self._ensemble_secondary_model(t3_s)), dims=[-2, -1]))
            p2 = torch.stack(preds, dim=0).mean(dim=0).squeeze().cpu().numpy()
        else:
            with torch.no_grad():
                p2 = torch.sigmoid(self._ensemble_secondary_model(sec_tensor)).squeeze().cpu().numpy()

        if len(p2.shape) == 3:
            p2 = p2[0]

        # Harmonize resolutions to self.image_size
        p2_resized = cv2.resize(p2, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        # Calibrated weighted fusion: 52% Model 3 (512px Dice champion) + 48% Model 5 (768px Recall champion)
        if self.image_size == 512:
            fused_prob = 0.52 * p1 + 0.48 * p2_resized
        else:
            fused_prob = 0.48 * p1 + 0.52 * p2_resized

        elapsed = time.time() - start
        return fused_prob, elapsed

    def predict_frangi(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run Frangi pipeline."""
        return self.frangi.process_resized(image)

    def predict(self, image: np.ndarray, method: str = 'mask2former',
                fusion_alpha: float = 0.5, colormap_name: str = 'halpha_gold',
                selected_filament_rank: int = 1) -> Dict[str, Any]:
        """
        Complete end-to-end inference pipeline:
        Preprocessing -> Segmentation -> Morphology & Scoring -> Multi-Filament Crop -> Super-Resolution -> False-Color.
        """
        results = {}
        results['original'] = image.copy()

        # 1. Preprocessing visualization
        preproc = self.preprocessor.preprocess(image, return_intermediates=True)
        results['preprocessed'] = preproc['preprocessed']
        results['inverted'] = preproc['inverted']

        # 2. Compute Frangi/Hessian only if required by model or user
        needs_frangi = (self.in_channels == 3) or (method.lower() in ['hybrid', 'frangi'])
        if needs_frangi:
            frangi_results = self.predict_frangi(image)
            frangi_resp = frangi_results.get('frangi_response', np.zeros((self.image_size, self.image_size)))
            hessian_resp = frangi_results.get('hessian_response', np.zeros((self.image_size, self.image_size)))
            frangi_prob = frangi_results.get('frangi_probability', np.zeros((self.image_size, self.image_size)))
            frangi_mask = frangi_results.get('filament_mask', np.zeros((self.image_size, self.image_size)))
        else:
            frangi_resp = None
            hessian_resp = None
            frangi_prob = np.zeros((self.image_size, self.image_size), dtype=np.float32)
            frangi_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        results['frangi_response'] = frangi_resp if frangi_resp is not None else np.zeros((self.image_size, self.image_size))
        results['hessian_response'] = hessian_resp if hessian_resp is not None else np.zeros((self.image_size, self.image_size))
        results['frangi_mask'] = frangi_mask
        results['frangi_probability'] = frangi_prob

        # 3. Deep Learning Prediction (Coarse-to-Fine, Tri-Model Stacking, Ensemble, TTA, Standard, or Classical)
        if method.lower() in ['tri_model', 'ensemble_stacking', 'stacking']:
            from inference.ensemble_stacker import TriModelEnsembleStacker
            if not hasattr(self, '_tri_stacker') or self._tri_stacker is None:
                self._tri_stacker = TriModelEnsembleStacker(device=self.device)
            start_stack = time.time()
            st_out = self._tri_stacker.predict(image, threshold=0.50, use_tta=True)
            inference_time = time.time() - start_stack
            dl_prob = cv2.resize(st_out['probability_map'], (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        elif method.lower() in ['coarse_to_fine', 'c2f', 'native_patch', '2stage']:
            from inference.coarse_to_fine import CoarseToFineFilamentPipeline
            if not hasattr(self, '_c2f_pipeline') or self._c2f_pipeline is None:
                self._c2f_pipeline = CoarseToFineFilamentPipeline(
                    global_ckpt_path="checkpoints/phase2_hybrid_loss_dice0.7249.pth",
                    refiner_ckpt_path="checkpoints/patch_refiner_best.pth",
                    device=self.device
                )
            start_c2f = time.time()
            c2f_out = self._c2f_pipeline.predict(image, threshold=0.5)
            c2f_probs_orig = c2f_out['probability_map']
            c2f_mask_orig = c2f_out['mask']
            inference_time = time.time() - start_c2f
            dl_prob = cv2.resize(c2f_probs_orig, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        elif method.lower() in ['ensemble', 'ultra_precision']:
            dl_prob, inference_time = self.predict_ensemble(image, use_tta=True)
        elif method.lower() in ['mask2former_tta', 'tta']:
            dl_prob, inference_time = self.predict_dl(image, frangi_resp=frangi_resp, hessian_resp=hessian_resp, use_tta=True)
        else:
            dl_prob, inference_time = self.predict_dl(image, frangi_resp=frangi_resp, hessian_resp=hessian_resp, use_tta=False)
        results['inference_time'] = inference_time

        if dl_prob is not None:
            results['unet_probability'] = dl_prob
            results['unet_mask'] = (dl_prob > 0.5).astype(np.uint8)

            if method.lower() == 'hybrid' and needs_frangi:
                fused_prob = fuse_predictions(dl_prob, frangi_prob, alpha=fusion_alpha)
                results['hybrid_probability'] = fused_prob
                results['hybrid_mask'] = (fused_prob > 0.5).astype(np.uint8)
                results['final_mask'] = results['hybrid_mask']
                results['final_probability'] = results['hybrid_probability']
            elif method.lower() == 'frangi':
                results['final_mask'] = frangi_mask
                results['final_probability'] = frangi_prob
            else:  # mask2former / ensemble / coarse_to_fine / deep learning default
                results['final_mask'] = results['unet_mask']
                results['final_probability'] = results['unet_probability']
        else:
            results['unet_probability'] = frangi_prob
            results['unet_mask'] = frangi_mask
            results['hybrid_probability'] = frangi_prob
            results['hybrid_mask'] = frangi_mask
            results['final_mask'] = frangi_mask
            results['final_probability'] = frangi_prob

        # 4. Semi-transparent segmentation overlay
        results['overlay'] = create_overlay(image, results['final_mask'], self.image_size)

        # 5. Filament Structural Score (0 - 100) & Detailed Morphology Breakdown
        score_info = calculate_filament_structural_score(
            mask=results['final_mask'],
            probability_map=results['final_probability'],
            original_image=results['preprocessed'],
        )
        results['structural_score'] = score_info['total_score']
        results['score_components'] = score_info['components']
        results['score_metrics'] = score_info['metrics']

        # Detailed per-filament morphology measurements
        filaments_data = analyze_filaments(
            mask=results['final_mask'],
            probability_map=results['final_probability'],
            min_area=20
        )
        results['filaments_morphology'] = filaments_data
        morphology_report = generate_morphology_report(filaments_data)
        
        # Combine structural score + detailed per-filament morphology report
        results['score_breakdown'] = (
            f"{score_info['breakdown_text']}\n\n"
            f"{morphology_report}"
        )

        # 6. Multi-Filament Bounding Box Detection & Selected Filament Cropping
        crop_data = crop_prominent_filament(
            image=results['preprocessed'],
            mask=results['final_mask'],
            selected_rank=selected_filament_rank,
            padding_fraction=0.25,
            min_area=20,
            target_crop_size=(256, 256),
        )
        results['zoomed_filament_crop'] = crop_data['cropped_image']
        results['full_sun_with_bbox'] = crop_data['full_image_with_bbox']
        results['crop_bbox'] = crop_data['bbox']
        results['filaments_list'] = crop_data['filaments_list']
        results['num_filaments'] = crop_data['num_filaments']
        results['selected_rank'] = crop_data['selected_rank']

        # 7. Super-Resolution Enhancement (2x and 4x) on Selected Filament Crop
        sr_data = self.sr_engine.generate_all_scales(results['zoomed_filament_crop'])
        results['super_resolution_2x'] = sr_data['super_res_2x']
        results['super_resolution_4x'] = sr_data['super_res_4x']
        results['super_resolution_disclaimer'] = sr_data['disclaimer']

        # 8. False-Color Solar Observation
        results['colored_solar_image'] = apply_solar_colormap(
            results['preprocessed'],
            palette_name=colormap_name
        )

        return results


def create_overlay(image: np.ndarray, mask: np.ndarray,
                    target_size: int = 512, color: tuple = (0, 0, 255),
                    alpha: float = 0.4) -> np.ndarray:
    """Create semi-transparent overlay of detected filaments on original image."""
    resized = cv2.resize(image, (target_size, target_size))
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    overlay = resized.copy()
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = color

    result = cv2.addWeighted(resized, 1 - alpha, overlay, alpha, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 255), 1)

    return result


SingleImagePredictor = SolarFilamentPredictor


