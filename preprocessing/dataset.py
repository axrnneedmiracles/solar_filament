"""
Solar Filament Dataset with Fast Caching
=========================================
PyTorch Dataset class for the MAGFiLO solar filament dataset.
Handles COCO-format polygon annotations, pre-rendered 512x512 mask caching,
data augmentation, and GPU-optimized DataLoaders.

Supports optional 3-channel input mode:
  Channel 0: H-alpha preprocessed grayscale
  Channel 1: Frangi vesselness response (multi-scale)
  Channel 2: Hessian ridge response (multi-scale)
"""

import os
import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict
from scipy.ndimage import gaussian_filter
from preprocessing.solar_preprocessor import SolarPreprocessor


# ─────────────────────────────────────────────────────────────
# Classical feature computation for multi-channel input
# ─────────────────────────────────────────────────────────────

def compute_frangi_channel(image_float: np.ndarray,
                           scales: list = [0.5, 1.0, 1.5, 2.0],
                           beta: float = 0.5,
                           c_frac: float = 0.5) -> np.ndarray:
    """
    Multi-scale Frangi vesselness for DARK ridges on a bright background.
    Input: float64 image normalized to [0, 1], shape (H, W).
    Returns: float32 vesselness map in [0, 1], shape (H, W).
    """
    img = (1.0 - image_float).astype(np.float64)
    vesselness = np.zeros_like(img)

    for sigma in scales:
        smoothed = gaussian_filter(img, sigma=sigma)
        Hyy = gaussian_filter(smoothed, sigma=sigma, order=[2, 0]) * (sigma ** 2)
        Hxx = gaussian_filter(smoothed, sigma=sigma, order=[0, 2]) * (sigma ** 2)
        Hxy = gaussian_filter(smoothed, sigma=sigma, order=[1, 1]) * (sigma ** 2)

        trace = Hxx + Hyy
        det = Hxx * Hyy - Hxy ** 2
        disc = np.sqrt(np.maximum(trace ** 2 - 4 * det, 0))

        lambda1 = (trace - disc) / 2
        lambda2 = (trace + disc) / 2

        abs1, abs2 = np.abs(lambda1), np.abs(lambda2)
        swap = abs1 > abs2
        l1 = np.where(swap, lambda2, lambda1)
        l2 = np.where(swap, lambda1, lambda2)

        valid = l2 < 0
        Rb = np.zeros_like(l1)
        Rb[valid] = (l1[valid] / (l2[valid] + 1e-10)) ** 2
        S2 = l1 ** 2 + l2 ** 2
        c = c_frac * np.max(np.sqrt(S2)) + 1e-7

        V = np.zeros_like(img)
        V[valid] = np.exp(-Rb[valid] / (2 * beta ** 2)) * (1 - np.exp(-S2[valid] / (2 * c ** 2)))
        vesselness = np.maximum(vesselness, V)

    v_max = vesselness.max()
    if v_max > 0:
        vesselness /= v_max
    return vesselness.astype(np.float32)


def compute_hessian_channel(image_float: np.ndarray,
                            scales: list = [0.5, 1.0, 1.5]) -> np.ndarray:
    """
    Multi-scale Hessian ridge response for DARK ridges.
    Input: float64 image normalized to [0, 1], shape (H, W).
    Returns: float32 ridge response in [0, 1], shape (H, W).
    """
    img = image_float.astype(np.float64)
    responses = []
    for sigma in scales:
        smoothed = gaussian_filter(img, sigma=sigma)
        Hyy = gaussian_filter(smoothed, sigma=sigma, order=[2, 0]) * (sigma ** 2)
        Hxx = gaussian_filter(smoothed, sigma=sigma, order=[0, 2]) * (sigma ** 2)
        Hxy = gaussian_filter(smoothed, sigma=sigma, order=[1, 1]) * (sigma ** 2)

        trace = Hxx + Hyy
        det = Hxx * Hyy - Hxy ** 2
        disc = np.sqrt(np.maximum(trace ** 2 - 4 * det, 0))

        lambda2 = (trace + disc) / 2
        response = np.maximum(lambda2, 0)
        responses.append(response)

    combined = np.max(responses, axis=0)
    c_max = combined.max()
    if c_max > 0:
        combined /= c_max
    return combined.astype(np.float32)


def coco_poly_to_mask(segmentation: List[List[float]], height: int, width: int) -> np.ndarray:
    """Convert COCO polygon segmentation to binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts = pts.astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def load_coco_annotations(json_path: str) -> Tuple[Dict, Dict, Dict]:
    """Load COCO annotations and organize by image."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    images_dict = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat for cat in data['categories']}

    annotations_by_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    return images_dict, annotations_by_image, categories


class SolarFilamentDataset(Dataset):
    """
    GPU-optimized PyTorch dataset with preprocessed 512x512 caching.
    """

    def __init__(
        self,
        image_dir: str,
        annotations_json: str,
        image_size: int = 512,
        augment: bool = False,
        image_ids: Optional[List[str]] = None,
        cache_dir: Optional[str] = "cache_512",
        use_frangi_channels: bool = False,
    ):
        self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment
        self.cache_dir = cache_dir
        self.use_frangi_channels = use_frangi_channels
        self.preprocessor = SolarPreprocessor(target_size=image_size)

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Load annotations
        self.images_dict, self.annotations_by_image, self.categories = \
            load_coco_annotations(annotations_json)

        available_files = set(os.listdir(image_dir))

        if image_ids is not None:
            self.image_ids = [
                iid for iid in image_ids
                if iid in self.images_dict
                and self.images_dict[iid]['file_name'] in available_files
                and iid in self.annotations_by_image
            ]
        else:
            self.image_ids = [
                iid for iid, img in self.images_dict.items()
                if img['file_name'] in available_files
                and iid in self.annotations_by_image
            ]

    def __len__(self):
        return len(self.image_ids)

    def _generate_mask(self, image_id: str, height: int, width: int) -> np.ndarray:
        """Generate binary segmentation mask from COCO polygon annotations."""
        mask = np.zeros((height, width), dtype=np.uint8)
        annotations = self.annotations_by_image.get(image_id, [])
        for ann in annotations:
            seg = ann.get('segmentation', [])
            if seg:
                ann_mask = coco_poly_to_mask(seg, height, width)
                mask = np.maximum(mask, ann_mask)
        return mask

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dihedral symmetry & contrast/brightness augmentation.
        Preserves exact sub-pixel boundary sharpness without interpolation artifacts.
        """
        # Exact discrete flips & 90-deg rotations (no interpolation error)
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        if random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        k = random.randint(0, 3)
        if k > 0:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        # Photometric variations
        if random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            image = np.clip(image * factor, 0.0, 1.0)

        if random.random() > 0.5:
            c_factor = random.uniform(0.90, 1.10)
            mean_val = float(np.mean(image))
            image = np.clip((image - mean_val) * c_factor + mean_val, 0.0, 1.0)

        return image.astype(np.float32), mask.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[idx]
        img_info = self.images_dict[image_id]
        file_name = img_info['file_name']
        base_id = os.path.splitext(file_name)[0]
        cache_dir = getattr(self, 'cache_dir', 'cache_512')
        cached_img_path = os.path.join(cache_dir, f"{base_id}_img.npy") if cache_dir else None
        cached_mask_path = os.path.join(cache_dir, f"{base_id}_mask.npy") if cache_dir else None

        if cached_img_path and os.path.exists(cached_img_path) and os.path.exists(cached_mask_path):
            # Ultra-fast load from cache (<1 ms)
            preprocessed = np.load(cached_img_path)
            mask_float = np.load(cached_mask_path)
        else:
            # First-time process & cache
            img_path = os.path.join(self.image_dir, file_name)
            raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raise FileNotFoundError(f"Could not load: {img_path}")

            orig_h, orig_w = raw.shape[:2]
            raw_mask = self._generate_mask(image_id, orig_h, orig_w)

            # Preprocess image
            preprocessed = self.preprocessor.preprocess_for_model(raw)

            # Resize mask
            mask_resized = cv2.resize(raw_mask, (self.image_size, self.image_size),
                                      interpolation=cv2.INTER_NEAREST)
            mask_float = mask_resized.astype(np.float32)

            # Save to cache
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                np.save(cached_img_path, preprocessed)
                np.save(cached_mask_path, mask_float)

        if self.augment:
            preprocessed, mask_float = self._augment(preprocessed, mask_float)

        # ── Build input tensor ──
        if self.use_frangi_channels:
            # 3-channel: [H-alpha, Frangi vesselness, Hessian ridge]
            # Check for cached 3-channel features
            frangi_cache_dir = cache_dir.rstrip('/').rstrip('\\') + '_frangi' if cache_dir else None
            cached_frangi_path = os.path.join(frangi_cache_dir, f"{base_id}_frangi.npy") if frangi_cache_dir else None
            cached_hessian_path = os.path.join(frangi_cache_dir, f"{base_id}_hessian.npy") if frangi_cache_dir else None

            if (cached_frangi_path and os.path.exists(cached_frangi_path)
                    and os.path.exists(cached_hessian_path)):
                frangi_ch = np.load(cached_frangi_path)
                hessian_ch = np.load(cached_hessian_path)
            else:
                frangi_ch = compute_frangi_channel(preprocessed, scales=[0.5, 1.0, 1.5, 2.0])
                hessian_ch = compute_hessian_channel(preprocessed, scales=[0.5, 1.0, 1.5])
                if frangi_cache_dir:
                    os.makedirs(frangi_cache_dir, exist_ok=True)
                    np.save(cached_frangi_path, frangi_ch)
                    np.save(cached_hessian_path, hessian_ch)

            # Apply same augmentation transforms to feature channels
            # (flips/rotations already applied to preprocessed, so recompute on augmented image)
            if self.augment:
                frangi_ch = compute_frangi_channel(preprocessed, scales=[0.5, 1.0, 1.5, 2.0])
                hessian_ch = compute_hessian_channel(preprocessed, scales=[0.5, 1.0, 1.5])

            # Stack: [3, H, W]
            stacked = np.stack([preprocessed, frangi_ch, hessian_ch], axis=0)
            image_tensor = torch.from_numpy(stacked)  # [3, H, W]
        else:
            image_tensor = torch.from_numpy(preprocessed).unsqueeze(0)  # [1, H, W]

        mask_tensor = torch.from_numpy(mask_float).unsqueeze(0)     # [1, H, W]

        return image_tensor, mask_tensor


def create_data_splits(
    annotations_json: str,
    image_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Create reproducible train/validation splits."""
    images_dict, annotations_by_image, _ = load_coco_annotations(annotations_json)
    available_files = set(os.listdir(image_dir))

    valid_ids = [
        iid for iid, img in images_dict.items()
        if img['file_name'] in available_files
        and iid in annotations_by_image
    ]

    valid_ids.sort()
    rng = random.Random(seed)
    rng.shuffle(valid_ids)

    split_idx = int(len(valid_ids) * train_ratio)
    train_ids = valid_ids[:split_idx]
    val_ids = valid_ids[split_idx:]

    return train_ids, val_ids


def get_dataloaders(
    image_dir: str,
    annotations_json: str,
    image_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 2,
    train_ratio: float = 0.8,
    seed: int = 42,
    pin_memory: bool = True,
    use_frangi_channels: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create high-performance train and validation DataLoaders."""
    train_ids, val_ids = create_data_splits(
        annotations_json, image_dir, train_ratio, seed
    )

    train_dataset = SolarFilamentDataset(
        image_dir=image_dir,
        annotations_json=annotations_json,
        image_size=image_size,
        augment=True,
        image_ids=train_ids,
        cache_dir=f"cache_{image_size}",
        use_frangi_channels=use_frangi_channels,
    )

    val_dataset = SolarFilamentDataset(
        image_dir=image_dir,
        annotations_json=annotations_json,
        image_size=image_size,
        augment=False,
        image_ids=val_ids,
        cache_dir=f"cache_{image_size}",
        use_frangi_channels=use_frangi_channels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
