"""
PyTorch Dataset for Native-Scale 512x512 Cached Patches (High-Speed RAM Cached)
================================================================================
Preloads preprocessed native patches into memory as uint8 (takes ~1.9 GB RAM total),
eliminating Windows filesystem bottlenecks and enabling >500 fps GPU training.
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict


class NativePatchDataset(Dataset):
    """
    Ultra High-Speed In-Memory PyTorch dataset for 512x512 native filament patches.
    """

    def __init__(
        self,
        cache_dir: str = "cache_patch_512",
        split: str = "train",
        augment: bool = False,
        preload: bool = True
    ):
        self.split_dir = os.path.join(cache_dir, split)
        self.augment = augment
        self.preload = preload
        self.manifest_path = os.path.join(cache_dir, f"{split}_manifest.json")

        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}. Run preprocessing/patch_extractor.py first.")

        with open(self.manifest_path, 'r') as f:
            self.manifest = json.load(f)

        self.images: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []

        if self.preload:
            print(f"[*] Preloading {len(self.manifest)} '{split}' patches into RAM for maximum throughput...", flush=True)
            for idx, record in enumerate(self.manifest):
                img_path = os.path.join(self.split_dir, record['img_file'])
                mask_path = os.path.join(self.split_dir, record['mask_file'])

                img = np.load(img_path)
                mask = np.load(mask_path)

                # Store as uint8 in RAM to keep footprint ~1.9 GB
                self.images.append((img * 255.0).astype(np.uint8))
                self.masks.append(mask.astype(np.uint8))

                if (idx + 1) % 2500 == 0 or (idx + 1) == len(self.manifest):
                    print(f"    Loaded {idx + 1}/{len(self.manifest)} patches...", flush=True)

            print(f"[+] '{split}' RAM cache ready ({len(self.images)} patches in memory).", flush=True)

    def __len__(self) -> int:
        return len(self.manifest)

    def _augment(self, image: np.ndarray, mask: np.ndarray, current_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact dihedral flips, 90-degree rotations, and Copy-Paste rare filament synthesis.
        Zero interpolation error — preserves sub-pixel filament boundary sharpness.
        """
        # 1. Dihedral transformations
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

        # 2. Copy-Paste / CutMix Augmentation (30% chance)
        if random.random() < 0.30 and len(self.images) > 1:
            rand_idx = random.randint(0, len(self.images) - 1)
            if rand_idx != current_idx:
                other_img = self.images[rand_idx].astype(np.float32) / 255.0 if self.preload else image
                other_mask = self.masks[rand_idx].astype(np.float32) if self.preload else mask
                if other_mask.sum() > 30:
                    fg = other_mask > 0
                    # Alpha blend overlapping intensity
                    image[fg] = 0.55 * image[fg] + 0.45 * other_img[fg]
                    mask[fg] = 1.0

        # 3. Slight photometric variation & gamma
        if random.random() > 0.5:
            factor = random.uniform(0.88, 1.12)
            gamma = random.uniform(0.92, 1.08)
            image = np.clip((image ** gamma) * factor, 0.0, 1.0)

        return image.astype(np.float32), mask.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.preload:
            image = self.images[idx].astype(np.float32) / 255.0
            mask = self.masks[idx].astype(np.float32)
        else:
            record = self.manifest[idx]
            img_path = os.path.join(self.split_dir, record['img_file'])
            mask_path = os.path.join(self.split_dir, record['mask_file'])
            image = np.load(img_path)
            mask = np.load(mask_path)

        if self.augment:
            image, mask = self._augment(image, mask, current_idx=idx)

        t_img = torch.from_numpy(image).unsqueeze(0)  # [1, 512, 512]
        t_mask = torch.from_numpy(mask).unsqueeze(0) # [1, 512, 512]
        return t_img, t_mask


def get_patch_dataloaders(
    cache_dir: str = "cache_patch_512",
    batch_size: int = 16,
    preload: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """Create GPU DataLoaders for train and validation patches with in-memory caching."""
    train_ds = NativePatchDataset(cache_dir=cache_dir, split="train", augment=True, preload=preload)
    val_ds = NativePatchDataset(cache_dir=cache_dir, split="val", augment=False, preload=preload)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader
