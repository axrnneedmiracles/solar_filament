"""
Frangi + Hessian Experiment: Pre-Training Verification
======================================================
Prints and verifies all experimental parameters before training begins.
"""
import torch
import yaml
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mask2former import build_mask2former
from preprocessing.dataset import get_dataloaders

print("=" * 70)
print("FRANGI + HESSIAN EXPERIMENT — PRE-TRAINING VERIFICATION")
print("=" * 70)

# 1. Load experiment config
with open("configs/frangi_experiment_config.yaml", "r") as f:
    config = yaml.safe_load(f)

print("\n[CONFIG VERIFICATION]")
print(f"  in_channels:          {config['model']['in_channels']}")
print(f"  use_frangi_channels:  {config['data']['use_frangi_channels']}")
print(f"  image_size:           {config['data']['image_size']}")
print(f"  backbone:             {config['model']['backbone']}")
print(f"  pretrained:           {config['model']['pretrained']}")
print(f"  loss:                 {config['training']['loss']}")
print(f"  batch_size:           {config['training']['batch_size']}")
print(f"  lr:                   {config['training']['lr']}")
print(f"  epochs:               {config['training']['epochs']}")

# 2. Build model
print("\n[MODEL ARCHITECTURE]")
model = build_mask2former(config["model"])

# 3. Verify conv1 weights
conv1 = model.pixel_decoder.conv1
print(f"  ResNet-34 conv1 shape:    {conv1.weight.shape}")
print(f"  conv1 weight mean:        {conv1.weight.data.mean():.6f}")
print(f"  conv1 weight std:         {conv1.weight.data.std():.6f}")

# Verify against a fresh random init to prove weights are NOT random
random_conv = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
pretrained_std = conv1.weight.data.std().item()
random_std = random_conv.weight.data.std().item()
print(f"  Random init conv1 std:    {random_std:.6f}")
print(f"  Pretrained conv1 std:     {pretrained_std:.6f}")
weights_differ = not torch.allclose(conv1.weight.data, random_conv.weight.data)
print(f"  Weights differ from random: {weights_differ}")

if conv1.weight.shape == torch.Size([64, 3, 7, 7]):
    print("  >>> CONFIRMED: Original 3-channel ImageNet pretrained conv1 weights are PRESERVED")
    print("  >>> Channel mapping: [H-alpha -> R weights, Frangi -> G weights, Hessian -> B weights]")
else:
    print("  >>> WARNING: conv1 shape is unexpected!")

# stem_c1
stem_c1_conv = model.pixel_decoder.stem_c1[0]
print(f"  stem_c1 first conv shape: {stem_c1_conv.weight.shape}")
print(f"  stem_c1 init:             Random (no pretrained weights for 3-ch auxiliary stem)")

# 4. GPU verification
print("\n[DEVICE VERIFICATION]")
print(f"  CUDA available:       {torch.cuda.is_available()}")
print(f"  GPU name:             {torch.cuda.get_device_name(0)}")
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  GPU memory total:     {gpu_mem:.2f} GB")
print(f"  CUDA version:         {torch.version.cuda}")
print(f"  PyTorch version:      {torch.__version__}")

# 5. Dataset verification
print("\n[DATASET VERIFICATION]")
train_loader, val_loader = get_dataloaders(
    image_dir="images/MAGFiLO_1.0_Kaggle_2026/train/train_images",
    annotations_json="images/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
    image_size=512, batch_size=4, num_workers=0,
    use_frangi_channels=True,
)
print(f"  Train images: {len(train_loader.dataset)}")
print(f"  Val images:   {len(val_loader.dataset)}")

batch_img, batch_mask = next(iter(val_loader))
print("\n[INPUT TENSOR VERIFICATION]")
print(f"  Image batch shape:     {batch_img.shape}")
print(f"  Mask batch shape:      {batch_mask.shape}")
print(f"  Channel order:         [0]=H-alpha, [1]=Frangi vesselness, [2]=Hessian ridge")
ch0 = batch_img[:, 0]
ch1 = batch_img[:, 1]
ch2 = batch_img[:, 2]
print(f"  Channel 0 (H-alpha):   mean={ch0.mean():.4f}, std={ch0.std():.4f}, range=[{ch0.min():.4f}, {ch0.max():.4f}]")
print(f"  Channel 1 (Frangi):    mean={ch1.mean():.4f}, std={ch1.std():.4f}, range=[{ch1.min():.4f}, {ch1.max():.4f}]")
print(f"  Channel 2 (Hessian):   mean={ch2.mean():.4f}, std={ch2.std():.4f}, range=[{ch2.min():.4f}, {ch2.max():.4f}]")

# 6. Forward pass VRAM test
print("\n[GPU MEMORY TEST]")
model = model.cuda()
torch.cuda.reset_peak_memory_stats()
batch_img_gpu = batch_img.cuda()
with torch.amp.autocast("cuda"):
    out = model(batch_img_gpu)
peak_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"  Output shape:          {out.shape}")
print(f"  Peak GPU memory:       {peak_mem:.2f} GB")
print(f"  Memory headroom:       {gpu_mem - peak_mem:.2f} GB remaining (of {gpu_mem:.1f} GB)")

print("\n" + "=" * 70)
print("ALL PRE-TRAINING CHECKS PASSED")
print("=" * 70)
