"""
Segmentation Losses for Solar Filament Detection
=================================================
Implements:
1. Soft Dice Loss (global region overlap)
2. Focal Loss (hard example mining & severe class imbalance handling)
3. Boundary Dice Loss (morphological boundary edge supervision)
4. Combined Hybrid Losses (Dice+BCE, Dice+Focal, Dice+Focal+Boundary)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / \
               (probs_flat.sum() + targets_flat.sum() + self.smooth)

        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """
    Combined Dice + Binary Cross-Entropy loss.
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(logits, targets)
        bce = self.bce_loss(logits, targets)
        return self.dice_weight * dice + self.bce_weight * bce


class FocalLoss(nn.Module):
    """
    Focal Loss with balanced class weighting (alpha) and modulating factor (gamma).
    Down-weights easy background pixels and forces focus on rare filament pixels.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = alpha_t * ((1.0 - pt) ** self.gamma)
        return (focal_weight * bce).mean()


class BoundaryLoss(nn.Module):
    """
    Morphological Boundary Loss:
    Extracts the thin boundary contour of ground truth masks using morphological gradient
    and calculates dedicated boundary Dice to penalize edge blurring and fragmentation.
    """

    def __init__(self, kernel_size: int = 3, smooth: float = 1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            dilated = F.max_pool2d(targets, kernel_size=self.kernel_size, stride=1, padding=self.padding)
            eroded = -F.max_pool2d(-targets, kernel_size=self.kernel_size, stride=1, padding=self.padding)
            boundary = (dilated - eroded).clamp(0.0, 1.0)

        probs = torch.sigmoid(logits)
        probs_b = (probs * boundary).view(-1)
        targets_b = (targets * boundary).view(-1)

        intersection = (probs_b * targets_b).sum()
        boundary_dice = (2.0 * intersection + self.smooth) / \
                        (probs_b.sum() + targets_b.sum() + self.smooth)

        return 1.0 - boundary_dice


class DiceFocalLoss(nn.Module):
    """
    Combined Soft Dice + Focal Loss.
    """

    def __init__(self, dice_weight: float = 0.5, focal_weight: float = 0.5,
                 alpha: float = 0.75, gamma: float = 2.0, smooth: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = DiceLoss(smooth=smooth)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, targets) + \
               self.focal_weight * self.focal(logits, targets)


class DiceFocalBoundaryLoss(nn.Module):
    """
    Combined Soft Dice + Focal + Morphological Boundary Loss.
    Combines:
    - Region-level spatial overlap (Dice: 40%)
    - Class imbalance hard pixel mining (Focal: 30%)
    - Sub-pixel boundary contour precision (Boundary: 30%)
    """

    def __init__(
        self,
        dice_weight: float = 0.4,
        focal_weight: float = 0.3,
        boundary_weight: float = 0.3,
        alpha: float = 0.75,
        gamma: float = 2.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.dice = DiceLoss(smooth=smooth)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.boundary = BoundaryLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(logits, targets)
            + self.focal_weight * self.focal(logits, targets)
            + self.boundary_weight * self.boundary(logits, targets)
        )


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss:
    Optimizes for high recall (beta=0.65) and precision (alpha=0.35) simultaneously.
    Focal exponent gamma (>1) penalizes hard boundary errors.
    """
    def __init__(self, alpha: float = 0.35, beta: float = 0.65, gamma: float = 1.33, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        targets_flat = targets.view(-1)

        tp = (probs * targets_flat).sum()
        fp = (probs * (1.0 - targets_flat)).sum()
        fn = ((1.0 - probs) * targets_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow(1.0 - tversky, self.gamma)
        return focal_tversky


class FocalTverskyBoundaryLoss(nn.Module):
    """
    Combined Focal Tversky Loss + Morphological Boundary Loss.
    """
    def __init__(
        self,
        tversky_weight: float = 0.70,
        boundary_weight: float = 0.30,
        alpha: float = 0.35,
        beta: float = 0.65,
        gamma: float = 1.33,
        smooth: float = 1.0
    ):
        super().__init__()
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        self.focal_tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, smooth=smooth)
        self.boundary = BoundaryLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.tversky_weight * self.focal_tversky(logits, targets)
            + self.boundary_weight * self.boundary(logits, targets)
        )


def build_loss(loss_name: str = 'focal_tversky_boundary', config: dict = None) -> nn.Module:
    """Factory function for segmentation loss functions."""
    name = (loss_name or 'focal_tversky_boundary').lower()
    if name in ('focal_tversky_boundary', 'tversky_boundary', 'focal_tversky'):
        return FocalTverskyBoundaryLoss(
            tversky_weight=config.get('tversky_weight', 0.70) if config else 0.70,
            boundary_weight=config.get('boundary_weight', 0.30) if config else 0.30,
            alpha=config.get('tversky_alpha', 0.35) if config else 0.35,
            beta=config.get('tversky_beta', 0.65) if config else 0.65,
            gamma=config.get('tversky_gamma', 1.33) if config else 1.33,
        )
    elif name in ('dice_focal_boundary', 'dice_boundary_focal', 'hybrid'):
        return DiceFocalBoundaryLoss(
            dice_weight=config.get('dice_weight', 0.4) if config else 0.4,
            focal_weight=config.get('focal_weight', 0.3) if config else 0.3,
            boundary_weight=config.get('boundary_weight', 0.3) if config else 0.3,
            alpha=config.get('focal_alpha', 0.75) if config else 0.75,
            gamma=config.get('focal_gamma', 2.0) if config else 2.0,
        )
    elif name in ('dice_focal', 'focal_dice'):
        return DiceFocalLoss(
            dice_weight=config.get('dice_weight', 0.5) if config else 0.5,
            focal_weight=config.get('focal_weight', 0.5) if config else 0.5,
            alpha=config.get('focal_alpha', 0.75) if config else 0.75,
            gamma=config.get('focal_gamma', 2.0) if config else 2.0,
        )
    elif name == 'focal':
        return FocalLoss()
    elif name == 'dice':
        return DiceLoss()
    else:
        return DiceBCELoss()
