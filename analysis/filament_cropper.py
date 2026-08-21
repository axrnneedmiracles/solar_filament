"""
Filament Bounding Box & Region Cropper
======================================
Automatically locates all detected filaments from segmentation masks,
draws labeled bounding boxes for all components on the full solar disk,
and produces high-resolution crops for any selected filament for zoom and super-resolution.
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional, List


def crop_prominent_filament(
    image: np.ndarray,
    mask: np.ndarray,
    selected_rank: int = 1,
    padding_fraction: float = 0.25,
    min_area: int = 20,
    target_crop_size: Optional[Tuple[int, int]] = (256, 256),
) -> Dict[str, Any]:
    """
    Finds all filament components in the mask, draws bounding boxes for all of them,
    and extracts a cropped region around the requested filament (selected_rank).

    Parameters:
    -----------
    image : np.ndarray
        Original or preprocessed solar image (H x W or H x W x C).
    mask : np.ndarray
        Binary filament mask (H x W).
    selected_rank : int
        1-based rank of the filament to crop (1 = largest/primary, 2 = 2nd largest, etc.).
    padding_fraction : float
        Proportion of bounding box dimensions to add as contextual margin (default 0.25 = 25%).
    min_area : int
        Minimum pixel area threshold to consider a filament (default 20 px).
    target_crop_size : tuple of int, optional
        If set, resizes the crop to standard dimensions for visualization (e.g. (256, 256)).

    Returns:
    --------
    dict containing:
        - cropped_image: np.ndarray
        - cropped_mask: np.ndarray
        - full_image_with_bbox: np.ndarray (Original image with bounding boxes for ALL filaments)
        - bbox: (x1, y1, x2, y2)
        - num_filaments: int
        - filaments_list: list of dicts with per-filament metadata
        - selected_rank: int
        - has_filament: bool
    """
    H, W = mask.shape[:2]
    
    # Ensure image matches mask dimensions
    if image.shape[:2] != (H, W):
        image = cv2.resize(image, (W, H))

    # Ensure binary mask
    if mask.max() > 1:
        bin_mask = (mask > 127).astype(np.uint8)
    else:
        bin_mask = (mask > 0.5).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

    # Filter components by min_area (excluding background label 0)
    valid_labels = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    # Sort by area descending
    valid_labels.sort(key=lambda idx: stats[idx, cv2.CC_STAT_AREA], reverse=True)

    if not valid_labels:
        cx, cy = W // 2, H // 2
        half_s = min(W, H) // 4
        x1, y1 = max(0, cx - half_s), max(0, cy - half_s)
        x2, y2 = min(W, cx + half_s), min(H, cy + half_s)
        
        crop_img = image[y1:y2, x1:x2]
        crop_m = bin_mask[y1:y2, x1:x2]
        
        if target_crop_size:
            crop_img = cv2.resize(crop_img, target_crop_size, interpolation=cv2.INTER_LANCZOS4)
            crop_m = cv2.resize(crop_m, target_crop_size, interpolation=cv2.INTER_NEAREST)

        return {
            "cropped_image": crop_img,
            "cropped_mask": crop_m,
            "full_image_with_bbox": image.copy(),
            "bbox": (x1, y1, x2, y2),
            "num_filaments": 0,
            "filaments_list": [],
            "selected_rank": 1,
            "has_filament": False,
        }

    # Clamp selected_rank
    selected_rank = max(1, min(selected_rank, len(valid_labels)))
    chosen_label = valid_labels[selected_rank - 1]

    bx = stats[chosen_label, cv2.CC_STAT_LEFT]
    by = stats[chosen_label, cv2.CC_STAT_TOP]
    bw = stats[chosen_label, cv2.CC_STAT_WIDTH]
    bh = stats[chosen_label, cv2.CC_STAT_HEIGHT]

    # Context padding for crop
    pad_x = max(15, int(bw * padding_fraction))
    pad_y = max(15, int(bh * padding_fraction))

    x1 = max(0, bx - pad_x)
    y1 = max(0, by - pad_y)
    x2 = min(W, bx + bw + pad_x)
    y2 = min(H, by + bh + pad_y)

    # Square aspect ratio for clean visual inspection
    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w > crop_h:
        diff = crop_w - crop_h
        y1 = max(0, y1 - diff // 2)
        y2 = min(H, y2 + (diff - diff // 2))
    elif crop_h > crop_w:
        diff = crop_h - crop_w
        x1 = max(0, x1 - diff // 2)
        x2 = min(W, x2 + (diff - diff // 2))

    cropped_image = image[y1:y2, x1:x2].copy()
    cropped_mask = bin_mask[y1:y2, x1:x2].copy()

    # Create overlay on full image showing ALL detected filament bounding boxes
    bbox_overlay = image.copy()
    if len(bbox_overlay.shape) == 2:
        bbox_overlay = cv2.cvtColor(bbox_overlay, cv2.COLOR_GRAY2BGR)

    box_colors = [
        (0, 255, 255),   # Cyan for Selected / Primary
        (0, 255, 0),     # Lime Green
        (255, 200, 0),   # Sky Blue / Amber
        (255, 0, 255),   # Magenta
        (0, 165, 255),   # Orange
        (50, 205, 50),   # Lime Green
        (200, 100, 255), # Lavender
        (255, 255, 0),   # Yellow
    ]

    filaments_list = []

    # Draw all detected filament bounding boxes
    for rank, l_idx in enumerate(valid_labels, 1):
        fx = stats[l_idx, cv2.CC_STAT_LEFT]
        fy = stats[l_idx, cv2.CC_STAT_TOP]
        fw = stats[l_idx, cv2.CC_STAT_WIDTH]
        fh = stats[l_idx, cv2.CC_STAT_HEIGHT]
        farea = int(stats[l_idx, cv2.CC_STAT_AREA])

        cx = int(centroids[l_idx, 0])
        cy = int(centroids[l_idx, 1])

        filaments_list.append({
            "rank": rank,
            "area_px": farea,
            "bbox": (fx, fy, fw, fh),
            "centroid_x": cx,
            "centroid_y": cy,
            "is_selected": (rank == selected_rank)
        })

        is_current = (rank == selected_rank)
        color = (0, 255, 255) if is_current else box_colors[(rank - 1) % len(box_colors)]
        thickness = 3 if is_current else 1

        cv2.rectangle(bbox_overlay, (fx, fy), (fx + fw, fy + fh), color, thickness)
        
        if is_current:
            tag = f"⭐ Filament #{rank} [ACTIVE] ({farea}px)"
        else:
            tag = f"Filament #{rank} ({farea}px)"

        # Text pill background
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        ty = max(th + 4, fy - 4)
        cv2.rectangle(bbox_overlay, (fx, ty - th - 3), (fx + tw + 4, ty + 2), (15, 15, 15), -1)
        cv2.putText(bbox_overlay, tag, (fx + 2, ty - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

    # Resize crop for standard output if requested
    if target_crop_size:
        disp_crop = cv2.resize(cropped_image, target_crop_size, interpolation=cv2.INTER_LANCZOS4)
        disp_mask = cv2.resize(cropped_mask, target_crop_size, interpolation=cv2.INTER_NEAREST)
    else:
        disp_crop = cropped_image
        disp_mask = cropped_mask

    return {
        "cropped_image": disp_crop,
        "cropped_mask": disp_mask,
        "raw_crop": cropped_image,
        "full_image_with_bbox": bbox_overlay,
        "bbox": (x1, y1, x2, y2),
        "num_filaments": len(valid_labels),
        "filaments_list": filaments_list,
        "selected_rank": selected_rank,
        "has_filament": True,
    }
