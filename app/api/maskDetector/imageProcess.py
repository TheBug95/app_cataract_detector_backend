#!/usr/bin/env python3
"""
SAM Mask Decoder - Standalone tool for testing RLE mask decoding
"""

import numpy as np
from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt

def decode_sam_rle(rle_data: dict) -> np.ndarray:
    """
    Decodes an RLE mask from SAM format to a binary matrix
    
    Args:
        rle_data: Dictionary with {'counts': str, 'size': [height, width]}
    
    Returns:
        Binary mask as numpy array (0=background, 1=object)
    """
    # Reconstruct the format that pycocotools expects
    rle = {
        "counts": rle_data['counts'].encode('utf-8'),
        "size": rle_data['size']
    }
    return mask_utils.decode(rle)

def visualize_mask(mask: np.ndarray, title: str = "Mask"):
    """Visualizes a mask using matplotlib"""
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, color: tuple = (255, 0, 0), alpha: float = 0.5):
    """
    Overlays a mask on top of an image
    
    Args:
        image: Numpy array of the image (H,W,3)
        mask: Binary mask (H,W)
        color: RGB color for the mask
        alpha: Transparency (0-1)
    """
    # Create color image for the mask
    mask_color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_color[mask == 1] = color
    
    # Blend with the original image
    blended = (image * (1 - alpha) + mask_color * alpha).astype(np.uint8)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(blended)
    plt.axis('off')
    plt.show()