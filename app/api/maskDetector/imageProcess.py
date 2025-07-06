#!/usr/bin/env python3
"""
SAM Mask Decoder - Herramienta independiente para probar la decodificación de máscaras RLE
"""

import numpy as np
from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt
from PIL import Image
import json
import argparse
import base64
import io

def decode_sam_rle(rle_data: dict) -> np.ndarray:
    """
    Decodifica una máscara RLE del formato SAM a una matriz binaria
    
    Args:
        rle_data: Diccionario con {'counts': str, 'size': [height, width]}
    
    Returns:
        Máscara binaria como array numpy (0=fondo, 1=objeto)
    """
    # Reconstruir el formato que pycocotools espera
    rle = {
        "counts": rle_data['counts'].encode('utf-8'),
        "size": rle_data['size']
    }
    return mask_utils.decode(rle)

def visualize_mask(mask: np.ndarray, title: str = "Máscara"):
    """Visualiza una máscara usando matplotlib"""
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, color: tuple = (255, 0, 0), alpha: float = 0.5):
    """
    Superpone una máscara sobre una imagen
    
    Args:
        image: Array numpy de la imagen (H,W,3)
        mask: Máscara binaria (H,W)
        color: Color RGB para la máscara
        alpha: Transparencia (0-1)
    """
    # Crear imagen de color para la máscara
    mask_color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_color[mask == 1] = color
    
    # Mezclar con la imagen original
    blended = (image * (1 - alpha) + mask_color * alpha).astype(np.uint8)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(blended)
    plt.axis('off')
    plt.show()

