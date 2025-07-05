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

def test_with_sample_data():
    """Prueba con datos de ejemplo del SAM"""
    # Ejemplo de respuesta del SAM (simplificado)
    sam_response = {
        "predictions": [{
            "masks_rle": [{
                "counts": "Raf4b0Y<8I5K7J5K4M2N3L3N3M2N3M2N2N1O2O1N3M2N101N2N2O1N2O0O2O1N101O0O2O001N101O1O0O2O00001O000O101O00001O0O101O000O2O0000WM]FV2c9jM]FV2c9jM]FV2c9iM^FW2a9jM_FV2a9kM^FU2b9d01N100000000O100O1000001N1000001N100O101N100O2O0O101N2O0O2N101N1O2N2N2N2N2O1N3L4M2N2N2O1N2M4L4L3M4L5K4L5I:Ee]Z4",
                "size": [415, 830]
            }]
        }]
    }
    
    print("=== Prueba con datos de ejemplo ===")
    mask = decode_sam_rle(sam_response['predictions'][0]['masks_rle'][0])
    print(f"Máscara decodificada - Dimensiones: {mask.shape}")
    print(f"Área segmentada: {np.sum(mask)} píxeles")
    
    # Visualización
    visualize_mask(mask, "Máscara Decodificada")

def test_with_image_file(image_path: str, rle_data: dict):
    """Prueba con una imagen real y su máscara RLE"""
    print(f"\n=== Prueba con imagen: {image_path} ===")
    
    # Cargar imagen
    image = np.array(Image.open(image_path).convert('RGB'))
    
    # Decodificar máscara
    mask = decode_sam_rle(rle_data)
    
    print(f"Tamaño imagen: {image.shape}")
    print(f"Tamaño máscara: {mask.shape}")
    print(f"Área segmentada: {np.sum(mask)} píxeles ({np.sum(mask)/np.prod(mask.shape):.1%} de la imagen)")
    
    # Visualización
    overlay_mask_on_image(image, mask, color=(0, 255, 0), alpha=0.4)

def main():
    parser = argparse.ArgumentParser(description='Decodificador de Máscaras RLE de SAM')
    parser.add_argument('--test', action='store_true', help='Ejecutar prueba con datos de ejemplo')
    parser.add_argument('--image', type=str, help='Ruta a imagen para pruebas')
    parser.add_argument('--rle', type=str, help='Cadena JSON con datos RLE (debe incluir counts y size)')
    
    args = parser.parse_args()
    
    if args.test:
        test_with_sample_data()
    elif args.image and args.rle:
        try:
            rle_data = json.loads(args.rle)
            if 'counts' not in rle_data or 'size' not in rle_data:
                raise ValueError("El JSON RLE debe contener 'counts' y 'size'")
            test_with_image_file(args.image, rle_data)
        except json.JSONDecodeError:
            print("Error: El argumento --rle debe ser un JSON válido")
    else:
        print("Modo de uso:")
        print("1. Prueba básica: python sam_mask_decoder.py --test")
        print("2. Prueba con imagen: python sam_mask_decoder.py --image imagen.jpg --rle '{\"counts\":\"...\",\"size\":[h,w]}'")

if __name__ == "__main__":
    main()