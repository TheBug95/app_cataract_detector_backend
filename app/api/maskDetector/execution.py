import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Optional, Dict, Any, List
import logging
import io
from pycocotools import mask as mask_utils
from .config import PROTO_VIT, get_emb_vit
from .imageProcess import decode_sam_rle

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_masks(
    maskData: List[Dict[str, Any]],
    processed_image: bytes,
    k: int = 36,
    visualize: bool = True
) -> Tuple[str, Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Procesa máscaras binarias en formato SAM aplicando exactamente el mismo
    procesamiento que catarata_mask_selector
    
    Args:
        maskData: Lista de diccionarios con:
            - counts: string con datos RLE
            - size: lista con [height, width]
        processed_image: Imagen procesada en bytes (resultado de preprocess_image)
        k: Índice del prototipo (por defecto 36)
        visualize: Mostrar visualizaciones durante el procesamiento
    
    Returns:
        Tuple con (etiqueta, máscara binaria, datos originales de la máscara)
        - etiqueta: "Catarata" o "No Catarata"
        - máscara binaria: array numpy o None
        - datos originales: dict con counts y size de la mejor máscara, o None
    """
    # Log inicial
    logger.info(f"=== INICIANDO PROCESAMIENTO DE {len(maskData)} MÁSCARAS ===")
    logger.info(f"Parámetro k: {k}")
    
    # 1) Cargar imagen procesada desde bytes
    try:
        img_pil: Image.Image = Image.open(io.BytesIO(processed_image)).convert("RGB")
        img_np: np.ndarray = np.asarray(img_pil)
        logger.info(f"Imagen cargada exitosamente. Dimensiones: {img_np.shape}")
        logger.info(f"Tamaño de imagen: {img_pil.size} (W x H)")
    except Exception as e:
        logger.error(f"Error cargando imagen desde bytes: {str(e)}")
        return "No Catarata", None
    
    # 2) Decodificar todas las máscaras RLE
    logger.info("=== DECODIFICANDO MÁSCARAS RLE ===")
    decoded_masks = []
    original_masks = []  # Guardar referencias a las máscaras originales
    for idx, m in enumerate(maskData):
        try:
            logger.debug(f"Procesando máscara {idx+1}/{len(maskData)}")
            
            # Verificar que tenemos los datos necesarios
            if not all(key in m for key in ['counts', 'size']):
                logger.warning(f"Máscara {idx+1} no tiene los campos requeridos (counts, size)")
                continue
                
            # Decodificar la máscara RLE
            seg = decode_sam_rle(m)
            logger.debug(f"Máscara {idx+1} decodificada. Shape: {seg.shape}")
            
            # Verificar que las dimensiones coinciden con la imagen
            if seg.shape != img_np.shape[:2]:
                logger.warning(f"Máscara {idx+1} tiene dimensiones {seg.shape} vs imagen {img_np.shape[:2]}")
                continue
            
            # Solo agregar máscaras no vacías
            if np.any(seg):
                pixel_count = np.sum(seg)
                decoded_masks.append({
                    'segmentation': seg,
                    'height': m['size'][0],
                    'width': m['size'][1],
                    'pixel_count': pixel_count
                })
                original_masks.append(m)  # Guardar la máscara original
                logger.info(f"Máscara {idx+1} agregada exitosamente. Píxeles: {pixel_count}")
            else:
                logger.warning(f"Máscara {idx+1} está vacía (todos ceros)")
                
        except Exception as e:
            logger.error(f"Error decodificando máscara {idx+1}: {str(e)}")
            continue

    if not decoded_masks:
        logger.error("No hay máscaras válidas para procesar")
        return "No Catarata", None, None

    logger.info(f"=== {len(decoded_masks)} MÁSCARAS DECODIFICADAS EXITOSAMENTE ===")

    # 3) Recupera estadísticos de KDE para este *k* (EXACTAMENTE IGUAL QUE LA FUNCIÓN ORIGINAL)
    proto = PROTO_VIT
    get_emb = get_emb_vit
    kdes = proto[k]["kdes"]
    t_min = proto[k]["theta_min"]
    t_max = proto[k]["theta_max"]
    
    logger.info(f"=== CONFIGURACIÓN DEL MODELO ===")
    logger.info(f"Usando k={k}")
    logger.info(f"θ_min={t_min:.4f}")
    logger.info(f"θ_max={t_max:.4f}")
    logger.info(f"Número de KDEs: {len(kdes)}")

    scores: List[float] = []
    preds: List[int] = []

    # 4) Evalúa cada máscara (EXACTAMENTE IGUAL QUE LA FUNCIÓN ORIGINAL)
    logger.info("=== EVALUANDO MÁSCARAS ===")
    for i, m in enumerate(decoded_masks):
        try:
            logger.info(f"--- Procesando máscara {i+1}/{len(decoded_masks)} ---")
            
            seg = m['segmentation']
            ys, xs = np.where(seg)
            
            if len(xs) == 0:
                logger.warning(f"Máscara {i+1} está vacía - saltando")
                continue
                
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            
            logger.info(f"Máscara {i+1} - Bounding box: x[{x0}-{x1}], y[{y0}-{y1}]")
            logger.info(f"Máscara {i+1} - Dimensiones bbox: {x1-x0+1} x {y1-y0+1}")
            logger.info(f"Máscara {i+1} - Píxeles activos: {m['pixel_count']}")

            # Recortar región de interés DE LA IMAGEN ORIGINAL
            crop = img_pil.crop((x0, y0, x1 + 1, y1 + 1))
            logger.debug(f"Máscara {i+1} - Crop creado. Tamaño: {crop.size}")
            
            # Obtener embedding
            emb = get_emb(crop)
            logger.debug(f"Máscara {i+1} - Embedding obtenido. Shape: {emb.shape}")
            
            # 4.1) Log‑score usando el KDE de *cada* dimensión
            logp = sum(kde.score_samples([[emb[d]]])[0] for d, kde in enumerate(kdes))
            scores.append(logp)
            
            # 4.2) Clasificación binaria: 1 = dentro del rango (catarata)
            pred = int(t_min <= logp <= t_max)
            preds.append(pred)
            
            # Log detallado del resultado
            logger.info(f"Máscara {i+1} - Log-probabilidad: {logp:.4f}")
            logger.info(f"Máscara {i+1} - Predicción: {'CATARATA' if pred else 'NO CATARATA'}")
            logger.info(f"Máscara {i+1} - En rango [{t_min:.4f}, {t_max:.4f}]: {pred == 1}")
            
            # 4.3) Visualización individual (DURANTE EL PROCESAMIENTO)
            if visualize:
                logger.debug(f"Visualizando máscara {i+1}")
                plt.figure(figsize=(12, 5))
                
                # Subplot 1: Imagen original con máscara superpuesta
                plt.subplot(1, 2, 1)
                plt.imshow(img_np)
                
                # Crear polígono de la máscara
                mask_points = np.column_stack([xs, ys])
                if len(mask_points) > 0:
                    # Tomar solo algunos puntos para el polígono (para evitar sobrecarga)
                    step = max(1, len(mask_points) // 100)
                    mask_points_sampled = mask_points[::step]
                    
                    poly = patches.Polygon(
                        mask_points_sampled,
                        closed=True,
                        facecolor=(0.8, 0.8, 0.8, 0.4),
                        edgecolor="red" if pred else "blue",
                        linewidth=2
                    )
                    plt.gca().add_patch(poly)
                
                # Texto con información
                plt.text(
                    x0,
                    y0 - 10,
                    f"Máscara #{i+1}\nlogp = {logp:.3f}\n→ {'CAT' if pred else 'NO'}",
                    color="white",
                    fontsize=10,
                    bbox=dict(facecolor="red" if pred else "blue", alpha=0.7, pad=4),
                )
                plt.axis("off")
                plt.title(f"Máscara {i+1}/{len(decoded_masks)} - {'CATARATA' if pred else 'NO CATARATA'}")
                
                # Subplot 2: Crop de la región
                plt.subplot(1, 2, 2)
                plt.imshow(crop)
                plt.title(f"Crop - {crop.size[0]}x{crop.size[1]}px")
                plt.axis("off")
                
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            logger.error(f"Error procesando máscara {i+1}: {str(e)}")
            continue

    # 5) Selecciona la mejor máscara dentro de los positivos (EXACTAMENTE IGUAL QUE LA FUNCIÓN ORIGINAL)
    logger.info("=== SELECCIÓN DE MEJOR MÁSCARA ===")
    positives = [i for i, p in enumerate(preds) if p == 1]
    
    logger.info(f"Máscaras positivas encontradas: {len(positives)} de {len(preds)}")
    
    if positives:
        # Entre las positivas, la de mayor log‑probabilidad
        best_idx = max(positives, key=lambda i: scores[i])
        result_label = "Catarata"
        
        logger.info(f"=== RESULTADO FINAL: CATARATA DETECTADA ===")
        logger.info(f"Mejor máscara: #{best_idx+1}")
        logger.info(f"Log-probabilidad de la mejor: {scores[best_idx]:.4f}")
        logger.info(f"Scores de todas las máscaras positivas:")
        for pos_idx in positives:
            logger.info(f"  Máscara #{pos_idx+1}: {scores[pos_idx]:.4f}")
            
    else:
        # Ninguna máscara fue considerada catarata
        logger.info("=== RESULTADO FINAL: NO CATARATA ===")
        logger.info("Ninguna máscara cayó dentro del rango especificado")
        logger.info("Scores de todas las máscaras:")
        for idx, score in enumerate(scores):
            logger.info(f"  Máscara #{idx+1}: {score:.4f} (fuera de rango)")
        
        if visualize:
            print("No Catarata — ninguna máscara cayó dentro del rango especificado.")
        return "No Catarata", None, None

    # 6) Extrae máscara binaria de la mejor selección (EXACTAMENTE IGUAL QUE LA FUNCIÓN ORIGINAL)
    best_mask_np: np.ndarray = decoded_masks[best_idx]["segmentation"].astype(np.uint8)
    best_mask_original: Dict[str, Any] = original_masks[best_idx]
    
    logger.info(f"Máscara final extraída. Shape: {best_mask_np.shape}")
    logger.info(f"Píxeles activos en máscara final: {np.sum(best_mask_np)}")
    logger.info(f"Datos originales de la mejor máscara: counts length={len(best_mask_original['counts'])}, size={best_mask_original['size']}")
    logger.info("=== PROCESAMIENTO COMPLETADO EXITOSAMENTE ===")

    return result_label, best_mask_np, best_mask_original