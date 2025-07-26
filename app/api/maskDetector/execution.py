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

# Basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_masks(
    maskData: List[Dict[str, Any]],
    processed_image: bytes,
    k: int = 36,
    visualize: bool = True
) -> Tuple[str, Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Processes binary masks in SAM format applying exactly the same
    processing as catarata_mask_selector
    
    Args:
        maskData: List of dictionaries with:
            - counts: string with RLE data
            - size: list with [height, width]
        processed_image: Processed image in bytes (result of preprocess_image)
        k: Prototype index (default 36)
        visualize: Show visualizations during processing
    
    Returns:
        Tuple with (label, binary mask, original mask data)
        - label: "Cataract" or "No Cataract"
        - binary mask: numpy array or None
        - original data: dict with counts and size of the best mask, or None
    """
    # Initial log
    logger.info(f"=== STARTING PROCESSING OF {len(maskData)} MASKS ===")
    logger.info(f"Parameter k: {k}")
    
    # 1) Load processed image from bytes
    try:
        img_pil: Image.Image = Image.open(io.BytesIO(processed_image)).convert("RGB")
        img_np: np.ndarray = np.asarray(img_pil)
        logger.info(f"Image loaded successfully. Dimensions: {img_np.shape}")
        logger.info(f"Image size: {img_pil.size} (W x H)")
    except Exception as e:
        logger.error(f"Error loading image from bytes: {str(e)}")
        return "No Cataract", None, None
    
    # 2) Decode all RLE masks
    logger.info("=== DECODING RLE MASKS ===")
    decoded_masks = []
    original_masks = []  # Store references to original masks
    for idx, m in enumerate(maskData):
        try:
            logger.debug(f"Processing mask {idx+1}/{len(maskData)}")
            
            # Verify we have the necessary data
            if not all(key in m for key in ['counts', 'size']):
                logger.warning(f"Mask {idx+1} does not have required fields (counts, size)")
                continue
                
            # Decode RLE mask
            seg = decode_sam_rle(m)
            logger.debug(f"Mask {idx+1} decoded. Shape: {seg.shape}")
            
            # Verify dimensions match the image
            if seg.shape != img_np.shape[:2]:
                logger.warning(f"Mask {idx+1} has dimensions {seg.shape} vs image {img_np.shape[:2]}")
                continue
            
            # Only add non-empty masks
            if np.any(seg):
                pixel_count = np.sum(seg)
                decoded_masks.append({
                    'segmentation': seg,
                    'height': m['size'][0],
                    'width': m['size'][1],
                    'pixel_count': pixel_count
                })
                original_masks.append(m)  # Store original mask
                logger.info(f"Mask {idx+1} added successfully. Pixels: {pixel_count}")
            else:
                logger.warning(f"Mask {idx+1} is empty (all zeros)")
                
        except Exception as e:
            logger.error(f"Error decoding mask {idx+1}: {str(e)}")
            continue

    if not decoded_masks:
        logger.error("No valid masks to process")
        return "No Cataract", None, None

    logger.info(f"=== {len(decoded_masks)} MASKS DECODED SUCCESSFULLY ===")

    # 3) Retrieve KDE statistics for this *k* (EXACTLY THE SAME AS THE ORIGINAL FUNCTION)
    proto = PROTO_VIT
    get_emb = get_emb_vit
    kdes = proto[k]["kdes"]
    t_min = proto[k]["theta_min"]
    t_max = proto[k]["theta_max"]
    
    logger.info(f"=== MODEL CONFIGURATION ===")
    logger.info(f"Using k={k}")
    logger.info(f"θ_min={t_min:.4f}")
    logger.info(f"θ_max={t_max:.4f}")
    logger.info(f"Number of KDEs: {len(kdes)}")

    scores: List[float] = []
    preds: List[int] = []

    # 4) Evaluate each mask (EXACTLY THE SAME AS THE ORIGINAL FUNCTION)
    logger.info("=== EVALUATING MASKS ===")
    for i, m in enumerate(decoded_masks):
        try:
            logger.info(f"--- Processing mask {i+1}/{len(decoded_masks)} ---")
            
            seg = m['segmentation']
            ys, xs = np.where(seg)
            
            if len(xs) == 0:
                logger.warning(f"Mask {i+1} is empty - skipping")
                continue
                
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            
            logger.info(f"Mask {i+1} - Bounding box: x[{x0}-{x1}], y[{y0}-{y1}]")
            logger.info(f"Mask {i+1} - Bbox dimensions: {x1-x0+1} x {y1-y0+1}")
            logger.info(f"Mask {i+1} - Active pixels: {m['pixel_count']}")

            # Crop region of interest FROM THE ORIGINAL IMAGE
            crop = img_pil.crop((x0, y0, x1 + 1, y1 + 1))
            logger.debug(f"Mask {i+1} - Crop created. Size: {crop.size}")
            
            # Get embedding
            emb = get_emb(crop)
            logger.debug(f"Mask {i+1} - Embedding obtained. Shape: {emb.shape}")
            
            # 4.1) Log-score using the KDE of *each* dimension
            logp = sum(kde.score_samples([[emb[d]]])[0] for d, kde in enumerate(kdes))
            scores.append(logp)
            
            # 4.2) Binary classification: 1 = within range (cataract)
            pred = int(t_min <= logp <= t_max)
            preds.append(pred)
            
            # Detailed result logging
            logger.info(f"Mask {i+1} - Log-probability: {logp:.4f}")
            logger.info(f"Mask {i+1} - Prediction: {'CATARACT' if pred else 'NO CATARACT'}")
            logger.info(f"Mask {i+1} - In range [{t_min:.4f}, {t_max:.4f}]: {pred == 1}")
            
            # 4.3) Individual visualization (DURING PROCESSING)
            if visualize:
                logger.debug(f"Visualizing mask {i+1}")
                plt.figure(figsize=(12, 5))
                
                # Subplot 1: Original image with mask overlay
                plt.subplot(1, 2, 1)
                plt.imshow(img_np)
                
                # Create mask polygon
                mask_points = np.column_stack([xs, ys])
                if len(mask_points) > 0:
                    # Take only some points for the polygon (to avoid overload)
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
                
                # Text with information
                plt.text(
                    x0,
                    y0 - 10,
                    f"Mask #{i+1}\nlogp = {logp:.3f}\n→ {'CAT' if pred else 'NO'}",
                    color="white",
                    fontsize=10,
                    bbox=dict(facecolor="red" if pred else "blue", alpha=0.7, pad=4),
                )
                plt.axis("off")
                plt.title(f"Mask {i+1}/{len(decoded_masks)} - {'CATARACT' if pred else 'NO CATARACT'}")
                
                # Subplot 2: Region crop
                plt.subplot(1, 2, 2)
                plt.imshow(crop)
                plt.title(f"Crop - {crop.size[0]}x{crop.size[1]}px")
                plt.axis("off")
                
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            logger.error(f"Error processing mask {i+1}: {str(e)}")
            continue

    # 5) Select the best mask among positives (EXACTLY THE SAME AS THE ORIGINAL FUNCTION)
    logger.info("=== BEST MASK SELECTION ===")
    positives = [i for i, p in enumerate(preds) if p == 1]
    
    logger.info(f"Positive masks found: {len(positives)} out of {len(preds)}")
    
    if positives:
        # Among positives, the one with highest log-probability
        best_idx = max(positives, key=lambda i: scores[i])
        result_label = "Cataract"
        
        logger.info(f"=== FINAL RESULT: CATARACT DETECTED ===")
        logger.info(f"Best mask: #{best_idx+1}")
        logger.info(f"Log-probability of the best: {scores[best_idx]:.4f}")
        logger.info(f"Scores of all positive masks:")
        for pos_idx in positives:
            logger.info(f"  Mask #{pos_idx+1}: {scores[pos_idx]:.4f}")
            
    else:
        # No mask was considered cataract
        logger.info("=== FINAL RESULT: NO CATARACT ===")
        logger.info("No mask fell within the specified range")
        logger.info("Scores of all masks:")
        for idx, score in enumerate(scores):
            logger.info(f"  Mask #{idx+1}: {score:.4f} (out of range)")
        
        if visualize:
            print("No Cataract — no mask fell within the specified range.")
        return "No Cataract", None, None

    # 6) Extract binary mask from the best selection (EXACTLY THE SAME AS THE ORIGINAL FUNCTION)
    best_mask_np: np.ndarray = decoded_masks[best_idx]["segmentation"].astype(np.uint8)
    best_mask_original: Dict[str, Any] = original_masks[best_idx]
    best_mask_score = scores[best_idx]

    logger.info(f"Final mask extracted. Shape: {best_mask_np.shape}")
    logger.info(f"Active pixels in final mask: {np.sum(best_mask_np)}")
    logger.info(f"Original data of the best mask: counts length={len(best_mask_original['counts'])}, size={best_mask_original['size']}")
    logger.info(f"Score NP: {scores[best_idx]:.4f}")
    logger.info("=== PROCESSING COMPLETED SUCCESSFULLY ===")

    return result_label, best_mask_np, best_mask_original, best_mask_score