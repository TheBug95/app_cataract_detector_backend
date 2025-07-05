import numpy as np
from pycocotools import mask as mask_utils
from PIL import Image
import tempfile
import os
from .inference import catarata_mask_selector

def process_masks(prediction_result: dict, proto: dict, k: int, get_emb_func) -> list:
    """
    Procesa todas las máscaras del resultado de Vertex AI usando catarata_mask_selector
    
    Args:
        prediction_result: Resultado de Vertex AI (debe contener 'masks')
        proto: Diccionario PROTO_VIT con los prototipos KDE
        k: Índice del prototipo a usar (ej. 36)
        get_emb_func: Función get_emb_vit para generar embeddings
        
    Returns:
        Lista de resultados para cada máscara
    """
    results = []
    
    if not prediction_result.get("masks"):
        return results
    
    # Creamos una imagen temporal del tamaño de las máscaras
    h, w = prediction_result["masks"][0]["size"]
    dummy_img = Image.new("RGB", (w, h), color=(255, 255, 255))
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
        dummy_img.save(tmp_img.name)
        
        # Procesamos cada máscara individualmente
        for i, mask_data in enumerate(prediction_result["masks"]):
            # Creamos un generador de máscaras simple para esta máscara específica
            class SingleMaskGenerator:
                def generate(self, _):
                    binary_mask = mask_utils.decode({
                        "counts": mask_data['counts'].encode('utf-8'),
                        "size": mask_data['size']
                    })
                    return [{
                        "segmentation": binary_mask.astype(bool),
                        "area": np.sum(binary_mask),
                        "bbox": mask_utils.toBbox({
                            "counts": mask_data['counts'].encode('utf-8'),
                            "size": mask_data['size']
                        }).tolist()
                    }]
            
            # Llamamos a catarata_mask_selector con esta máscara
            label, mask_np = catarata_mask_selector(
                image_path=tmp_img.name,
                k=k,
                proto=proto,
                mask_generator=SingleMaskGenerator(),
                get_emb=get_emb_func,
                visualize=False
            )
            
            results.append({
                "mask_index": i,
                "label": label,
                "mask_size": mask_data["size"],
                "has_cataract": label == "Catarata",
                "mask_data": mask_data if label == "Catarata" else None
            })
        
        # Eliminamos la imagen temporal
        os.unlink(tmp_img.name)
    
    return results