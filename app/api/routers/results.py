from fastapi import APIRouter
import os

router = APIRouter(prefix="/results", tags=["Resultados"])

@router.get("/")
async def get_results():
    # Datos de inferencia simulados
    inference_data = {
        "probability": 0.85,
        "success_rate": 0.92,
    }

    # Listar imágenes en la carpeta 'results'
    images = []
    for filename in os.listdir("results"):
        if filename.lower().endswith((".jpg", ".png")):
            images.append({
                "url": f"/results/{filename}",
                "name": filename
            })

    return {
        "inference": inference_data,
        "images": images[:3]  # Devuelve solo las 3 primeras imágenes
    }