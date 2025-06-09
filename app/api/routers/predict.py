from fastapi import APIRouter, UploadFile, File, HTTPException
import os

router = APIRouter(prefix="/predict", tags=["Inferencia"])

@router.post("/")
async def upload_image(image: UploadFile = File(...)):
    # Validar tipo de archivo
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(415, "Solo se aceptan JPG/PNG")

    # Guardar imagen en la carpeta 'results'
    file_path = f"results/{image.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await image.read())

    return {"status": "Imagen recibida", "filename": image.filename}