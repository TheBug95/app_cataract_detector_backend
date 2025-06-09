from fastapi import APIRouter, UploadFile, File
import os
import json
from datetime import datetime

router = APIRouter(prefix="/predict", tags=["Inferencia"])

@router.post("/")
async def predict(image: UploadFile = File(...)):
    # Rutas
    images_dir = "app/images"
    storage_path = "app/storage.json"
    os.makedirs(images_dir, exist_ok=True)

    # Guardar imagen
    contents = await image.read()
    image_path = os.path.join(images_dir, image.filename)
    with open(image_path, "wb") as f:
        f.write(contents)

    # Leer storage.json como lista, forzadamente
    data = []
    if os.path.exists(storage_path):
        with open(storage_path, "r") as f:
            try:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    data = loaded
            except json.JSONDecodeError:
                pass  # archivo mal formado

    # Crear entrada
    new_entry = {
        "filename": image.filename,
        "path": image_path,
        "timestamp": datetime.utcnow().isoformat()
    }

    # Agregar y guardar
    data.append(new_entry)
    with open(storage_path, "w") as f:
        json.dump(data, f, indent=2)

    return {
        "message": "Image received and stored",
        "stored": new_entry
    }
