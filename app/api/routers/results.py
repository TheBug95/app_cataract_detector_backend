from fastapi import APIRouter, Response
import json
import base64
import io
import random
from PIL import Image
from typing import List

router = APIRouter(tags=["Inferencia"])

STORAGE_PATH = "app/storage.json"

def rotate_image_3x(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    for _ in range(3):
        img = img.transpose(Image.ROTATE_180)
    output = io.BytesIO()
    img.save(output, format="PNG")
    return output.getvalue()

@router.get("/results/")
async def get_results():
    import os
    import json

    storage_path = "app/storage.json"

    # Leer JSON
    if not os.path.exists(storage_path):
        return {"error": "Storage file not found."}

    with open(storage_path, "r") as f:
        content = f.read().strip()
        if not content:
            return {"error": "Storage is empty."}
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return {"error": "Invalid storage format."}

    if not data:
        return {"error": "No entries in storage."}

    last_entry = data[-1]
    image_path = last_entry["path"]

    # Simular resultado del modelo
    result = fake_predict(image_path)  # reemplazar por tu lógica real

    return {
        "filename": last_entry["filename"],
        "result": result
    }

# Función de prueba (puedes sustituirla por tu modelo real)
def fake_predict(image_path):
    return {
        "diagnosis": "cataract",
        "confidence": 0.92,
        "image_path": image_path
    }
