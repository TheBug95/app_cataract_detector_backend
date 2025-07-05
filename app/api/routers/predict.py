from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import json
from datetime import datetime
from .ai_rest import predict_vertex_ai_rest, preprocess_image
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/predict", tags=["Inferencia"])

images_dir = "app/images"
storage_path = "app/storage.json"
os.makedirs(images_dir, exist_ok=True)

@router.post("/", response_class=JSONResponse)
async def predict(image: UploadFile = File(...)):
    try:
        # Leer y preprocesar la imagen
        contents = await image.read()
        try:
            processed_image = preprocess_image(contents)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Error procesando imagen",
                    "details": str(e),
                    "suggestion": "Intente con una imagen en formato JPG o PNG sin transparencia"
                }
            )
        
        # Guardar la imagen (opcional, para registro)
        image_path = os.path.join(images_dir, image.filename)
        with open(image_path, "wb") as f:
            f.write(processed_image)

        # Guardar registro en storage.json
        data = []
        if os.path.exists(storage_path):
            try:
                with open(storage_path, "r") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        data = loaded
            except (json.JSONDecodeError, FileNotFoundError):
                pass

        new_entry = {
            "filename": image.filename,
            "path": image_path,
            "timestamp": datetime.utcnow().isoformat(),
            "size": len(processed_image)
        }

        data.append(new_entry)
        with open(storage_path, "w") as f:
            json.dump(data, f, indent=2)

        # Llamar a Vertex AI
        prediction_result = predict_vertex_ai_rest(image_path)

        return {
            "status": "success",
            "filename": image.filename,
            "vertex_ai_response": prediction_result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Error processing image",
                "details": str(e)
            }
        )
