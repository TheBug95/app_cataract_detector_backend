from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from .ai_rest import predict_vertex_ai_rest, preprocess_image
import os
import tempfile
from ..maskDetector.execution import process_masks
from ..maskDetector.config import PROTO_VIT, get_emb_vit


router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("/", response_class=JSONResponse)
async def predict(image: UploadFile = File(...)):
    """
    Process image through Vertex AI SAM model
    
    Args:
        image: Uploaded image file
        
    Returns:
        Processed mask data from SAM model
    """
    try:
        # Read and preprocess image
        contents = await image.read()
        try:
            processed_image = preprocess_image(contents)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Image processing error",
                    "details": str(e),
                    "suggestion": "Try with a JPG or non-transparent PNG image"
                }
            )
        
        # Use tempfile for processing (no permanent storage)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp:
            tmp.write(processed_image)
            tmp.flush()
            
            # Get predictions from Vertex AI
            prediction_result = predict_vertex_ai_rest(tmp.name)
            maskResult = resultados = process_masks(
                                    prediction_result=prediction_result,  # El JSON que viene de Vertex AI
                                    proto=PROTO_VIT,                     # Los prototipos cargados
                                    k=36,                                # Índice del prototipo
                                    get_emb_func=get_emb_vit             # Función para embeddings
                                )
            
            return {
                "status": "success",
                "filename": image.filename,
                "mask_data": prediction_result,
                "MASK Results": maskResult
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction error",
                "details": str(e)
            }
        )