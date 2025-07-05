from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from .ai_rest import predict_vertex_ai_rest, preprocess_image
import os
import tempfile

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
            
            return {
                "status": "success",
                "filename": image.filename,
                "mask_data": prediction_result
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction error",
                "details": str(e)
            }
        )