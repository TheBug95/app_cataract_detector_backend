from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from .ai_rest import predict_vertex_ai_rest, preprocess_image
import os
import tempfile
from ..maskDetector.execution import process_masks


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
            APIresults = predict_vertex_ai_rest(tmp.name)
            APIprocessed_results = [{"counts": mask["counts"], "size": mask["size"]} for mask in APIresults["masks"]]

            # Process the masks
            result_label, best_mask_np, best_mask_original, best_score = process_masks(APIprocessed_results, processed_image)

            # Prepare response according to the result
            if result_label == "Cataract" and best_mask_original is not None:
                return {
                    "status": "success",
                    "result": {
                        "prediction": "Cataract",
                        "score NP": best_score, 
                        "best_mask": {
                            "counts": best_mask_original["counts"],
                            "size": best_mask_original["size"]
                        },
                        "total_masks_processed": len(APIprocessed_results)
                    }
                }
            else:
                return {
                    "status": "success",
                    "result": {
                        "prediction": "No Cataract",
                        "message": "No cataracts found in the image",
                        "total_masks_processed": len(APIprocessed_results)
                    }
                }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction error",
                "details": str(e)
            }
        )