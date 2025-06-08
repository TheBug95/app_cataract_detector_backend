# app/schemas.py  (nombre sugerido)
from pydantic import BaseModel
from typing import Dict, Any


"""class PredictResponse(BaseModel):
    cataract_detected: bool
    confidence: float
    reason: str
    mask_png_base64: str
    features: Dict[str, Any]

    class Config:
        schema_extra = {
            "example": {
                "cataract_detected": True,
                "confidence": 0.75,
                "reason": "Alta densidad de segmentación detectada",
                "mask_png_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
                "features": {
                    "activated_pixels": 1250,
                    "activation_ratio": 0.0625,
                    "num_contours": 3,
                    "max_contour_area": 800,
                    "total_pixels": 20000
                }
            }
        }"""

from pydantic import BaseModel
from typing import List

class SegmentResponse(BaseModel):
    masks: List[str]   # cada máscara PNG codificada en Base64

    class Config:
        schema_extra = {
            "example": {
                "masks": [
                    "iVBORw0KGgoAAAANSUhEUgAA...",
                    "iVBORw0KGgoAAAANSUhEUgAA..."
                ]
            }
        }
