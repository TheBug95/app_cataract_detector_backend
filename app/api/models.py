from pydantic import BaseModel
from typing import List, Dict

class SegmentResponse(BaseModel):
    masks: List[str]  # Máscaras en Base64

class InferenceData(BaseModel):
    probability: float
    success_rate: float

class ResultsResponse(BaseModel):
    inference: InferenceData
    images: List[Dict[str, str]]