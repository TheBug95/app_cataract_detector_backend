import imghdr
from dotenv import load_dotenv; load_dotenv()
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from .model import segment_image
from .schemas import SegmentResponse

app = FastAPI(
    title="Cataract Detection API",
    description=(
        "Servicio REST que recibe una **imagen** (JPG/PNG) de un ojo y "
        "devuelve si se detectó catarata junto con la máscara de segmentación "
        "codificada en Base64.\n\n"
        "Powered by Segment Anything (Hugging Face) + IrisScience (Few-Shot Densities)."
    ),
    version="1.0.0",
    contact={
        "name": "Team I+D Iriscience",
        "email": "soporte@iriscience.com",
    },
    docs_url="/docs",         # Swagger
    redoc_url="/redoc",       # ReDoc
    openapi_url="/openapi.json",
)

# ── Endpoint principal ──────────────────────────────────
@app.post("/predict",
    summary="Analizar imagen de ojo para detectar si presenta catarata",
    response_model=SegmentResponse,
    #status_code=status.HTTP_200_OK,
    tags=["Inferencia"]
)
async def predict(image: UploadFile = File(..., description="Archivo JPG/PNG")):
    """
        Sube una **imagen** de un ojo y recibe el diagnóstico de catarata.

        * **`cataract_detected`** — `true` si se detecta catarata
        * **`mask_png_base64`** — máscara de la catarata (PNG Base64)
    """
    print("DEBUG: Content type:", image.content_type)
    if image.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=415, detail="Solo se acepta JPG/PNG")

    img_bytes = await image.read()
    print("Tamaño imagen:", len(img_bytes))
    try:
        print("Formato detectado:", imghdr.what(None, h=img_bytes))

        masks = segment_image(img_bytes)
        return {"masks": masks}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error HuggingFace: {e}")
