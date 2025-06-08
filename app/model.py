# model.py ── versión “ligera”, sin huggingface_hub
import os, requests, base64

HF_TOKEN  = os.getenv("HF_API_TOKEN")          # tu token
MODEL_ID  = os.getenv("SAM_MODEL_ID")
print(MODEL_ID)
API_URL   = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

HEADERS   = {
    "Authorization": f"Bearer {HF_TOKEN}",
    # fuerza a que espere a que el modelo despierte
    "X-Wait-For-Model": "true",
    "Content-Type": "application/json",
}

def segment_image(img_bytes: bytes) -> list[str]:
    """
    Envía la imagen al endpoint de HF y devuelve las máscaras PNG-Base64.
    """
    # 1. Codificamos la imagen a Base64 (sin saltos de línea)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    # 2. Construimos el JSON tal como lo espera la API
    payload = {"inputs": b64}

    # 3. Hacemos la petición
    resp = requests.post(API_URL, headers=HEADERS, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"HF {resp.status_code}: {resp.text}")

    return resp.json()
