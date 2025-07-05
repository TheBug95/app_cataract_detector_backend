import requests
import base64
import json
import google.auth
from google.auth.transport.requests import Request
from PIL import Image
import io

def predict_vertex_ai_rest(image_path: str):
    # Obtiene credenciales
    credentials, _ = google.auth.default()
    credentials.refresh(Request())
    token = credentials.token

    # Cargar y preprocesar la imagen
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # Convertir a base64
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Construir instancia según lo que el modelo SAM espera
    instance = {"image": image_b64}

    body = {
        "instances": [instance]
    }

    # URL CORREGIDA - Usa el dominio dedicado
    url = (
        "https://6399717325074857984.us-central1-202589491823.prediction.vertexai.goog/"
        "v1/projects/202589491823/locations/us-central1/endpoints/6399717325074857984:predict"
    )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=body)

    if response.status_code != 200:
        error_msg = response.text
        try:
            error_json = response.json()
            error_msg = json.dumps(error_json, indent=2)
        except:
            pass
        raise Exception(f"Error {response.status_code}: {error_msg}")

    return response.json()

def preprocess_image(image_bytes: bytes, max_size: int = 1024) -> bytes:
    """Redimensiona la imagen si es muy grande manteniendo el aspect ratio"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convertir a RGB si la imagen tiene canal alfa o es paletizada
        if img.mode != 'RGB':
            if img.mode == 'RGBA':
                # Crear fondo blanco para imágenes con transparencia
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])  # Usar canal alfa como máscara
                img = background
            else:
                img = img.convert('RGB')
        
        # Redimensionar si es necesario
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        
        # Convertir a JPEG
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=85)
        return output.getvalue()
    
    except Exception as e:
        raise ValueError(f"Error procesando imagen: {str(e)}")