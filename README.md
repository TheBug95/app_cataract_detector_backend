# app_cataract_detector_backend
Backend de la aplicacion de deteccion de catarata

para crear el entorno virtual:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


para windows:

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt



para arrancar la aplicacion use:
``uvicorn app.main:app --reload``


comando para arrancar la aplicacion y usarla como host local de red:

uvicorn app.main:app --host 0.0.0.0 --port 8000


recordar que se debe de actualizar la direccion ip:

http://IP:8000/predict/ a donde IP es el nuevo valor de la direccion IP


para comprobar que la aplicacion sirve:
http://127.0.0.1:8000/docs











def build_vertex_ai_endpoint(project_id: str, endpoint_id: str, region: str = "us-central1") -> str:
    """
    Construye la URL completa para un endpoint de Vertex AI
    
    Args:
        project_id: ID numérico del proyecto (ej: '202589491823')
        endpoint_id: ID del endpoint (ej: '6399717325074857984')
        region: Región de despliegue (default: 'us-central1')
    
    Returns:
        URL completa del endpoint
    """
    base_url = (
        f"https://{endpoint_id}.{region}-{project_id}.prediction.vertexai.goog/"
        f"v1/projects/{project_id}/locations/{region}/endpoints/{endpoint_id}:predict"
    )
    return base_url


# Configuración de tu endpoint
PROJECT_ID = "202589491823"
ENDPOINT_ID = "6399717325074857984"
REGION = "us-central1"

# Construir URL
endpoint_url = build_vertex_ai_endpoint(PROJECT_ID, ENDPOINT_ID, REGION)
print(endpoint_url)