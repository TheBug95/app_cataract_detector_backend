from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .api.routers import predict, results
import os
import json

# Limpia la carpeta 'results' al iniciar
RESULTS_DIR = "app/images"
STORAGE_PATH = "app/storage.json"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
else:
    for filename in os.listdir(RESULTS_DIR):
        file_path = os.path.join(RESULTS_DIR, filename)
        os.remove(file_path)

# Inicializa el archivo storage.json si no existe
if not os.path.exists(STORAGE_PATH):
    with open(STORAGE_PATH, "w") as f:
        json.dump({}, f)

app = FastAPI(
    title="Cataract Detection API",
    description="API para detectar cataratas en imágenes de ojos.",
    version="1.0.0",
)
app.mount("/images", StaticFiles(directory="app/images"), name="images")
app.include_router(predict.router)
app.include_router(results.router)


