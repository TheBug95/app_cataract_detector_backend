from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api.routers import predict, results

app = FastAPI(
    title="Cataract Detection API",
    description="API para detectar cataratas en imágenes de ojos.",
    version="1.0.0",
)

# Incluye las rutas
app.include_router(predict.router)
app.include_router(results.router)

# Sirve imágenes estáticas desde la carpeta 'results'
app.mount("/results", StaticFiles(directory="results"), name="results")