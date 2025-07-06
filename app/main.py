from fastapi import FastAPI
from .api.routers import predict

app = FastAPI(
    title="Cataract Detection API",
    description="API for detecting cataracts in eye images.",
    version="1.0.0",
)

app.include_router(predict.router)  