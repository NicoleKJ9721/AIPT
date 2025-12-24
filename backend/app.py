from fastapi import FastAPI

from backend.datasets.routes import router as datasets_router
from backend.training.routes import router as training_router

app = FastAPI(title="AI Industrial Inspection Platform")


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


app.include_router(datasets_router, prefix="/datasets", tags=["datasets"])
app.include_router(training_router, prefix="/training", tags=["training"])
