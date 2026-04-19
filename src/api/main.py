from fastapi import FastAPI
import uvicorn

from src.api.openapi_utils import apply_custom_openapi
from src.api.retrain_api import retrain_router
from src.api.routers.health import router as health_router
from src.api.routers.ppe import router as ppe_router
from src.api.routers.scoring import router as scoring_router
from src.config.settings import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title="CleanOps AI Service",
        description="Unified AI service for scoring and PPE inference.",
        version="1.0.0",
        openapi_tags=[
            {
                "name": "health",
                "description": "Liveness and readiness endpoints for the unified AI service.",
            },
            {
                "name": "scoring",
                "description": "Cleaning scoring inference and visualization endpoints.",
            },
            {
                "name": "ppe",
                "description": "PPE inference and validation endpoints.",
            },
            {
                "name": "retrain",
                "description": "Model retraining endpoints.",
            },
        ],
    )
    apply_custom_openapi(app)
    app.include_router(health_router)
    app.include_router(scoring_router)
    app.include_router(ppe_router)
    app.include_router(retrain_router)
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_reload,
    )
