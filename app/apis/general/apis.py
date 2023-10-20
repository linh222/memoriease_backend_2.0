from functools import lru_cache

from fastapi import APIRouter, Depends, status
from fastapi.openapi.models import APIKey

from app import config
from app.api_key import get_api_key

router = APIRouter()


@lru_cache()
def get_settings():
    return config.Settings()


@router.get("/")
async def root():
    # Root path, show the API use for
    return {"message": "Image Retrieval Model"}


@router.get("/info")
async def info(settings: config.Settings = Depends(get_settings), api_key: APIKey = Depends(get_api_key)):
    # Information of the app
    return {
        "app_name": settings.APP_NAME,
        "model_version": settings.API_MODEL_VERSION,
    }


@router.get("/health", status_code=status.HTTP_200_OK, )
async def health_check():
    # Health check
    return {"status": "OK", "message": "I am healthy"}


def include_router(app):
    app.include_router(router)
