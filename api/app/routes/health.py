from fastapi import APIRouter

from ..detector import detector
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="healthy", model_loaded=detector.is_loaded)
