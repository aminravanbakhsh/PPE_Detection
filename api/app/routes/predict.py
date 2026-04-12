import io
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from PIL import Image

from ..auth import get_current_user
from ..config import settings
from ..detector import detector
from ..rate_limit import limiter
from ..schemas import BatchResult, ImageResult

router = APIRouter(prefix="/predict", tags=["predict"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_BYTES = settings.max_file_size_mb * 1024 * 1024


async def _validate_image(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            detail=f"Unsupported file type: {file.content_type}. Allowed: JPEG, PNG")

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=f"File too large ({len(data)} bytes). Max: {MAX_BYTES} bytes")

    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Invalid or corrupted image file")
    return data


@router.post("/image", response_model=ImageResult)
@limiter.limit(settings.rate_limit)
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    _user: str = Depends(get_current_user),
):
    if not detector.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Inference model is not loaded. Place your YOLO weights at MODEL_PATH "
                f"(currently {settings.model_path!r}) and restart the API. "
                "See README (API / Docker) for mount paths."
            ),
        )
    data = await _validate_image(file)
    image_id = str(uuid.uuid4())
    return detector.predict(data, image_id)


@router.post("/batch", response_model=BatchResult)
@limiter.limit(settings.rate_limit)
async def predict_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    _user: str = Depends(get_current_user),
):
    if not detector.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Inference model is not loaded. Place your YOLO weights at MODEL_PATH "
                f"(currently {settings.model_path!r}) and restart the API. "
                "See README (API / Docker) for mount paths."
            ),
        )
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided")
    if len(files) > settings.max_batch_images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files ({len(files)}). Maximum is {settings.max_batch_images}.",
        )

    results = []
    total_ms = 0.0
    for f in files:
        data = await _validate_image(f)
        image_id = str(uuid.uuid4())
        result = detector.predict(data, image_id)
        results.append(result)
        total_ms += result.inference_time_ms

    return BatchResult(results=results, total_inference_time_ms=round(total_ms, 2))
