import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from PIL import Image
import io

from ..auth import get_current_user
from ..config import settings
from ..detector import detector
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
async def predict_image(file: UploadFile = File(...), _user: str = Depends(get_current_user)):
    data = await _validate_image(file)
    image_id = str(uuid.uuid4())
    return detector.predict(data, image_id)


@router.post("/batch", response_model=BatchResult)
async def predict_batch(files: list[UploadFile] = File(...), _user: str = Depends(get_current_user)):
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided")

    results = []
    total_ms = 0.0
    for f in files:
        data = await _validate_image(f)
        image_id = str(uuid.uuid4())
        result = detector.predict(data, image_id)
        results.append(result)
        total_ms += result.inference_time_ms

    return BatchResult(results=results, total_inference_time_ms=round(total_ms, 2))
