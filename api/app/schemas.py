from pydantic import BaseModel


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_name: str
    class_id: int
    confidence: float
    bbox: BBox


class ImageResult(BaseModel):
    image_id: str
    detections: list[Detection]
    inference_time_ms: float


class BatchResult(BaseModel):
    results: list[ImageResult]
    total_inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
