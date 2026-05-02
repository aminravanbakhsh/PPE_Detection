from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings

# Directory containing the `app` package (…/api locally, /app in the Docker image).
_API_DIR = Path(__file__).resolve().parent.parent


def _resolve_model_path(model_path: str) -> str:
    """Resolve relative MODEL_PATH against api dir, repo root (parent of api/), then cwd."""
    p = Path(model_path)
    if p.is_absolute():
        return str(p.resolve())
    rel = model_path
    for base in (_API_DIR, _API_DIR.parent, Path.cwd()):
        candidate = (base / rel).resolve()
        if candidate.is_file():
            return str(candidate)
    return str((_API_DIR.parent / rel).resolve())


class Settings(BaseSettings):
    jwt_secret_key: str = "change-me-to-a-random-secret"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 30
    api_username: str = "admin"
    api_password: str = "changeme"
    model_path: str = "models/extended_sh17/weights/best.pt"
    max_file_size_mb: int = 10
    max_batch_images: int = 16
    rate_limit: str = "30/minute"
    # Include "null" so a demo page opened as file:// can call the API (dev only).
    cors_origins: str = '["http://localhost:3000","null"]'

    model_config = {"env_file": ".env", "extra": "ignore"}

    @model_validator(mode="after")
    def _resolve_model_path_field(self) -> "Settings":
        object.__setattr__(self, "model_path", _resolve_model_path(self.model_path))
        return self


settings = Settings()
