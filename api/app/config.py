from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    jwt_secret_key: str = "change-me-to-a-random-secret"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 30
    api_username: str = "admin"
    api_password: str = "changeme"
    model_path: str = "model/best.pt"
    max_file_size_mb: int = 10
    max_batch_images: int = 16
    rate_limit: str = "30/minute"
    cors_origins: str = '["http://localhost:3000"]'

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
