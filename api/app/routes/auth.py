from fastapi import APIRouter, HTTPException, Request, status

from ..auth import create_access_token, get_password_hash, verify_password
from ..config import settings
from ..rate_limit import limiter
from ..schemas import TokenRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])

_hashed_pw = get_password_hash(settings.api_password)


@router.post("/token", response_model=TokenResponse)
@limiter.limit(settings.rate_limit)
def login(request: Request, payload: TokenRequest):
    if payload.username != settings.api_username or not verify_password(payload.password, _hashed_pw):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(subject=payload.username)
    return TokenResponse(access_token=token)
