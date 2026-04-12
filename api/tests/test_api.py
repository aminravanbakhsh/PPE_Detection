"""Basic API tests for all endpoints."""

import io
import os
import sys

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.config import settings
from app.main import app

client = TestClient(app)


class _DetectorLoadedStub:
    """Avoid requiring weights on disk for tests that stop before inference."""

    is_loaded = True

    def predict(self, *args, **kwargs):
        raise AssertionError("predict() should not run in this test")


def _create_test_image() -> bytes:
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestAuth:
    def test_login_success(self):
        resp = client.post("/auth/token", json={
            "username": settings.api_username,
            "password": settings.api_password,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_bad_credentials(self):
        resp = client.post("/auth/token", json={
            "username": "wrong",
            "password": "wrong",
        })
        assert resp.status_code == 401

    def test_login_missing_fields(self):
        resp = client.post("/auth/token", json={"username": "admin"})
        assert resp.status_code == 422


class TestHealth:
    def test_health_endpoint(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data


class TestPredict:
    def _get_token(self) -> str:
        resp = client.post("/auth/token", json={
            "username": settings.api_username,
            "password": settings.api_password,
        })
        return resp.json()["access_token"]

    def test_predict_no_auth(self):
        img_bytes = _create_test_image()
        resp = client.post("/predict/image", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
        assert resp.status_code in (401, 403)

    def test_predict_invalid_file_type(self, monkeypatch):
        monkeypatch.setattr("app.routes.predict.detector", _DetectorLoadedStub())
        token = self._get_token()
        resp = client.post("/predict/image",
                           files={"file": ("test.txt", b"not an image", "text/plain")},
                           headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 415

    def test_batch_no_files(self):
        token = self._get_token()
        resp = client.post("/predict/batch",
                           headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 422

    def test_batch_too_many_files(self, monkeypatch):
        monkeypatch.setattr("app.routes.predict.detector", _DetectorLoadedStub())
        monkeypatch.setattr(settings, "max_batch_images", 2)
        token = self._get_token()
        img = _create_test_image()
        files = [
            ("files", ("a.jpg", img, "image/jpeg")),
            ("files", ("b.jpg", img, "image/jpeg")),
            ("files", ("c.jpg", img, "image/jpeg")),
        ]
        resp = client.post(
            "/predict/batch",
            files=files,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 400
        assert "Maximum" in resp.json().get("detail", "")


class TestMetrics:
    def test_metrics_endpoint(self):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "http_request" in resp.text or "process" in resp.text
