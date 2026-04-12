# Task 03 — Inference API, containerization, and security review

This document summarizes the PPE Detection inference API, how to run it locally and in Docker, intentional security controls, identified risks with mitigations, and cloud (Azure) deployment considerations for a presentation or rubric.

## 1. Summary

The service exposes a fine-tuned YOLOv8 model for Personal Protective Equipment (PPE) detection over HTTP. Clients obtain a short-lived JWT, then upload one or more images (JPEG/PNG) for bounding-box predictions. The stack is **FastAPI**, **Ultralytics YOLO**, **JWT (HS256)** with **bcrypt**-verified credentials, **Prometheus**-compatible HTTP metrics, and **SlowAPI** rate limits on authentication and inference routes.

## 2. Design

### 2.1 Components

| Piece | Role |
|--------|------|
| [`api/app/main.py`](../api/app/main.py) | FastAPI app, CORS, lifespan (load model), Prometheus `/metrics`, global error handler |
| [`api/app/detector.py`](../api/app/detector.py) | Singleton `Detector`: loads `.pt` once, runs `YOLO` inference, returns structured detections |
| [`api/app/routes/auth.py`](../api/app/routes/auth.py) | `POST /auth/token` — username/password → JWT |
| [`api/app/routes/predict.py`](../api/app/routes/predict.py) | `POST /predict/image`, `POST /predict/batch` — Bearer JWT required |
| [`api/app/routes/health.py`](../api/app/routes/health.py) | `GET /health` — process up + `model_loaded` flag |
| [`api/app/auth.py`](../api/app/auth.py) | bcrypt + `python-jose` encode/decode JWT |
| [`api/app/rate_limit.py`](../api/app/rate_limit.py) | Shared SlowAPI `Limiter` (avoids circular imports) |
| [`api/app/config.py`](../api/app/config.py) | `pydantic-settings`: secrets, paths, `MAX_FILE_SIZE_MB`, `MAX_BATCH_IMAGES`, `RATE_LIMIT`, CORS |

### 2.2 Authentication flow

1. Client `POST /auth/token` with JSON `{ "username", "password" }` (rate-limited per client IP).
2. Server checks credentials against env-configured user and bcrypt hash, returns `{ "access_token", "token_type": "bearer" }`.
3. Client calls `/predict/*` with header `Authorization: Bearer <token>`.
4. JWT is validated (signature, expiry); `sub` carries the username.

### 2.3 Response shapes (JSON)

Defined in [`api/app/schemas.py`](../api/app/schemas.py):

- **Single image:** `image_id`, `detections[]` (`class_name`, `class_id`, `confidence`, `bbox` x1/y1/x2/y2), `inference_time_ms`.
- **Batch:** `results` (list of the above), `total_inference_time_ms`.

### 2.4 Metrics

`GET /metrics` is served by `prometheus-fastapi-instrumentator`: HTTP request counts, latencies, and process metrics suitable for Prometheus scraping (not custom business counters unless extended later).

## 3. How to run

### 3.1 Virtual environment (no Docker)

From repo root:

```bash
cp .env.example .env   # set JWT_SECRET_KEY, API_PASSWORD, MODEL_PATH, etc.
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

`MODEL_PATH` must point to an existing `.pt` file on the host (for example a path under your training `runs/` or `models/` directory).

### 3.2 Docker Compose

From repo root:

```bash
cp .env.example .env
docker compose up --build
```

[`docker-compose.yml`](../docker-compose.yml) mounts `./models` read-only at `/app/model` and sets `MODEL_PATH=model/extended_20ep/weights/best.pt` by default. Adjust `MODEL_PATH` in `.env` if your weights live elsewhere under that mount. `*.pt` files are gitignored; you must supply weights locally.

### 3.3 Example `curl` flows

**Health (no auth):**

```bash
curl -s http://localhost:8000/health
```

**Token:**

```bash
curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"changeme"}'
```

**Single image (replace `TOKEN` and path):**

```bash
curl -s -X POST http://localhost:8000/predict/image \
  -H "Authorization: Bearer TOKEN" \
  -F "file=@/path/to/image.jpg"
```

**Batch:**

```bash
curl -s -X POST http://localhost:8000/predict/batch \
  -H "Authorization: Bearer TOKEN" \
  -F "files=@/path/to/a.jpg" \
  -F "files=@/path/to/b.jpg"
```

### 3.4 Tests

```bash
cd api && PYTHONPATH=. python -m pytest tests/test_api.py -v
```

## 4. Security review (vulnerabilities, risks, mitigations)

| Topic | Risk | Mitigation (implemented or recommended) |
|--------|------|-------------------------------------------|
| Default `JWT_SECRET_KEY` / `API_PASSWORD` | Anyone could forge JWTs or guess admin password. | **Implemented:** values loaded from `.env`; **must** override in any shared or production environment. Rotate on compromise. |
| JWT in `Authorization` header | Token theft (XSS, logs, MITM without TLS) allows inference abuse until expiry. | Short `JWT_EXPIRY_MINUTES`; **production:** HTTPS only, no token in URLs, secure client storage; consider refresh tokens + revocation list only if product needs it. |
| No token revocation | Stolen token valid until expiry. | Acceptable for local demo; **production:** shorter TTL, optional server-side denylist or opaque session store. |
| Brute force on `/auth/token` | Credential guessing. | **Implemented:** SlowAPI rate limit on `/auth/token` (configurable `RATE_LIMIT`); **add:** account lockout or CAPTCHA behind a gateway if public. |
| Inference DoS | Huge uploads or many requests exhaust CPU/RAM. | **Implemented:** per-file size cap, MIME allowlist, PIL `verify()`, `MAX_BATCH_IMAGES`, per-route rate limits; **add:** request body limits at reverse proxy, worker timeouts, autoscaling. |
| Batch endpoint abuse | Many large images in one request. | **Implemented:** `MAX_BATCH_IMAGES` (env); tune per hardware. |
| Malicious / decompression-heavy images | CPU/memory spikes or parser bugs. | **Implemented:** size limit + `verify()`; **gap:** not a full anti-malware pipeline; **add:** image dimension caps after decode, separate scanning service if threat model requires it. |
| Unauthenticated `/metrics` | Exposes request patterns, paths, process info to scanners. | **By design** for Prometheus; **mitigate:** bind metrics on internal port, network policy, or scrape auth (e.g. mTLS) in production. |
| Unauthenticated `/health` | Reveals liveness and whether model is loaded. | Normal for load balancers; **mitigate:** restrict network access if sensitive. |
| CORS `allow_credentials` + broad origins | Cross-site abuse if origins misconfigured. | **Implemented:** `CORS_ORIGINS` JSON in env; **must** list explicit front-end origins, not `*` when using credentials. |
| Error messages | Information leakage (paths, stack traces). | **Implemented:** generic 500 body; detailed errors only in server logs. |
| Dependency / base image CVEs | RCE or data exposure via library bugs. | Pin versions in [`api/requirements.txt`](../api/requirements.txt); scan images (Trivy, Snyk); rebuild periodically. |
| Container runs as root | Container breakout impact. | **Implemented:** non-root `appuser` in [`api/Dockerfile`](../api/Dockerfile). |
| Logging | Accidental logging of tokens or image bytes. | **Policy:** never log `Authorization` headers or raw uploads; review log pipelines in production. |

## 5. Azure / cloud deployment considerations

High-level checklist if this API were deployed to Azure (or similar):

- **Compute:** Azure Container Apps or AKS to run the API image; scale on CPU/request queue; set min/max replicas.
- **Registry:** Azure Container Registry (ACR) with private access; image signing / trusted publishing where possible.
- **Secrets:** Store `JWT_SECRET_KEY` and API credentials in **Azure Key Vault**; inject via **Managed Identity** or ACA secrets, not plain text in compose files.
- **Identity:** Prefer **Managed Identity** for Key Vault and storage access instead of long-lived client secrets.
- **Networking:** **Application Gateway** or **Azure Front Door** for TLS termination, WAF rules, and rate limiting at the edge; **Private Endpoint** if the API is internal-only.
- **Storage:** If weights live in **Blob Storage**, download at startup or mount via secure pattern with least-privilege SAS or identity-based access.
- **Observability:** **Azure Monitor** / Application Insights or Prometheus + Grafana; alert on 5xx rate, latency, and auth failures.
- **Platform security:** **Microsoft Defender for Cloud** for posture and registry scanning; **DDoS Protection** for public endpoints.
- **Data / privacy:** Uploaded images may contain people or workplaces; define retention, encryption at rest, and regional compliance (GDPR, etc.).

## 6. Files touched for Task 03 hardening

- **Rate limiting:** [`api/app/rate_limit.py`](../api/app/rate_limit.py), decorators on `/auth/token`, `/predict/image`, `/predict/batch`; [`api/app/main.py`](../api/app/main.py) imports shared limiter.
- **Batch cap:** `max_batch_images` in [`api/app/config.py`](../api/app/config.py), enforced in [`api/app/routes/predict.py`](../api/app/routes/predict.py); `MAX_BATCH_IMAGES` in [`.env.example`](../.env.example).
- **Docker:** [`api/Dockerfile`](../api/Dockerfile) non-root user and Python env flags; [`docker-compose.yml`](../docker-compose.yml) uses `./models` mount and default `MODEL_PATH` for extended 20-epoch layout.
- **Docs / tests:** this report; [`README.md`](../README.md) Docker notes; [`api/tests/test_api.py`](../api/tests/test_api.py) batch limit test.

---

*This report is intended to accompany Task 03 deliverables: working API, Dockerfile, and explicit security analysis for presentation.*
