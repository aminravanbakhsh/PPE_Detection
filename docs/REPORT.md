# PPE Detection — Final Report

## 1. Data

### Sources
- **SH17**: 8,099 images, 17 PPE classes (helmets, gloves, vests, goggles, boots, etc.)
- **APD v22**: 5,324 images, 9 classes — used for face shield annotations

### Merge Process
- Polygon-to-bbox conversion for 12.4% of APD annotations
- Negative class removal (4 absence classes discarded)
- Class ID remapping: 5 APD classes mapped to SH17 equivalents + 1 new class (face shield = 17)

### Final Dataset Statistics
<!-- TODO: Fill after running prepare_data.py -->

## 2. Model

### Architecture
- YOLOv8m pretrained on COCO
- Fine-tuned at 640x640 resolution

### Training Configuration
- Baseline: SH17-only (17 classes), 100 epochs
- Extended: SH17 + APD merged (18 classes), 100 epochs
- Optimizer: SGD with cosine LR schedule, lr0=0.01
- Device: Apple M4 MPS, batch=8

## 3. Performance

### Baseline vs Extended Comparison
<!-- TODO: Fill after training -->

### Face Shield mAP@0.5
<!-- TODO: Must be >= 0.65 -->

### Regression Analysis
<!-- TODO: No class should regress > 3pp -->

## 4. API Design

### Endpoints
- `POST /auth/token` — JWT token generation
- `POST /predict/image` — single image inference
- `POST /predict/batch` — batch inference
- `GET /health` — service health check
- `GET /metrics` — Prometheus metrics

### Authentication
- JWT-based, configurable expiry
- Required on `/predict/*` routes

## 5. Security Review

### Implemented Mitigations
- JWT authentication on inference endpoints
- File type validation (JPEG/PNG only)
- File size limits (configurable, default 10MB)
- Rate limiting (slowapi)
- CORS origin restrictions
- Input image sanitization via PIL

### Cloud Deployment Considerations
- Azure Key Vault for secrets management
- Azure WAF / API Management for DDoS protection
- Managed Identity instead of static credentials
- VNET + Private Endpoints for internal traffic
- HTTPS/TLS termination at load balancer
- Container image scanning (Trivy/Azure Defender)
- Azure Monitor / Application Insights for observability

## 6. Key Takeaways
<!-- TODO: Fill after completing all phases -->
