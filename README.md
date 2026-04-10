# PPE Detection with Fine-tuned YOLO + Inference API

Fine-tuned YOLOv8m for Personal Protective Equipment detection, extending the SH17 dataset (17 classes) with face shield data from the APD dataset (Roboflow). Includes a containerized FastAPI inference API with JWT auth, batch support, and Prometheus metrics.

## Quick Start

### 1. Environment Setup

```bash
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
python training/prepare_data.py --sample   # small subset for testing
python training/prepare_data.py            # full merge
```

### 3. Training

```bash
# Baseline (SH17-only, 17 classes)
python training/train.py --config training/configs/sh17_baseline.yaml --epochs 2 --device mps

# Extended (SH17 + APD, 18 classes with face shield)
python training/train.py --config training/configs/sh17_extended.yaml --epochs 100 --device mps
```

### 4. Evaluation

```bash
python training/evaluate.py --baseline runs/baseline/weights/best.pt --extended runs/extended/weights/best.pt
```

### 5. API

```bash
cp .env.example .env   # edit secrets
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 6. Docker

```bash
docker compose up --build
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/auth/token` | Get JWT access token |
| POST | `/predict/image` | Single image inference |
| POST | `/predict/batch` | Batch image inference |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

## Project Structure

```
PPE_Detection/
├── data/                   # gitignored — SH17, APD, merged datasets
├── training/               # data prep, training, evaluation scripts
│   ├── configs/            # YOLO dataset YAML configs
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
├── api/                    # FastAPI inference service
│   ├── app/
│   ├── Dockerfile
│   └── tests/
├── docs/                   # DECISIONS.md, REPORT.md
├── docker-compose.yml
└── requirements.txt
```

## License

Datasets used under CC BY 4.0. See `docs/DECISIONS.md` for details.
