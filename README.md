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

# command
python training/train.py \
  --config training/configs/sh17_baseline_resized.yaml \
  --model yolov8m.pt \
  --epochs 10 \
  --batch 32 \
  --imgsz 640 \
  --device mps \
  --name baseline \
  --project runs/detect/runs \
  --no-amp \
  --freeze 22 \
  --cache ram
```

### 4. Inference & Evaluation

```bash
# Evaluate any model on any dataset (baseline example)
python -m inference.infer \
  --model runs/detect/runs/baseline/weights/best.pt \
  --data training/configs/sh17_baseline.yaml \
  --device mps \
  --output results_baseline.json \
  --name baseline_eval

# Same model on merged (extended) dataset
python -m inference.infer \
  --model runs/detect/runs/baseline/weights/best.pt \
  --data training/configs/sh17_extended.yaml \
  --device mps \
  --output results_merged.json \
  --name merged_eval
```

Key options: `--conf` (confidence threshold, default 0.25), `--iou` (NMS IoU, default 0.7), `--imgsz` (image size, default 640), `--split` (`val` or `test`), `--batch` (default 8).

Outputs a JSON report with mAP\@0.5, mAP\@0.5:0.95, precision, recall, F1 (overall and per-class), plus a formatted console summary.

### 5. Baseline vs Extended Comparison

Compare two inference JSON reports against the rubric criteria (Face-shield mAP\@0.5 >= 0.65, no class regresses > 3pp):

```bash
python -m inference.compare \
  --baseline results_baseline.json \
  --extended results_extended.json \
  --output comparison_results.json
```

Options: `--face-shield-target` (default 0.65), `--regression-threshold` (default 0.03).

### 6. API

```bash
cp .env.example .env   # edit secrets
cd api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 7. Docker

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
├── inference/              # standalone inference & metrics
│   ├── infer.py            # CLI: evaluate any model on any dataset
│   ├── compare.py          # CLI: baseline vs extended comparison with rubric checks
│   └── metrics.py          # metric extraction, formatting, JSON export
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
