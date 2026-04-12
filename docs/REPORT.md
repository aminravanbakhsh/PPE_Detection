# PPE Detection — Project Report

This document summarizes data, training, quantitative evaluation (`inference.infer`), rubric comparison (`inference.compare`), qualitative samples, and the inference API. It aligns with the ML engineer assignment brief (*PPE Detection with Fine-tuned YOLO + Real-time Inference API*).

---

## 1. Environment and setup

| Item | Value |
|------|--------|
| Python | 3.11.x (see repo `README` / local `pyenv`) |
| PyTorch | 2.11.0 |
| Ultralytics | 8.4.36 |
| Evaluation device | Apple MPS (`--device mps`) |
| Inference image size | 640 |
| Default `conf` / IoU (eval) | 0.25 / 0.7 |

Dataset roots (from YAML):

- Baseline (17 classes): `data/sh17_baseline_resized`
- Extended (18 classes, +Face-shield): `data/merged_resized`

---

## 2. Data and pipelines

### Sources

- **SH17**: base PPE dataset (17 classes), CC BY 4.0.
- **APD v22 (Roboflow)**: additional source for face-shield and aligned classes; merged into the extended split.

### Merge and labels

See [DECISIONS.md](DECISIONS.md) for class mapping, polygon→bbox conversion (≈12.4% of APD lines), and removal of negative “absence” classes.

### Training script note

Training uses [`training/train.py`](../training/train.py) from the repo root so [`training/tal_patch.py`](../training/tal_patch.py) patches the TaskAlignedAssigner (required on MPS; do not substitute raw `yolo train` if you need identical assigner behavior).

---

## 3. Training configuration (reproducibility)

### Baseline (matches `logs/training_terminal/baseline_train_10ep.log`)

YOLOv8m, `training/configs/sh17_baseline_resized.yaml`, 10 epochs, batch 32, imgsz 640, MPS, workers 4, `lr0=0.01`, cosine LR, cache off, AMP off, `freeze=22`, `max_det=100`, val every 5 epochs, `project=models`, `name=baseline`.

```bash
python training/train.py \
  --config training/configs/sh17_baseline_resized.yaml \
  --model yolov8m.pt \
  --epochs 10 --batch 32 --imgsz 640 --device mps --workers 4 \
  --lr0 0.01 --cos-lr --cache false --fraction 1.0 --no-amp \
  --max-det 100 --val-period 5 --freeze 22 --name baseline
```

### Extended (same hyperparameters as baseline, extended YAML)

The checkpoint under `models/extended/` is trained with the **same 10-epoch full-data recipe** as baseline (see `logs/training_terminal/extended_train_10ep.log`). Older `models/extended/args.yaml` on disk may still reflect an earlier smoke run; trust the training log and `results/extended/metrics_finetuned.json` for the current weights.

```bash
python training/train.py \
  --config training/configs/sh17_extended_resized.yaml \
  --model yolov8m.pt \
  --epochs 10 --batch 32 --imgsz 640 --device mps --workers 4 \
  --lr0 0.01 --cos-lr --cache false --fraction 1.0 --no-amp \
  --max-det 100 --val-period 5 --freeze 22 --name extended
```

(Longer `--epochs` is optional if you need higher Face-shield mAP.)

---

## 4. Quantitative results (validation)

Metrics from `python -m inference.infer` on the **val** split, batch 8, same conf/IoU/imgsz. JSON artifacts:

| Run | Model | Data YAML | Output JSON |
|-----|--------|-----------|-------------|
| Pretrained COCO | `yolov8m.pt` | `sh17_baseline_resized.yaml` | `results/baseline/metrics_pretrained_yolov8m.json` |
| Baseline fine-tuned | `models/baseline/weights/best.pt` | `sh17_baseline_resized.yaml` | `results/baseline/metrics_finetuned.json` |
| Extended fine-tuned | `models/extended/weights/best.pt` | `sh17_extended_resized.yaml` | `results/extended/metrics_finetuned.json` |

### Overall detection metrics

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 |
|-------|---------|--------------|-----------|--------|-----|
| Pretrained YOLOv8m (COCO) | 0.0537 | 0.0478 | 0.0524 | 0.0519 | 0.0521 |
| Baseline fine-tuned | 0.4945 | 0.3190 | 0.5597 | 0.3996 | 0.4395 |
| Extended fine-tuned | 0.5006 | 0.3143 | 0.5643 | 0.3993 | 0.4342 |

**Interpretation:** COCO-pretrained weights do not match SH17 class names or count; the reported per-class AP for that row is COCO-aligned and **not** a meaningful PPE benchmark—only the fine-tuned rows use the SH17 / extended taxonomies. Extended fine-tuning now matches baseline overall mAP (18-class merged val) but the assignment rubric still fails on Face-shield AP@0.5 and several class regressions (see §5).

### Speed (ms/image, from JSON `speed` field)

| Model | Preprocess | Inference | Postprocess |
|-------|------------|-----------|-------------|
| Pretrained | 0.22 | 19.90 | 13.82 |
| Baseline fine-tuned | 0.21 | 19.56 | 13.62 |
| Extended fine-tuned | 0.20 | 20.74 | 14.59 |

---

## 5. Rubric (assignment TASK 02)

Criteria from the brief: **Face-shield mAP@0.5 ≥ 0.65**; **no existing class regresses more than 3 percentage points** vs the baseline checkpoint.

Comparison command:

```bash
python -m inference.compare \
  --baseline results/baseline/metrics_finetuned.json \
  --extended results/extended/metrics_finetuned.json \
  --output results/comparison/rubric.json
```

**Result with current checkpoints:** **FAIL** (see `results/comparison/rubric.json`). Face-shield mAP@0.5 is **0.3595** (target ≥ 0.65). Regression failures (>3 pp drop vs baseline on shared classes): **Person, Face, Foot, Medical-suit**. Rerun `infer` + `compare` after further extended training or hyperparameter tuning if you need a passing rubric.

---

## 6. Qualitative samples (SH17 PPE + Face-shield focus)

Saved prediction images use [`inference/sample_predictions.py`](../inference/sample_predictions.py): val frames with **SH17 equipment ids 1–16** (excludes Person-only) and, on the 18-class split, a mix of **Face-shield (id 17)** and other SH17 PPE images.

| Checkpoint | Output directory (Ultralytics `predict` saves, e.g. `image0.jpg` …) |
|------------|---------------------------------------------------------------------|
| `yolov8m.pt` | `results/baseline/samples/pretrained_yolov8m_sh17_focus/` |
| Baseline `best.pt` | `results/baseline/samples/finetuned_baseline_sh17_focus/` |
| Extended `best.pt` | `results/extended/samples/finetuned_extended_mixed_focus/` |

Pretrained visuals use **COCO class names** on PPE-centric frames (illustrative only).

---

## 7. Inference API (assignment TASK 03)

Required capabilities from the brief: **single and batch image inference**, **health**, **metrics**, **auth token**, **containerized local run**.

Implemented routes (FastAPI, see `api/` and [README.md](../README.md)):

| Method | Path | Role |
|--------|------|------|
| POST | `/auth/token` | JWT for `/predict/*` |
| POST | `/predict/image` | Single image |
| POST | `/predict/batch` | Batch |
| GET | `/health` | Liveness |
| GET | `/metrics` | Prometheus-style metrics |

Local: `cd api && uvicorn app.main:app --reload`; Docker: `docker compose up --build`.

### Security (brief: vulnerabilities + mitigations + cloud)

Documented mitigations in code and [DECISIONS.md](DECISIONS.md): JWT on predict routes, file type/size limits, rate limiting, CORS, PIL validation. For Azure/cloud: Key Vault, WAF/API Management, private endpoints, managed identity, TLS termination, container scanning, central logging—see section 5 of the earlier REPORT template and expand in presentation slides as required.

---

## 8. Assignment deliverables checklist

| Deliverable | Status / location |
|-------------|-------------------|
| Reproducible fine-tuning pipeline | `training/train.py`, configs under `training/configs/` |
| Extended data with face shield | `data/merged_resized`, `sh17_extended_resized.yaml` |
| Evaluation metrics JSON | `results/baseline/*.json`, `results/extended/*.json` |
| Baseline vs extended rubric | `results/comparison/rubric.json`, `inference/compare.py` |
| REST API + optional WebSocket | REST implemented; WebSocket not required by core brief |
| Containerized API | `docker-compose.yml`, `api/Dockerfile` |
| Presentation (slides + demo) | Out of repo — prepare separately |

---

## 9. Reproducibility commands (end-to-end)

```bash
# Eval (three models)
python -m inference.infer --model yolov8m.pt --data training/configs/sh17_baseline_resized.yaml \
  --output results/baseline/metrics_pretrained_yolov8m.json --project results --name baseline_pretrained_eval
python -m inference.infer --model models/baseline/weights/best.pt --data training/configs/sh17_baseline_resized.yaml \
  --output results/baseline/metrics_finetuned.json --project results --name baseline_finetuned_eval
python -m inference.infer --model models/extended/weights/best.pt --data training/configs/sh17_extended_resized.yaml \
  --output results/extended/metrics_finetuned.json --project results --name extended_finetuned_eval

# Rubric (exits non-zero on FAIL — use `|| true` if chaining shell steps)
python -m inference.compare \
  --baseline results/baseline/metrics_finetuned.json \
  --extended results/extended/metrics_finetuned.json \
  --output results/comparison/rubric.json || true

# Focused qualitative samples
python -m inference.sample_predictions --model yolov8m.pt \
  --data training/configs/sh17_baseline_resized.yaml \
  --project results/baseline/samples --name pretrained_yolov8m_sh17_focus --min-with-face-shield 0
python -m inference.sample_predictions --model models/baseline/weights/best.pt \
  --data training/configs/sh17_baseline_resized.yaml \
  --project results/baseline/samples --name finetuned_baseline_sh17_focus --min-with-face-shield 0
python -m inference.sample_predictions --model models/extended/weights/best.pt \
  --data training/configs/sh17_extended_resized.yaml \
  --project results/extended/samples --name finetuned_extended_mixed_focus
```

---

## 10. Key takeaways

- **Fine-tuned baseline** substantially outperforms COCO-pretrained weights on SH17 val (expected).
- **Extended** is trained with the same 10-epoch full-data recipe as baseline; overall val mAP is in line with baseline, but the rubric still **fails** (Face-shield AP@0.5 below 0.65; regressions on Person, Face, Foot, Medical-suit).
- **Qualitative figures** should use `sample_predictions.py` so visuals emphasize SH17 PPE and Face-shield rather than Person-only val images.
- **API** matches the brief’s core endpoints; security and cloud notes belong in the **presentation** as well as this report.
