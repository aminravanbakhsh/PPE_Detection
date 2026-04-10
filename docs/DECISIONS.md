# Decision Log

Key decisions and reasoning made during the build.

## Data

### Dataset Selection
- **SH17** as the base dataset: 8,099 images, 17 PPE classes, YOLO format, CC BY 4.0.
- **APD v22 (Roboflow)** for face shield data: 5,324 images, 9 classes, CC BY 4.0. Chosen because it provides face shield annotations in YOLO format compatible with SH17.

### Class Mapping
- APD `face shield` (class 1) becomes merged class 17 (new class).
- Overlapping classes mapped to SH17 equivalents: person->0, mask->4, earmuff->7, gloves->9.
- APD negative classes (`no earmuff`, `no face shield`, `no glove`, `no mask`) discarded — they represent absence of PPE and would confuse a detection model.

### Polygon-to-Bbox Conversion
- 12.4% of APD annotation lines use polygon/segmentation format (>5 fields per line).
- Converted to axis-aligned bounding boxes via min/max of x/y polygon vertices.
- Simpler than training a segmentation model and matches SH17's bbox-only format.

## Training

### Joint Training vs Sequential Fine-Tuning
- Chose joint training from a COCO-pretrained checkpoint on all data simultaneously rather than sequentially fine-tuning the baseline with new data.
- Joint training is the primary defense against catastrophic forgetting on existing classes.

### YOLOv8m Selection
- YOLOv8m balances accuracy and speed for PPE detection.
- Pretrained on COCO provides strong transfer learning foundation.

### MPS Device for Local Training
- Apple M4 with MPS backend provides ~3-5x speedup over CPU.
- Batch size set to 8 to stay within 16GB unified memory budget.

## API

### JWT Authentication
- Chose JWT over API keys for stateless auth that scales horizontally.
- Tokens have configurable expiry (default 30 min).

### Rate Limiting
- `slowapi` for per-IP rate limiting in the application layer.
- Cloud deployment would add WAF/API Management layer.

### File Validation
- PIL-based image validation before YOLO inference catches corrupted/malicious uploads.
- MIME type + extension check prevents non-image uploads.

## Infrastructure

### Docker Base Image
- `python:3.11-slim` — minimal footprint, compatible with ARM64 and AMD64.
- Multi-arch builds supported via Docker Desktop on macOS.
