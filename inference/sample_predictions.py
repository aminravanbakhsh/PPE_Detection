"""Save prediction images for val samples that emphasize SH17 PPE + Face-shield.

Selects images whose YOLO labels include at least one class in the SH17
equipment set (class ids 1–16, excluding Person-only frames) or class 17
(Face-shield) on the extended dataset. Then runs ``YOLO.predict`` with
``save=True`` into a dedicated Ultralytics run directory.

Examples
--------
# Baseline val, fine-tuned baseline weights
python -m inference.sample_predictions \\
    --model models/baseline/weights/best.pt \\
    --data training/configs/sh17_baseline_resized.yaml \\
    --project results/baseline/samples \\
    --name finetuned

# Extended val (includes Face-shield labels), extended weights
python -m inference.sample_predictions \\
    --model models/extended/weights/best.pt \\
    --data training/configs/sh17_extended_resized.yaml \\
    --project results/extended/samples \\
    --name finetuned
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ultralytics import YOLO

from training.log_utils import setup_logger

# SH17: 0=Person … 16=Safety-suit. We require at least one box in 1–16 (PPE) or 17 (Face-shield).
_SH17_PPE_MIN = 1
_SH17_PPE_MAX = 16
_FACE_SHIELD_ID = 17


def _label_classes(label_path: Path) -> set[int]:
    if not label_path.is_file():
        return set()
    classes: set[int] = set()
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            classes.add(int(float(parts[0])))
        except (ValueError, IndexError):
            continue
    return classes


def _is_focus_image(classes: set[int], nc: int) -> bool:
    """True if labels include SH17 PPE (1–16) and/or Face-shield (17 when present in dataset)."""
    if not classes:
        return False
    if any(_SH17_PPE_MIN <= c <= _SH17_PPE_MAX for c in classes):
        return True
    if nc > _FACE_SHIELD_ID and _FACE_SHIELD_ID in classes:
        return True
    return False


def _val_dirs(data_yaml: Path) -> tuple[Path, Path, int]:
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["path"]).resolve()
    val_rel = cfg.get("val", "images/val")
    val_img = root / val_rel
    lbl_rel = val_rel.replace("images", "labels", 1)
    val_lbl = root / lbl_rel
    nc = int(cfg.get("nc", 0))
    return val_img, val_lbl, nc


def list_focus_images(
    data_yaml: Path,
    *,
    seed: int,
    max_images: int,
    min_with_face_shield: int = 0,
) -> list[Path]:
    val_img, val_lbl, nc = _val_dirs(data_yaml)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    candidates: list[Path] = []
    for img in sorted(val_img.iterdir()):
        if img.suffix.lower() not in exts:
            continue
        lbl = val_lbl / f"{img.stem}.txt"
        cls = _label_classes(lbl)
        if not _is_focus_image(cls, nc):
            continue
        candidates.append(img)

    rng = random.Random(seed)

    if min_with_face_shield > 0 and nc > _FACE_SHIELD_ID:
        with_fs: list[Path] = []
        sh17_ppe_only: list[Path] = []
        for img in candidates:
            lbl = val_lbl / f"{img.stem}.txt"
            cls = _label_classes(lbl)
            if _FACE_SHIELD_ID in cls:
                with_fs.append(img)
            else:
                sh17_ppe_only.append(img)
        rng.shuffle(with_fs)
        rng.shuffle(sh17_ppe_only)
        take_fs = min(min_with_face_shield, len(with_fs), max_images)
        out = with_fs[:take_fs]
        need = max_images - len(out)
        out.extend(sh17_ppe_only[:need])
        if len(out) < max_images:
            spill_need = max_images - len(out)
            spill = [p for p in with_fs[take_fs:] if p not in out]
            out.extend(spill[:spill_need])
        return out

    rng.shuffle(candidates)
    return candidates[:max_images]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict on val images that contain SH17 PPE and/or Face-shield labels.",
    )
    p.add_argument("--model", type=str, required=True, help="Path to .pt weights")
    p.add_argument("--data", type=str, required=True, help="Dataset YAML (for val paths + nc)")
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-images", type=int, default=8, help="Max images to run")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed for selection")
    p.add_argument(
        "--min-with-face-shield",
        type=int,
        default=-1,
        help="Prefer this many images that include class 17 (Face-shield). "
        "Default: 2 for 18-class data, 0 for 17-class.",
    )
    p.add_argument("--project", type=str, required=True, help="Ultralytics project dir")
    p.add_argument("--name", type=str, default="samples", help="Ultralytics run name")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("sample_predictions", "05_inference.log")
    data_yaml = Path(args.data).resolve()
    if not data_yaml.is_file():
        logger.error("Dataset YAML not found: %s", data_yaml)
        sys.exit(1)
    if not Path(args.model).is_file():
        logger.error("Weights not found: %s", args.model)
        sys.exit(1)

    with open(data_yaml) as f:
        _nc = int(yaml.safe_load(f).get("nc", 0))
    min_fs = args.min_with_face_shield
    if min_fs < 0:
        min_fs = min(2, args.max_images) if _nc > _FACE_SHIELD_ID else 0

    paths = list_focus_images(
        data_yaml,
        seed=args.seed,
        max_images=args.max_images,
        min_with_face_shield=min_fs,
    )
    if not paths:
        logger.error(
            "No val images matched focus criteria (SH17 PPE ids 1–16 and/or Face-shield id 17)."
        )
        sys.exit(1)

    logger.info("Model %s | data %s | %d focus images", args.model, data_yaml, len(paths))
    for p in paths:
        logger.info("  %s", p.name)

    model = YOLO(args.model)
    # Ultralytics accepts a list of paths as source
    model.predict(
        source=[str(p) for p in paths],
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        project=os.path.abspath(args.project),
        name=args.name,
        exist_ok=True,
        verbose=False,
    )
    save_root = Path(args.project).resolve() / args.name
    logger.info("Saved under %s", save_root)


if __name__ == "__main__":
    main()
