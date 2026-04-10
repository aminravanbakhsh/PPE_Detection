"""Merge SH17 + APD datasets into a unified YOLO-format dataset.

Handles polygon-to-bbox conversion, negative class removal, class ID remapping,
annotation validation, and produces a merge report.
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

from PIL import Image

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.log_utils import setup_logger

logger = setup_logger("data_prep", "01_data_preparation.log")

# ---------------------------------------------------------------------------
# SH17 class names (0-16)
# ---------------------------------------------------------------------------
SH17_CLASSES = [
    "Person", "Head", "Face", "Glasses", "Face-mask-medical",
    "Face-guard", "Ear", "Earmuffs", "Hands", "Gloves",
    "Foot", "Shoes", "Safety-vest", "Tools", "Helmet",
    "Medical-suit", "Safety-suit",
]

FACE_SHIELD_CLASS_ID = 17
EXTENDED_CLASSES = SH17_CLASSES + ["Face-shield"]

# APD original classes: 0=earmuff, 1=face shield, 2=gloves, 3=mask,
#   4=no earmuff, 5=no face shield, 6=no glove, 7=no mask, 8=person
APD_DISCARD = {4, 5, 6, 7}  # negative classes
APD_REMAP = {
    0: 7,   # earmuff  -> Earmuffs
    1: 17,  # face shield -> Face-shield (new)
    2: 9,   # gloves   -> Gloves
    3: 4,   # mask     -> Face-mask-medical
    8: 0,   # person   -> Person
}


def polygon_to_bbox(fields: list[float]) -> tuple[float, float, float, float]:
    """Convert polygon coordinates to axis-aligned bounding box (x_center, y_center, w, h)."""
    xs = fields[0::2]
    ys = fields[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return cx, cy, w, h


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def process_apd_label(line: str, stats: dict) -> str | None:
    """Process a single APD annotation line. Returns remapped YOLO line or None."""
    parts = line.strip().split()
    if len(parts) < 5:
        stats["skipped_short"] += 1
        return None

    cls_id = int(parts[0])
    if cls_id in APD_DISCARD:
        stats["discarded_negative"] += 1
        return None
    if cls_id not in APD_REMAP:
        stats["skipped_unknown_class"] += 1
        return None

    new_cls = APD_REMAP[cls_id]
    coords = [float(x) for x in parts[1:]]

    if len(coords) == 4:
        cx, cy, w, h = coords
    else:
        stats["polygon_converted"] += 1
        cx, cy, w, h = polygon_to_bbox(coords)

    cx, cy = clamp(cx), clamp(cy)
    w, h = clamp(w, 0.0, 1.0), clamp(h, 0.0, 1.0)

    area = w * h
    if area < 0.001:
        stats["warn_tiny_bbox"] += 1
    if area > 0.9:
        stats["warn_large_bbox"] += 1

    stats["class_counts"][new_cls] = stats["class_counts"].get(new_cls, 0) + 1
    return f"{new_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def process_sh17_label(line: str, stats: dict) -> str | None:
    """Validate and pass through a SH17 annotation line."""
    parts = line.strip().split()
    if len(parts) != 5:
        stats["skipped_bad_format"] += 1
        return None

    cls_id = int(parts[0])
    if cls_id < 0 or cls_id > 16:
        stats["skipped_bad_class"] += 1
        return None

    cx, cy, w, h = [float(x) for x in parts[1:]]
    cx, cy = clamp(cx), clamp(cy)
    w, h = clamp(w, 0.0, 1.0), clamp(h, 0.0, 1.0)

    stats["class_counts"][cls_id] = stats["class_counts"].get(cls_id, 0) + 1
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def validate_image(path: Path) -> bool:
    """Check that an image file is valid and readable."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def copy_and_process_apd(apd_dir: Path, out_dir: Path, split: str,
                         stats: dict, sample_n: int | None = None):
    """Copy APD images and process labels into the merged output."""
    img_dir = apd_dir / split / "images"
    lbl_dir = apd_dir / split / "labels"
    out_img = out_dir / "images" / ("train" if split != "valid" else "val")
    out_lbl = out_dir / "labels" / ("train" if split != "valid" else "val")

    if not img_dir.exists():
        logger.warning(f"APD {split} images dir not found: {img_dir}")
        return

    images = sorted(img_dir.glob("*"))
    if sample_n is not None:
        images = images[:sample_n]

    for img_path in images:
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        if not lbl_path.exists():
            stats["missing_labels"] += 1
            continue

        lines = lbl_path.read_text().strip().split("\n")
        processed = []
        for line in lines:
            if not line.strip():
                continue
            result = process_apd_label(line, stats)
            if result:
                processed.append(result)

        if not processed:
            stats["skipped_empty_after_filter"] += 1
            continue

        if not validate_image(img_path):
            stats["invalid_images"] += 1
            continue

        dest_name = f"apd_{stem}"
        shutil.copy2(img_path, out_img / f"{dest_name}{img_path.suffix}")
        (out_lbl / f"{dest_name}.txt").write_text("\n".join(processed) + "\n")
        stats["images_copied"] += 1


def copy_and_process_sh17(sh17_dir: Path, out_dir: Path, split_file: str,
                          stats: dict, sample_n: int | None = None):
    """Copy SH17 images and labels using the train/val split files."""
    split_path = sh17_dir / split_file
    if not split_path.exists():
        logger.error(f"Split file not found: {split_path}")
        return

    filenames = [f.strip() for f in split_path.read_text().strip().split("\n") if f.strip()]
    if sample_n is not None:
        filenames = filenames[:sample_n]

    is_train = "train" in split_file
    out_img = out_dir / "images" / ("train" if is_train else "val")
    out_lbl = out_dir / "labels" / ("train" if is_train else "val")
    img_dir = sh17_dir / "images"
    lbl_dir = sh17_dir / "labels"

    for fname in filenames:
        stem = Path(fname).stem
        img_path = None
        for ext in [".jpeg", ".jpg", ".png"]:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            stats["missing_images"] += 1
            continue

        lbl_path = lbl_dir / f"{stem}.txt"
        if not lbl_path.exists():
            stats["missing_labels"] += 1
            continue

        lines = lbl_path.read_text().strip().split("\n")
        processed = []
        for line in lines:
            if not line.strip():
                continue
            result = process_sh17_label(line, stats)
            if result:
                processed.append(result)

        if not processed:
            stats["skipped_empty_after_filter"] += 1
            continue

        dest_name = f"sh17_{stem}"
        shutil.copy2(img_path, out_img / f"{dest_name}{img_path.suffix}")
        (out_lbl / f"{dest_name}.txt").write_text("\n".join(processed) + "\n")
        stats["images_copied"] += 1


def write_data_yaml(out_dir: Path, nc: int, names: list[str]):
    """Write the Ultralytics data.yaml config."""
    yaml_content = f"""path: {out_dir.resolve()}
train: images/train
val: images/val

nc: {nc}
names: {names}
"""
    (out_dir / "data.yaml").write_text(yaml_content)
    logger.info(f"Wrote data.yaml with {nc} classes")


def main():
    parser = argparse.ArgumentParser(description="Merge SH17 + APD datasets")
    parser.add_argument("--sh17-dir", type=Path, default=Path("data/sh17"),
                        help="Path to SH17 dataset root")
    parser.add_argument("--apd-dir", type=Path, default=Path("data/apd.v22i.yolov8"),
                        help="Path to APD dataset root")
    parser.add_argument("--output-dir", type=Path, default=Path("data/merged"),
                        help="Output directory for merged dataset")
    parser.add_argument("--sample", action="store_true",
                        help="Use small subset (50 SH17 + 20 APD images per split)")
    parser.add_argument("--sh17-only", action="store_true",
                        help="Only process SH17 data (for baseline config)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = args.output_dir
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    sh17_sample = 50 if args.sample else None
    apd_sample = 20 if args.sample else None

    sh17_stats = {
        "class_counts": {}, "images_copied": 0, "missing_labels": 0,
        "missing_images": 0, "skipped_empty_after_filter": 0,
        "skipped_bad_format": 0, "skipped_bad_class": 0,
    }
    apd_stats = {
        "class_counts": {}, "images_copied": 0, "missing_labels": 0,
        "missing_images": 0, "invalid_images": 0,
        "skipped_empty_after_filter": 0, "skipped_short": 0,
        "discarded_negative": 0, "skipped_unknown_class": 0,
        "polygon_converted": 0, "warn_tiny_bbox": 0, "warn_large_bbox": 0,
    }

    logger.info(f"SH17 dir: {args.sh17_dir.resolve()}")
    logger.info(f"APD dir: {args.apd_dir.resolve()}")
    logger.info(f"Output dir: {out_dir.resolve()}")
    logger.info(f"Sample mode: {args.sample}")
    logger.info(f"SH17-only mode: {args.sh17_only}")

    # Process SH17
    logger.info("Processing SH17 train split...")
    copy_and_process_sh17(args.sh17_dir, out_dir, "train_files.txt", sh17_stats, sh17_sample)
    logger.info(f"SH17 train: {sh17_stats['images_copied']} images copied")

    logger.info("Processing SH17 val split...")
    val_before = sh17_stats["images_copied"]
    copy_and_process_sh17(args.sh17_dir, out_dir, "val_files.txt", sh17_stats, sh17_sample)
    logger.info(f"SH17 val: {sh17_stats['images_copied'] - val_before} images copied")

    # Process APD
    if not args.sh17_only:
        logger.info("Processing APD train split...")
        copy_and_process_apd(args.apd_dir, out_dir, "train", apd_stats, apd_sample)
        logger.info(f"APD train: {apd_stats['images_copied']} images copied")

        logger.info("Processing APD valid split...")
        val_before_apd = apd_stats["images_copied"]
        copy_and_process_apd(args.apd_dir, out_dir, "valid", apd_stats, apd_sample)
        logger.info(f"APD valid: {apd_stats['images_copied'] - val_before_apd} images copied")

        logger.info("Processing APD test split (merged into val)...")
        test_before = apd_stats["images_copied"]
        copy_and_process_apd(args.apd_dir, out_dir, "test", apd_stats, apd_sample)
        logger.info(f"APD test->val: {apd_stats['images_copied'] - test_before} images copied")

    # Write data.yaml
    if args.sh17_only:
        write_data_yaml(out_dir, len(SH17_CLASSES), SH17_CLASSES)
    else:
        write_data_yaml(out_dir, len(EXTENDED_CLASSES), EXTENDED_CLASSES)

    # Merge report
    all_class_counts = {}
    for cid, cnt in sh17_stats["class_counts"].items():
        all_class_counts[cid] = all_class_counts.get(cid, 0) + cnt
    if not args.sh17_only:
        for cid, cnt in apd_stats["class_counts"].items():
            all_class_counts[cid] = all_class_counts.get(cid, 0) + cnt

    class_names = EXTENDED_CLASSES if not args.sh17_only else SH17_CLASSES
    named_counts = {class_names[int(k)]: v for k, v in sorted(all_class_counts.items(), key=lambda x: int(x[0]))}

    report = {
        "sh17_stats": sh17_stats,
        "apd_stats": apd_stats if not args.sh17_only else "skipped",
        "total_images": sh17_stats["images_copied"] + (apd_stats["images_copied"] if not args.sh17_only else 0),
        "class_distribution": named_counts,
        "num_classes": len(class_names),
    }
    report_path = out_dir / "merge_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))

    logger.info(f"Total images merged: {report['total_images']}")
    logger.info(f"Class distribution: {json.dumps(named_counts, indent=2)}")
    if not args.sh17_only:
        logger.info(f"APD polygons converted: {apd_stats['polygon_converted']}")
        logger.info(f"APD negative classes discarded: {apd_stats['discarded_negative']}")
    logger.info(f"Merge report saved to {report_path}")


if __name__ == "__main__":
    main()
