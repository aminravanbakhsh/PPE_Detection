"""Resize a YOLO-format dataset with explicit bounding-box recalculation.

Proportionally resizes every image so its longest side equals --max-dim,
then recalculates each bounding box through the full transform:
  normalised -> pixel (original) -> pixel (resized) -> normalised (new).
"""

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.log_utils import setup_logger

logger = setup_logger("resize_dataset", "00_resize_dataset.log")


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def recalculate_bbox(
    cls: int,
    cx_norm: float,
    cy_norm: float,
    w_norm: float,
    h_norm: float,
    w_orig: int,
    h_orig: int,
    w_new: int,
    h_new: int,
    scale: float,
) -> tuple[int, float, float, float, float]:
    """Recalculate a single YOLO bbox through the resize transform."""
    # Step a: normalised -> pixel coords at original size
    cx_px = cx_norm * w_orig
    cy_px = cy_norm * h_orig
    w_px = w_norm * w_orig
    h_px = h_norm * h_orig

    # Step b: apply resize scale
    cx_px_new = cx_px * scale
    cy_px_new = cy_px * scale
    w_px_new = w_px * scale
    h_px_new = h_px * scale

    # Step c: re-normalise to new image dimensions
    cx_out = clamp(cx_px_new / w_new)
    cy_out = clamp(cy_px_new / h_new)
    w_out = clamp(w_px_new / w_new, 0.0, 1.0)
    h_out = clamp(h_px_new / h_new, 0.0, 1.0)

    return cls, cx_out, cy_out, w_out, h_out


def process_split(
    img_in: Path, lbl_in: Path, img_out: Path, lbl_out: Path,
    max_dim: int, quality: int, stats: dict,
) -> None:
    """Resize all images in a split and recalculate their labels."""
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in img_in.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    total = len(image_files)
    logger.info(f"  Found {total} images in {img_in}")

    for idx, img_path in enumerate(image_files):
        lbl_path = lbl_in / f"{img_path.stem}.txt"

        try:
            with Image.open(img_path) as img:
                w_orig, h_orig = img.size

                if max(w_orig, h_orig) <= max_dim:
                    scale = 1.0
                    w_new, h_new = w_orig, h_orig
                    resized = img
                else:
                    scale = max_dim / max(w_orig, h_orig)
                    w_new = round(w_orig * scale)
                    h_new = round(h_orig * scale)
                    resized = img.resize((w_new, h_new), Image.LANCZOS)

                out_path = img_out / f"{img_path.stem}.jpg"
                if resized.mode in ("RGBA", "P"):
                    resized = resized.convert("RGB")
                resized.save(out_path, "JPEG", quality=quality)
                stats["images"] += 1
        except Exception as e:
            logger.warning(f"  Skipping {img_path.name}: {e}")
            stats["skipped_images"] += 1
            continue

        if not lbl_path.exists():
            (lbl_out / f"{img_path.stem}.txt").write_text("")
            stats["missing_labels"] += 1
            continue

        lines = lbl_path.read_text().strip().splitlines()
        out_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                stats["skipped_bboxes"] += 1
                continue

            cls = int(float(parts[0]))
            cx_norm, cy_norm, w_norm, h_norm = map(float, parts[1:])

            cls, cx, cy, w, h = recalculate_bbox(
                cls, cx_norm, cy_norm, w_norm, h_norm,
                w_orig, h_orig, w_new, h_new, scale,
            )

            if w < 1e-4 or h < 1e-4:
                stats["dropped_tiny"] += 1
                continue

            out_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stats["bboxes"] += 1

        (lbl_out / f"{img_path.stem}.txt").write_text("\n".join(out_lines) + "\n" if out_lines else "")

        if (idx + 1) % 500 == 0 or (idx + 1) == total:
            logger.info(f"  Processed {idx + 1}/{total}")


def write_yaml(out_dir: Path, src_yaml: Path) -> Path:
    """Read class info from source YAML and write a new one for the resized dataset."""
    import yaml
    with open(src_yaml) as f:
        cfg = yaml.safe_load(f)

    new_cfg = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": cfg["nc"],
        "names": cfg["names"],
    }

    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(new_cfg, f, default_flow_style=False, sort_keys=False)

    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="Resize a YOLO dataset with explicit bbox recalculation",
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Root of YOLO dataset (must contain images/ and labels/)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write resized dataset")
    parser.add_argument("--max-dim", type=int, default=640,
                        help="Longest side of resized images (default: 640)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG save quality (default: 95)")
    parser.add_argument("--src-yaml", type=Path, default=None,
                        help="Source dataset YAML (for class names). Auto-detected if omitted.")
    args = parser.parse_args()

    inp = args.input_dir
    out = args.output_dir

    logger.info(f"Input:   {inp.resolve()}")
    logger.info(f"Output:  {out.resolve()}")
    logger.info(f"Max dim: {args.max_dim}, JPEG quality: {args.quality}")

    src_yaml = args.src_yaml or (inp / "data.yaml")
    if not src_yaml.exists():
        logger.error(f"Source YAML not found: {src_yaml}")
        sys.exit(1)

    for split in ("train", "val"):
        img_in = inp / "images" / split
        lbl_in = inp / "labels" / split
        if not img_in.exists():
            logger.warning(f"Split '{split}' not found at {img_in}, skipping")
            continue

        logger.info(f"Processing split: {split}")
        stats: dict = {
            "images": 0, "bboxes": 0,
            "skipped_images": 0, "missing_labels": 0,
            "skipped_bboxes": 0, "dropped_tiny": 0,
        }

        process_split(
            img_in, lbl_in,
            out / "images" / split, out / "labels" / split,
            args.max_dim, args.quality, stats,
        )

        logger.info(f"  {split} stats: {json.dumps(stats)}")

    yaml_path = write_yaml(out, src_yaml)
    logger.info(f"Dataset YAML written to {yaml_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
