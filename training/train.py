"""Unified training script for baseline and extended PPE detection models."""

import argparse
import os
import sys

from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.log_utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for PPE Detection")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to dataset YAML config")
    parser.add_argument("--model", type=str, default="yolov8m.pt",
                        help="Pretrained model checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="mps",
                        help="Training device: mps, cpu, or cuda device id")
    parser.add_argument("--name", type=str, default="run",
                        help="Run name (e.g. 'baseline' or 'extended')")
    parser.add_argument("--project", type=str, default="runs",
                        help="Project directory for outputs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--cos-lr", action="store_true", default=True,
                        help="Use cosine LR schedule")
    parser.add_argument(
        "--cache",
        type=str,
        default="false",
        choices=["false", "ram", "disk"],
        help="Dataset cache mode. Use 'false' to avoid stale/corrupt .cache files (recommended on mps).",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to use for quick smoke tests (0.0-1.0).",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Disable validation (avoids NMS-timeout issues during smoke tests).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP (mixed precision). Required on MPS to avoid index-corruption bugs in TAL.",
    )
    args = parser.parse_args()

    log_file = "02_baseline_training.log" if "baseline" in args.name else "03_extended_training.log"
    logger = setup_logger(f"train_{args.name}", log_file)

    logger.info(f"Config: {args.config}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}, Batch: {args.batch}, ImgSz: {args.imgsz}")
    logger.info(f"Device: {args.device}, Workers: {args.workers}")
    logger.info(f"LR: {args.lr0}, Cosine LR: {args.cos_lr}")
    logger.info(f"Cache: {args.cache}")
    logger.info(f"Fraction: {args.fraction}, Val: {not args.no_val}, AMP: {not args.no_amp}")

    model = YOLO(args.model)

    cache_arg = False if args.cache == "false" else args.cache
    use_amp = not args.no_amp
    if args.device == "mps" and use_amp:
        logger.warning("AMP on MPS can cause index-corruption in TAL; consider --no-amp if training crashes")

    results = model.train(
        data=args.config,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        lr0=args.lr0,
        cos_lr=args.cos_lr,
        cache=cache_arg,
        fraction=args.fraction,
        val=not args.no_val,
        amp=use_amp,
        exist_ok=True,
        verbose=True,
    )

    logger.info("Training complete.")
    if results and hasattr(results, "results_dict"):
        for k, v in results.results_dict.items():
            logger.info(f"  {k}: {v:.4f}")

    best_path = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.exists(best_path):
        logger.info(f"Best weights saved to: {best_path}")
    else:
        logger.warning("best.pt not found — check training output")


if __name__ == "__main__":
    main()
