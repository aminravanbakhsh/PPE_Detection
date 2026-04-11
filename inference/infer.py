"""Flexible YOLO inference / evaluation CLI.

Run any YOLO checkpoint against any Ultralytics-format dataset and produce a
comprehensive metrics report (console table + JSON file).

Usage examples
--------------
# Baseline model on baseline val set
python -m inference.infer \
    --model models/baseline/weights/best.pt \
    --data  training/configs/sh17_baseline_resized.yaml \
    --output results/baseline/metrics.json --name baseline_eval

# Extended model on extended val set
python -m inference.infer \
    --model models/extended/weights/best.pt \
    --data  training/configs/sh17_extended_resized.yaml \
    --output results/extended/metrics.json --name extended_eval

For saved prediction images on val frames with SH17 PPE and/or Face-shield labels
(not Person-only), see ``python -m inference.sample_predictions --help``.
"""

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ultralytics import YOLO

from inference.metrics import extract_metrics, print_summary, save_report
from training.log_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO inference/evaluation on a dataset and report detection metrics."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to YOLO .pt weights")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to Ultralytics dataset YAML config")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device: mps, cpu, or cuda id (default: mps)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="IoU threshold for NMS (default: 0.7)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="Dataset split to evaluate (default: val)")
    parser.add_argument("--output", type=str, default="results/metrics.json",
                        help="Path for JSON results file (default: results/metrics.json)")
    parser.add_argument("--project", type=str, default="results",
                        help="Directory for Ultralytics output artifacts (default: results)")
    parser.add_argument("--name", type=str, default="run",
                        help="Run name (default: run)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger = setup_logger("inference", "05_inference.log")

    logger.info(f"Model   : {args.model}")
    logger.info(f"Dataset : {args.data}")
    logger.info(f"Split   : {args.split}")
    logger.info(f"Device  : {args.device}")
    logger.info(f"ImgSize : {args.imgsz}")
    logger.info(f"Conf    : {args.conf}, IoU: {args.iou}")
    logger.info(f"Batch   : {args.batch}")

    if not os.path.isfile(args.model):
        logger.error(f"Model weights not found: {args.model}")
        sys.exit(1)
    if not os.path.isfile(args.data):
        logger.error(f"Dataset config not found: {args.data}")
        sys.exit(1)

    model = YOLO(args.model)
    logger.info(f"Loaded model with {len(model.names)} classes: {list(model.names.values())}")

    abs_project = os.path.abspath(args.project)

    logger.info("Running validation …")
    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch=args.batch,
        device=args.device,
        project=abs_project,
        name=args.name,
        exist_ok=True,
        verbose=False,
    )

    run_config = {
        "model": args.model,
        "dataset": args.data,
        "split": args.split,
        "image_size": args.imgsz,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    metrics = extract_metrics(results, model, config=run_config)
    print_summary(metrics)
    save_report(metrics, args.output)

    logger.info(f"mAP@0.5={metrics['overall']['mAP50']:.4f}  "
                f"mAP@0.5:0.95={metrics['overall']['mAP50_95']:.4f}  "
                f"P={metrics['overall']['precision']:.4f}  "
                f"R={metrics['overall']['recall']:.4f}  "
                f"F1={metrics['overall']['f1']:.4f}")
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
