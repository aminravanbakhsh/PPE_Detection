"""Unified training script for baseline and extended PPE detection models."""

import argparse
import atexit
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

# Repo root on path, then TAL patches **before** ultralytics loads the detection loss / assigner.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import training.tal_patch  # noqa: E402 — side effect: patches TaskAlignedAssigner

from ultralytics import YOLO
from ultralytics.utils.tal import TaskAlignedAssigner  # noqa: E402

from training.log_utils import setup_logger  # noqa: E402


def _make_fractional_val_config(config_path: str, fraction: float, seed: int = 42) -> str:
    """Create a temp dataset YAML whose val split is a symlinked subset of the original.

    Ultralytics' ``fraction`` param only affects training data.  This function
    creates a temporary ``val_subset`` directory with symlinks to *fraction* of
    the original val images/labels so that validation is equally fast.

    Returns the path to the temporary YAML (cleaned up via atexit).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg["path"])
    val_rel = cfg.get("val", "images/val")
    val_img_dir = dataset_root / val_rel
    val_lbl_dir = dataset_root / val_rel.replace("images", "labels")

    images = sorted(
        p for p in val_img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    n_subset = max(1, int(len(images) * fraction))
    rng = random.Random(seed)
    subset = rng.sample(images, n_subset)

    subset_img = dataset_root / "images" / "val_subset"
    subset_lbl = dataset_root / "labels" / "val_subset"
    for d in (subset_img, subset_lbl):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for img in subset:
        (subset_img / img.name).symlink_to(img.resolve())
        lbl = val_lbl_dir / f"{img.stem}.txt"
        if lbl.exists():
            (subset_lbl / lbl.name).symlink_to(lbl.resolve())

    cfg["val"] = "images/val_subset"
    tmp_yaml = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="ppe_smoke_", delete=False,
    )
    yaml.dump(cfg, tmp_yaml, default_flow_style=False, sort_keys=False)
    tmp_yaml.close()

    def _cleanup():
        shutil.rmtree(subset_img, ignore_errors=True)
        shutil.rmtree(subset_lbl, ignore_errors=True)
        try:
            os.unlink(tmp_yaml.name)
        except OSError:
            pass
        cache = subset_img.parent / "val_subset.cache"
        cache.unlink(missing_ok=True)

    atexit.register(_cleanup)
    return tmp_yaml.name


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
    parser.add_argument("--project", type=str, default="models",
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
        help="Disable AMP (mixed precision). Recommended on MPS to avoid TAL issues.",
    )
    parser.add_argument("--max-det", type=int, default=100,
                        help="Max detections per image for NMS (lower = faster val). Default 100.")
    parser.add_argument("--val-period", type=int, default=5,
                        help="Run validation every N epochs (and always on the final epoch). Default 5.")
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Freeze first N layers (0=none, 10=backbone, 22=all-but-head). Default 10.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a run from last.pt (pass that path as --model). Total --epochs should be the "
        "desired final epoch count (e.g. 20 to add 10 epochs after a finished 10-epoch run).",
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
    logger.info(
        f"Fraction: {args.fraction}, Val: {not args.no_val}, AMP: {not args.no_amp}, "
        f"MaxDet: {args.max_det}, ValPeriod: {args.val_period}, Freeze: {args.freeze}"
    )

    abs_project = os.path.abspath(args.project)

    model = YOLO(args.model)

    cache_arg = False if args.cache == "false" else args.cache
    use_amp = not args.no_amp
    if args.device == "mps" and use_amp:
        logger.warning("AMP on MPS can cause index-corruption in TAL; consider --no-amp if training crashes")

    val_period = args.val_period
    total_epochs = args.epochs

    def _on_train_epoch_end(trainer):
        """Toggle trainer.args.val so validation only runs every val_period epochs (and the final epoch)."""
        epoch = trainer.epoch + 1  # 0-indexed internally
        is_val_epoch = (epoch % val_period == 0) or (epoch >= total_epochs)
        trainer.args.val = is_val_epoch

    if not args.no_val and val_period > 1:
        model.add_callback("on_train_epoch_end", _on_train_epoch_end)
        logger.info(f"Validation will run every {val_period} epochs (and the final epoch)")

    training.tal_patch.apply_tal_patches()
    if TaskAlignedAssigner.get_box_metrics.__name__ != "_fixed_get_box_metrics":
        raise RuntimeError(
            "TaskAlignedAssigner is not patched. Run: python training/train.py ... from repo root "
            "(not `yolo train`, which skips training/tal_patch.py)."
        )
    logger.info("TAL patch active: TaskAlignedAssigner.get_box_metrics=%s", TaskAlignedAssigner.get_box_metrics.__name__)

    data_config = args.config
    if args.fraction < 1.0 and not args.no_val:
        data_config = _make_fractional_val_config(args.config, args.fraction)
        logger.info(f"Validation subset created ({args.fraction*100:.0f}%% of val data) -> {data_config}")

    results = model.train(
        data=data_config,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=abs_project,
        name=args.name,
        workers=args.workers,
        lr0=args.lr0,
        cos_lr=args.cos_lr,
        cache=cache_arg,
        fraction=args.fraction,
        val=not args.no_val,
        amp=use_amp,
        max_det=args.max_det,
        freeze=args.freeze,
        mosaic=0.0,
        close_mosaic=0,
        exist_ok=True,
        verbose=True,
        resume=args.resume,
    )

    logger.info("Training complete.")
    if results and hasattr(results, "results_dict"):
        for k, v in results.results_dict.items():
            logger.info(f"  {k}: {v:.4f}")

    save_dir = getattr(results, "save_dir", None) if results else None
    if save_dir:
        best_path = os.path.join(save_dir, "weights", "best.pt")
    else:
        best_path = os.path.join(args.project, args.name, "weights", "best.pt")

    if os.path.exists(best_path):
        logger.info(f"Best weights saved to: {best_path}")
    else:
        logger.warning("best.pt not found — check training output")


if __name__ == "__main__":
    main()
