"""Extract, format, and persist object-detection metrics from Ultralytics validation results."""

import json
from typing import Any

import numpy as np


def extract_metrics(results, model, *, config: dict | None = None) -> dict[str, Any]:
    """Pull every relevant metric from an Ultralytics DetMetrics result.

    Parameters
    ----------
    results : ultralytics.utils.metrics.DetMetrics
        Object returned by ``model.val()``.
    model : ultralytics.YOLO
        The model that was evaluated (used for class-name lookup).
    config : dict, optional
        Run metadata (model path, data path, thresholds, …) to embed in the
        report.

    Returns
    -------
    dict
        Nested dictionary ready for JSON serialisation.
    """
    names = model.names
    box = results.box

    overall = {
        "mAP50": float(box.map50),
        "mAP75": float(box.map75),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }

    f1_values = box.f1
    if isinstance(f1_values, np.ndarray) and f1_values.size > 0:
        overall["f1"] = float(f1_values.mean())
    else:
        overall["f1"] = 0.0

    per_class_p = box.p
    per_class_r = box.r
    per_class_f1 = f1_values
    per_class_ap50 = box.maps
    per_class_ap50_95 = box.ap50 if hasattr(box, "ap50") else per_class_ap50

    per_class: list[dict[str, Any]] = []
    for i, name in names.items():
        entry: dict[str, Any] = {"class": name}
        if isinstance(per_class_p, np.ndarray) and i < len(per_class_p):
            entry["precision"] = round(float(per_class_p[i]), 4)
        if isinstance(per_class_r, np.ndarray) and i < len(per_class_r):
            entry["recall"] = round(float(per_class_r[i]), 4)
        if isinstance(per_class_f1, np.ndarray) and i < len(per_class_f1):
            entry["f1"] = round(float(per_class_f1[i]), 4)
        if isinstance(per_class_ap50, np.ndarray) and i < len(per_class_ap50):
            entry["ap50"] = round(float(per_class_ap50[i]), 4)
        per_class.append(entry)

    speed = {}
    if hasattr(results, "speed") and results.speed:
        speed = {k: round(v, 2) for k, v in results.speed.items()}

    report: dict[str, Any] = {}
    if config:
        report["config"] = config
    report["overall"] = overall
    report["per_class"] = per_class
    report["speed"] = speed

    return report


def print_summary(metrics: dict[str, Any]) -> None:
    """Print a human-readable summary table to stdout."""
    if "config" in metrics:
        cfg = metrics["config"]
        print(f"\n{'=' * 80}")
        print(f"  Model   : {cfg.get('model', 'N/A')}")
        print(f"  Dataset : {cfg.get('dataset', 'N/A')}")
        print(f"  Split   : {cfg.get('split', 'N/A')}")
        print(f"  ImgSize : {cfg.get('image_size', 'N/A')}")
        print(f"  Conf    : {cfg.get('confidence_threshold', 'N/A')}")
        print(f"  IoU     : {cfg.get('iou_threshold', 'N/A')}")
        print(f"{'=' * 80}")

    overall = metrics.get("overall", {})
    print(f"\n{'─' * 50}")
    print("  Overall Metrics")
    print(f"{'─' * 50}")
    print(f"  mAP@0.5      : {overall.get('mAP50', 0):.4f}")
    print(f"  mAP@0.75     : {overall.get('mAP75', 0):.4f}")
    print(f"  mAP@0.5:0.95 : {overall.get('mAP50_95', 0):.4f}")
    print(f"  Precision     : {overall.get('precision', 0):.4f}")
    print(f"  Recall        : {overall.get('recall', 0):.4f}")
    print(f"  F1            : {overall.get('f1', 0):.4f}")

    per_class = metrics.get("per_class", [])
    if per_class:
        header = f"  {'Class':<20} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AP50':>8}"
        print(f"\n{'─' * len(header)}")
        print("  Per-Class Metrics")
        print(f"{'─' * len(header)}")
        print(header)
        print(f"  {'─' * 18:<20} {'─' * 6:>8} {'─' * 6:>8} {'─' * 6:>8} {'─' * 6:>8}")
        for cls in per_class:
            print(
                f"  {cls['class']:<20}"
                f" {cls.get('precision', 0):>8.4f}"
                f" {cls.get('recall', 0):>8.4f}"
                f" {cls.get('f1', 0):>8.4f}"
                f" {cls.get('ap50', 0):>8.4f}"
            )

    speed = metrics.get("speed", {})
    if speed:
        print(f"\n{'─' * 50}")
        print("  Speed (ms per image)")
        print(f"{'─' * 50}")
        for stage, ms in speed.items():
            print(f"  {stage:<20}: {ms:.2f}")

    print()


def save_report(metrics: dict[str, Any], path: str) -> None:
    """Write the full metrics dictionary to a JSON file."""
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
