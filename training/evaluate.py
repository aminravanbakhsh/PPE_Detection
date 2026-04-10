"""Compare baseline vs extended model: per-class mAP50, regression check, face shield target."""

import argparse
import json
import os
import sys

from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.log_utils import setup_logger

logger = setup_logger("evaluate", "04_evaluation.log")

FACE_SHIELD_IDX = 17
FACE_SHIELD_TARGET = 0.65
REGRESSION_THRESHOLD = 0.03  # 3 percentage points


def get_per_class_map50(model: YOLO, data_yaml: str, device: str) -> dict:
    """Run validation and return {class_name: mAP50}."""
    results = model.val(data=data_yaml, device=device, verbose=False)
    names = model.names
    per_class = {}
    if hasattr(results, "box") and hasattr(results.box, "maps"):
        maps = results.box.maps
        for i, m in enumerate(maps):
            if i in names:
                per_class[names[i]] = float(m)
    return per_class, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline vs extended model")
    parser.add_argument("--baseline", type=str, required=True,
                        help="Path to baseline best.pt")
    parser.add_argument("--extended", type=str, required=True,
                        help="Path to extended best.pt")
    parser.add_argument("--data", type=str, default="data/merged/data.yaml",
                        help="Validation data YAML")
    parser.add_argument("--baseline-data", type=str, default=None,
                        help="Baseline data YAML (if different from extended)")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    args = parser.parse_args()

    baseline_data = args.baseline_data or args.data

    logger.info(f"Baseline model: {args.baseline}")
    logger.info(f"Extended model: {args.extended}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Device: {args.device}")

    # Evaluate baseline
    logger.info("Evaluating baseline model...")
    baseline_model = YOLO(args.baseline)
    baseline_map, baseline_results = get_per_class_map50(baseline_model, baseline_data, args.device)

    # Evaluate extended
    logger.info("Evaluating extended model...")
    extended_model = YOLO(args.extended)
    extended_map, extended_results = get_per_class_map50(extended_model, args.data, args.device)

    # Comparison table
    logger.info("")
    logger.info(f"{'Class':<20} {'Baseline':>10} {'Extended':>10} {'Diff':>10} {'Status':>10}")
    logger.info("-" * 65)

    all_pass = True
    regression_failures = []

    all_classes = sorted(set(list(baseline_map.keys()) + list(extended_map.keys())))
    for cls in all_classes:
        b_val = baseline_map.get(cls, None)
        e_val = extended_map.get(cls, None)

        b_str = f"{b_val:.4f}" if b_val is not None else "N/A"
        e_str = f"{e_val:.4f}" if e_val is not None else "N/A"

        if b_val is not None and e_val is not None:
            diff = e_val - b_val
            status = "PASS" if diff >= -REGRESSION_THRESHOLD else "FAIL"
            if status == "FAIL":
                all_pass = False
                regression_failures.append(cls)
            logger.info(f"{cls:<20} {b_str:>10} {e_str:>10} {diff:>+10.4f} {status:>10}")
        elif cls == "Face-shield" and e_val is not None:
            status = "PASS" if e_val >= FACE_SHIELD_TARGET else "FAIL"
            if status == "FAIL":
                all_pass = False
            logger.info(f"{cls:<20} {'NEW':>10} {e_str:>10} {'---':>10} {status:>10}")
        else:
            logger.info(f"{cls:<20} {b_str:>10} {e_str:>10} {'---':>10} {'---':>10}")

    # Face shield specific check
    face_shield_map = extended_map.get("Face-shield", 0.0)
    face_shield_pass = face_shield_map >= FACE_SHIELD_TARGET

    logger.info("")
    logger.info(f"Face-shield mAP@0.5: {face_shield_map:.4f} (target >= {FACE_SHIELD_TARGET}) "
                f"=> {'PASS' if face_shield_pass else 'FAIL'}")
    if regression_failures:
        logger.info(f"Regression failures (> {REGRESSION_THRESHOLD*100:.0f}pp drop): {regression_failures}")
    else:
        logger.info("No class regressions detected.")

    overall = "PASS" if (all_pass and face_shield_pass) else "FAIL"
    logger.info(f"\nOverall verdict: {overall}")

    report = {
        "baseline_per_class_map50": baseline_map,
        "extended_per_class_map50": extended_map,
        "face_shield_map50": face_shield_map,
        "face_shield_pass": face_shield_pass,
        "regression_failures": regression_failures,
        "overall": overall,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Results saved to {args.output}")

    sys.exit(0 if overall == "PASS" else 1)


if __name__ == "__main__":
    main()
