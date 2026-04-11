"""Compare baseline vs extended inference results against rubric criteria.

Reads two JSON reports produced by ``inference/infer.py`` and checks:
  1. Face-shield mAP@0.5 >= target (default 0.65)
  2. No existing class regresses by more than the threshold (default 3 pp)

Usage
-----
python -m inference.compare \
    --baseline results_baseline.json \
    --extended results_extended.json \
    --output comparison_results.json
"""

import argparse
import json
import os
import sys
from typing import Any


FACE_SHIELD_CLASS = "Face-shield"


def load_ap50_map(path: str) -> dict[str, float]:
    """Return ``{class_name: ap50}`` from an ``infer.py`` JSON report."""
    with open(path) as f:
        data = json.load(f)
    return {entry["class"]: entry.get("ap50", 0.0) for entry in data.get("per_class", [])}


def compare(
    baseline_map: dict[str, float],
    extended_map: dict[str, float],
    face_shield_target: float,
    regression_threshold: float,
) -> dict[str, Any]:
    """Run the two rubric checks and return a structured report."""
    all_classes = list(baseline_map.keys())
    new_classes = [c for c in extended_map if c not in baseline_map]
    all_classes.extend(new_classes)

    per_class: list[dict[str, Any]] = []
    regression_failures: list[str] = []

    for cls in all_classes:
        b_val = baseline_map.get(cls)
        e_val = extended_map.get(cls)

        entry: dict[str, Any] = {
            "class": cls,
            "baseline_ap50": b_val,
            "extended_ap50": e_val,
        }

        if b_val is not None and e_val is not None:
            diff = e_val - b_val
            entry["diff"] = round(diff, 4)
            entry["status"] = "PASS" if diff >= -regression_threshold else "FAIL"
            if entry["status"] == "FAIL":
                regression_failures.append(cls)
        elif cls == FACE_SHIELD_CLASS and e_val is not None:
            entry["diff"] = None
            entry["status"] = "PASS" if e_val >= face_shield_target else "FAIL"
        else:
            entry["diff"] = None
            entry["status"] = "---"

        per_class.append(entry)

    face_shield_ap50 = extended_map.get(FACE_SHIELD_CLASS, 0.0)
    face_shield_pass = face_shield_ap50 >= face_shield_target
    overall = "PASS" if (face_shield_pass and not regression_failures) else "FAIL"

    return {
        "per_class_comparison": per_class,
        "face_shield_ap50": face_shield_ap50,
        "face_shield_pass": face_shield_pass,
        "regression_failures": regression_failures,
        "overall": overall,
    }


def print_comparison(
    report: dict[str, Any],
    baseline_path: str,
    extended_path: str,
    face_shield_target: float,
    regression_threshold: float,
) -> None:
    """Print a presentation-ready comparison table."""
    print(f"\n{'=' * 75}")
    print("  Baseline vs Extended — Per-Class mAP@0.5 Comparison")
    print(f"{'=' * 75}")
    print(f"  Baseline : {baseline_path}")
    print(f"  Extended : {extended_path}")
    print(f"  Regression threshold : {regression_threshold * 100:.0f} pp")
    print(f"  Face-shield target   : {face_shield_target}")
    print(f"{'=' * 75}")

    header = f"  {'Class':<22} {'Baseline':>10} {'Extended':>10} {'Diff':>10} {'Status':>8}"
    print(header)
    print(f"  {'─' * 20:<22} {'─' * 8:>10} {'─' * 8:>10} {'─' * 8:>10} {'─' * 6:>8}")

    for entry in report["per_class_comparison"]:
        cls = entry["class"]
        b = entry["baseline_ap50"]
        e = entry["extended_ap50"]
        diff = entry["diff"]
        status = entry["status"]

        b_str = f"{b:.4f}" if b is not None else "NEW"
        e_str = f"{e:.4f}" if e is not None else "N/A"
        d_str = f"{diff:+.4f}" if diff is not None else "---"

        print(f"  {cls:<22} {b_str:>10} {e_str:>10} {d_str:>10} {status:>8}")

    fs_ap = report["face_shield_ap50"]
    fs_pass = report["face_shield_pass"]
    failures = report["regression_failures"]
    overall = report["overall"]

    print(f"\n{'─' * 75}")
    print(f"  Face-shield mAP@0.5 : {fs_ap:.4f}  (target >= {face_shield_target})  => {'PASS' if fs_pass else 'FAIL'}")
    if failures:
        print(f"  Regression failures (> {regression_threshold * 100:.0f}pp drop): {failures}")
    else:
        print("  Regression failures : None")
    print(f"\n  Overall verdict: {overall}")
    print(f"{'─' * 75}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs extended model inference results."
    )
    parser.add_argument("--baseline", type=str, required=True,
                        help="Path to baseline inference JSON (from infer.py)")
    parser.add_argument("--extended", type=str, required=True,
                        help="Path to extended inference JSON (from infer.py)")
    parser.add_argument("--face-shield-target", type=float, default=0.65,
                        help="Min mAP@0.5 for Face-shield class (default: 0.65)")
    parser.add_argument("--regression-threshold", type=float, default=0.03,
                        help="Max allowed per-class AP50 drop in absolute points (default: 0.03)")
    parser.add_argument("--output", type=str, default="results/comparison/comparison.json",
                        help="Path for comparison report JSON (default: results/comparison/comparison.json)")
    args = parser.parse_args()

    baseline_map = load_ap50_map(args.baseline)
    extended_map = load_ap50_map(args.extended)

    report = compare(baseline_map, extended_map, args.face_shield_target, args.regression_threshold)

    report_with_meta = {
        "baseline_file": args.baseline,
        "extended_file": args.extended,
        "face_shield_target": args.face_shield_target,
        "regression_threshold": args.regression_threshold,
        **report,
    }

    print_comparison(report, args.baseline, args.extended, args.face_shield_target, args.regression_threshold)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report_with_meta, f, indent=2)
    print(f"  Comparison saved to {args.output}\n")

    sys.exit(0 if report["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
