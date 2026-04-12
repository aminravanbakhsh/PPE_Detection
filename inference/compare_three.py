"""Compare baseline vs extended (10 ep) vs extended (20 ep) inference JSONs.

Extends the rubric in ``inference/compare.py`` to three checkpoints: baseline
remains the reference for regression; both extended runs are checked against it.
Face-shield targets apply to each extended model separately.

Usage
-----
python -m inference.compare_three \\
    --baseline results/baseline/metrics_finetuned.json \\
    --extended-10ep results/extended/metrics_finetuned.json \\
    --extended-20ep results/extended_20ep/metrics_finetuned.json \\
    --output results/comparison/rubric_three.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from inference.compare import compare, load_ap50_map


def _load_overall_metrics(path: str) -> dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    overall = data.get("overall") or {}
    return {k: overall.get(k) for k in ("mAP50", "mAP50_95", "precision", "recall", "f1")}


def compare_three(
    baseline_map: dict[str, float],
    map_10: dict[str, float],
    map_20: dict[str, float],
    face_shield_target: float,
    regression_threshold: float,
) -> dict[str, Any]:
    """Merge per-class rows from two `compare()` reports plus deltas 20 vs 10."""
    r10 = compare(baseline_map, map_10, face_shield_target, regression_threshold)
    r20 = compare(baseline_map, map_20, face_shield_target, regression_threshold)

    by_class: dict[str, dict[str, Any]] = {}
    for entry in r10["per_class_comparison"]:
        by_class[entry["class"]] = {
            "class": entry["class"],
            "baseline_ap50": entry["baseline_ap50"],
            "extended_10ep_ap50": entry["extended_ap50"],
            "diff_10ep_vs_baseline": entry.get("diff"),
            "status_10ep_vs_baseline": entry["status"],
            "extended_20ep_ap50": None,
            "diff_20ep_vs_baseline": None,
            "status_20ep_vs_baseline": "---",
            "diff_20ep_vs_10ep": None,
        }

    for entry in r20["per_class_comparison"]:
        cls = entry["class"]
        if cls not in by_class:
            by_class[cls] = {
                "class": cls,
                "baseline_ap50": entry["baseline_ap50"],
                "extended_10ep_ap50": None,
                "diff_10ep_vs_baseline": None,
                "status_10ep_vs_baseline": "---",
                "extended_20ep_ap50": None,
                "diff_20ep_vs_baseline": None,
                "status_20ep_vs_baseline": "---",
                "diff_20ep_vs_10ep": None,
            }
        row = by_class[cls]
        row["extended_20ep_ap50"] = entry["extended_ap50"]
        row["diff_20ep_vs_baseline"] = entry.get("diff")
        row["status_20ep_vs_baseline"] = entry["status"]

        v10 = row.get("extended_10ep_ap50")
        v20 = entry.get("extended_ap50")
        if v10 is not None and v20 is not None:
            d = round(v20 - v10, 4)
            row["diff_20ep_vs_10ep"] = d
        else:
            row["diff_20ep_vs_10ep"] = None

    per_class = sorted(by_class.values(), key=lambda x: x["class"])

    return {
        "per_class_comparison": per_class,
        "face_shield_ap50_10ep": r10["face_shield_ap50"],
        "face_shield_pass_10ep": r10["face_shield_pass"],
        "face_shield_ap50_20ep": r20["face_shield_ap50"],
        "face_shield_pass_20ep": r20["face_shield_pass"],
        "regression_failures_10ep": r10["regression_failures"],
        "regression_failures_20ep": r20["regression_failures"],
        "overall_10ep_vs_baseline": r10["overall"],
        "overall_20ep_vs_baseline": r20["overall"],
    }


def print_three(report: dict[str, Any], paths: tuple[str, str, str]) -> None:
    b, e10, e20 = paths
    print(f"\n{'=' * 90}")
    print("  Baseline vs Extended-10ep vs Extended-20ep — Per-Class mAP@0.5")
    print(f"{'=' * 90}")
    print(f"  Baseline      : {b}")
    print(f"  Extended 10ep : {e10}")
    print(f"  Extended 20ep : {e20}")
    print(f"{'=' * 90}")

    header = (
        f"  {'Class':<20} {'Base':>8} {'Ext10':>8} {'Ext20':>8} "
        f"{'d10':>8} {'d20':>8} {'20v10':>8} {'S10':>5} {'S20':>5}"
    )
    print(header)
    print(f"  {'─' * 18:<20} {'─' * 6:>8} {'─' * 6:>8} {'─' * 6:>8} {'─' * 6:>8} {'─' * 6:>8} {'─' * 6:>8} {'─' * 3:>5} {'─' * 3:>5}")

    for row in report["per_class_comparison"]:
        cls = row["class"]
        fmt = lambda x: f"{x:.4f}" if x is not None else "---"
        b_ap = row.get("baseline_ap50")
        t10 = row.get("extended_10ep_ap50")
        t20 = row.get("extended_20ep_ap50")
        d10 = row.get("diff_10ep_vs_baseline")
        d20 = row.get("diff_20ep_vs_baseline")
        d2010 = row.get("diff_20ep_vs_10ep")
        s10 = row.get("status_10ep_vs_baseline", "---")
        s20 = row.get("status_20ep_vs_baseline", "---")

        print(
            f"  {cls:<20} {fmt(b_ap):>8} {fmt(t10):>8} {fmt(t20):>8} "
            f"{(f'{d10:+.4f}' if d10 is not None else '---'):>8} "
            f"{(f'{d20:+.4f}' if d20 is not None else '---'):>8} "
            f"{(f'{d2010:+.4f}' if d2010 is not None else '---'):>8} "
            f"{s10:>5} {s20:>5}"
        )

    print(f"\n  Face-shield 10ep: {report['face_shield_ap50_10ep']:.4f} => "
          f"{'PASS' if report['face_shield_pass_10ep'] else 'FAIL'}")
    print(f"  Face-shield 20ep: {report['face_shield_ap50_20ep']:.4f} => "
          f"{'PASS' if report['face_shield_pass_20ep'] else 'FAIL'}")
    print(f"  Overall 10ep vs baseline : {report['overall_10ep_vs_baseline']}")
    print(f"  Overall 20ep vs baseline : {report['overall_20ep_vs_baseline']}")
    print(f"{'=' * 90}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-way rubric: baseline vs extended 10ep vs extended 20ep."
    )
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--extended-10ep", type=str, required=True)
    parser.add_argument("--extended-20ep", type=str, required=True)
    parser.add_argument("--face-shield-target", type=float, default=0.65)
    parser.add_argument("--regression-threshold", type=float, default=0.03)
    parser.add_argument(
        "--output",
        type=str,
        default="results/comparison/rubric_three.json",
    )
    args = parser.parse_args()

    baseline_map = load_ap50_map(args.baseline)
    map_10 = load_ap50_map(args.extended_10ep)
    map_20 = load_ap50_map(args.extended_20ep)

    report = compare_three(
        baseline_map,
        map_10,
        map_20,
        args.face_shield_target,
        args.regression_threshold,
    )

    out = {
        "baseline_file": args.baseline,
        "extended_10ep_file": args.extended_10ep,
        "extended_20ep_file": args.extended_20ep,
        "face_shield_target": args.face_shield_target,
        "regression_threshold": args.regression_threshold,
        "overall_metrics": {
            "baseline": _load_overall_metrics(args.baseline),
            "extended_10ep": _load_overall_metrics(args.extended_10ep),
            "extended_20ep": _load_overall_metrics(args.extended_20ep),
        },
        **report,
    }

    print_three(
        report,
        (args.baseline, args.extended_10ep, args.extended_20ep),
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved {args.output}\n")

    any_pass = (
        report["overall_10ep_vs_baseline"] == "PASS"
        or report["overall_20ep_vs_baseline"] == "PASS"
    )
    sys.exit(0 if any_pass else 1)


if __name__ == "__main__":
    main()
