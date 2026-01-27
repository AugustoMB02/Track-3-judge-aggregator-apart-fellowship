#!/usr/bin/env python3
"""
Create a single summary plot for HelpSteer2 dimension experiments.

Outputs a two-panel figure with test R2 and best validation R2 for:
- Parent-only (single judge)
- Children-all (5 judges)
- Children-pruned (3 judges)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt


DIMENSIONS = ["coherence", "complexity", "correctness", "helpfulness", "verbosity"]


def _load_summary(path: Path) -> Tuple[float, float]:
    data = json.loads(path.read_text())
    test_r2 = float(data.get("final_r2", 0.0))
    best_val_r2 = float(data.get("best_r2", 0.0))
    return test_r2, best_val_r2


def _find_latest_timestamp(results_root: Path) -> str:
    candidates = sorted(results_root.glob("helpsteer2_*_parent_*"))
    if not candidates:
        raise FileNotFoundError(f"No runs found in {results_root}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.name.split("_")[-2] + "_" + latest.name.split("_")[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HelpSteer2 dimension summary")
    parser.add_argument(
        "--timestamp",
        type=str,
        default="",
        help="Run timestamp (e.g., 20260126_112545). Defaults to latest.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/track3_full_dataset",
        help="Results root directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output path (without extension). Defaults to results root.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    timestamp = args.timestamp or _find_latest_timestamp(results_root)

    series: Dict[str, Dict[str, float]] = {
        "parent": {},
        "children_all": {},
        "children_prune": {},
    }
    val_series: Dict[str, Dict[str, float]] = {
        "parent": {},
        "children_all": {},
        "children_prune": {},
    }

    for dim in DIMENSIONS:
        parent = results_root / f"helpsteer2_{dim}_parent_{timestamp}" / "summary.json"
        children_all = results_root / f"helpsteer2_{dim}_children_all_{timestamp}" / "summary.json"
        children_prune = results_root / f"helpsteer2_{dim}_children_prune_{timestamp}" / "summary.json"

        if not parent.exists() or not children_all.exists() or not children_prune.exists():
            raise FileNotFoundError(f"Missing summaries for {dim} at {timestamp}")

        p_test, p_val = _load_summary(parent)
        c_test, c_val = _load_summary(children_all)
        pr_test, pr_val = _load_summary(children_prune)

        series["parent"][dim] = p_test
        series["children_all"][dim] = c_test
        series["children_prune"][dim] = pr_test
        val_series["parent"][dim] = p_val
        val_series["children_all"][dim] = c_val
        val_series["children_prune"][dim] = pr_val

    x = range(len(DIMENSIONS))
    width = 0.25

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("HelpSteer2 Dimension Results (test=0.2, val=0.1)")

    axes[0].bar([i - width for i in x], [series["parent"][d] for d in DIMENSIONS], width, label="Parent only")
    axes[0].bar(x, [series["children_all"][d] for d in DIMENSIONS], width, label="Children all")
    axes[0].bar([i + width for i in x], [series["children_prune"][d] for d in DIMENSIONS], width, label="Children pruned")
    axes[0].set_ylabel("Test R2")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].bar([i - width for i in x], [val_series["parent"][d] for d in DIMENSIONS], width, label="Parent only")
    axes[1].bar(x, [val_series["children_all"][d] for d in DIMENSIONS], width, label="Children all")
    axes[1].bar([i + width for i in x], [val_series["children_prune"][d] for d in DIMENSIONS], width, label="Children pruned")
    axes[1].set_ylabel("Best Validation R2")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(DIMENSIONS, rotation=20, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if args.output:
        output_base = Path(args.output)
    else:
        output_base = results_root / f"helpsteer2_dimension_summary_{timestamp}"

    fig.savefig(str(output_base) + ".png", dpi=200)
    fig.savefig(str(output_base) + ".pdf")
    plt.close(fig)

    print(f"Saved {output_base}.png and {output_base}.pdf")


if __name__ == "__main__":
    main()
