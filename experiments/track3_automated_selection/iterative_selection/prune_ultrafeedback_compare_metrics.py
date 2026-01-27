#!/usr/bin/env python3
"""
Compare pruning metrics for the UltraFeedback full dataset.

Runs the iterative selection pipeline multiple times with different pruning
strategies, then summarizes which strategy performs best.

Strategies 1-5:
1. importance - Remove judge with lowest (combined) importance score
2. redundancy - Remove judge with highest mean pairwise score correlation
3. attribution_correlation - Remove judge from most-correlated attribution pair
4. human_correlation - Remove judge with lowest correlation to targets
5. combined - Remove judge with lowest importance × (1 - redundancy)

Usage:
    PYTHONPATH=. .venv/bin/python \
      experiments/track3_automated_selection/iterative_selection/prune_ultrafeedback_compare_metrics.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

from experiments.track3_automated_selection.iterative_selection.iterative_selection import (
    IterativeJudgeSelector,
    SelectionConfig,
)


# Strategies 1-5 requested by the user.
STRATEGIES_1_TO_5 = [
    "importance",
    "redundancy",
    "attribution_correlation",
    "human_correlation",
    "combined",
]


def _detect_current_strategy(default: str = "importance") -> str:
    """
    Detect the current pruning strategy from the most recent UltraFeedback run.

    Falls back to the provided default if no previous run is found.
    """
    runs_root = Path("results/track3_full_dataset")
    candidates = sorted(
        runs_root.glob("ultrafeedback_prune_half_*/config.yaml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return default

    try:
        data = yaml.safe_load(candidates[0].read_text()) or {}
        return data.get("pruning_strategy") or default
    except Exception:
        return default


def _load_iteration_metrics(selection_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """Load iteration 0, final iteration, and summary JSONs."""
    summary = json.loads((selection_dir / "summary.json").read_text())
    iter0 = json.loads((selection_dir / "iteration_00" / "result.json").read_text())
    final_idx = summary["total_iterations"] - 1
    final_iter = json.loads(
        (selection_dir / f"iteration_{final_idx:02d}" / "result.json").read_text()
    )
    return iter0, final_iter, summary


def _removed_judges_from_summary(summary: Dict) -> List[str]:
    removed = []
    for row in summary.get("iterations", []):
        judge = row.get("removed")
        if judge:
            removed.append(judge)
    return removed


def run_strategy(
    base_config: SelectionConfig,
    strategy: str,
    output_root: Path,
    label: str | None = None,
) -> Dict:
    """Run a single strategy and return a compact metrics summary."""
    label = label or strategy
    selection_dir = output_root / label / "selection"

    config = replace(
        base_config,
        name=f"{base_config.name}-{label}",
        description=f"{base_config.description} [strategy={strategy}]",
        pruning_strategy=strategy,
        output_dir=str(selection_dir),
    )

    selector = IterativeJudgeSelector(config)
    selector.run()

    iter0, final_iter, summary = _load_iteration_metrics(selection_dir)
    removed = _removed_judges_from_summary(summary)

    iter0_test = iter0.get("test_metrics", {})
    final_test = final_iter.get("test_metrics", {})

    iter0_r2 = float(iter0_test.get("r2", 0.0) or 0.0)
    final_r2 = float(final_test.get("r2", 0.0) or 0.0)
    iter0_mae = float(iter0_test.get("mae", 0.0) or 0.0)
    final_mae = float(final_test.get("mae", 0.0) or 0.0)

    return {
        "label": label,
        "strategy": strategy,
        "iter0_r2": iter0_r2,
        "iter0_mae": iter0_mae,
        "final_r2": final_r2,
        "final_mae": final_mae,
        "delta_r2": final_r2 - iter0_r2,
        "delta_mae": final_mae - iter0_mae,
        "final_judge_count": summary.get("final_n_judges"),
        "total_iterations": summary.get("total_iterations"),
        "removed_judges": ", ".join(removed),
        "selection_dir": str(selection_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare pruning strategies on UltraFeedback full dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/selection_experiment_prune_half_ultrafeedback.yaml",
        help="Base SelectionConfig YAML to clone for each strategy",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=STRATEGIES_1_TO_5,
        choices=STRATEGIES_1_TO_5,
        help="Strategies 1-5 to run (default: all 1-5)",
    )
    parser.add_argument(
        "--include-current",
        action="store_true",
        help="Also run the current strategy from the latest UF prune-half run",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root output directory (default: timestamped under results/)",
    )
    parser.add_argument(
        "--min-judges",
        type=int,
        default=None,
        help="Override SelectionConfig.min_judges",
    )
    parser.add_argument(
        "--target-judges",
        type=int,
        default=None,
        help="Override SelectionConfig.target_judges",
    )
    parser.add_argument(
        "--max-judges",
        type=int,
        default=None,
        help="Override SelectionConfig.max_judges",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override SelectionConfig.max_iterations",
    )
    parser.add_argument(
        "--clear-protected",
        action="store_true",
        help="Clear protected_judges so parents can be removed",
    )
    args = parser.parse_args()

    base_config = SelectionConfig.from_yaml(args.config)
    if args.min_judges is not None:
        base_config.min_judges = args.min_judges
    if args.target_judges is not None:
        base_config.target_judges = args.target_judges
    if args.max_judges is not None:
        base_config.max_judges = args.max_judges
    if args.max_iterations is not None:
        base_config.max_iterations = args.max_iterations
    if args.clear_protected:
        base_config.protected_judges = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root)
        if args.output_root
        else Path("results") / f"ultrafeedback_pruning_metric_compare_{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    current_strategy = _detect_current_strategy(default="importance")
    current_label = f"current_{current_strategy}"

    # Build explicit (strategy, label) runs so we can include the current
    # strategy even if it duplicates one of 1-5.
    runs: List[Tuple[str, str]] = [(s, s) for s in args.strategies]
    if args.include_current:
        runs.append((current_strategy, current_label))

    rows: List[Dict] = []

    for strategy, label in runs:
        print(f"\n=== Running strategy: {strategy} (label={label}) ===")
        row = run_strategy(base_config, strategy=strategy, output_root=output_root, label=label)
        rows.append(row)
        print(
            f"Done: {label} | final_r2={row['final_r2']:.4f} "
            f"| final_mae={row['final_mae']:.4f} | iters={row['total_iterations']}"
        )

    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(["final_r2", "delta_r2"], ascending=[False, False])

    comparison_json = output_root / "comparison_summary.json"
    comparison_csv = output_root / "comparison_summary.csv"
    df_sorted.to_json(comparison_json, orient="records", indent=2)
    df_sorted.to_csv(comparison_csv, index=False)

    print("\n=== Strategy Comparison (sorted by final_r2) ===")
    display_cols = [
        "label",
        "strategy",
        "iter0_r2",
        "final_r2",
        "delta_r2",
        "iter0_mae",
        "final_mae",
        "delta_mae",
        "final_judge_count",
        "total_iterations",
        "selection_dir",
    ]
    print(df_sorted[display_cols].to_string(index=False))
    print(f"\nWrote: {comparison_json}")
    print(f"Wrote: {comparison_csv}")


if __name__ == "__main__":
    main()
