#!/usr/bin/env python3
"""
Plot per-iteration score trajectories for each pruning strategy.

Produces line charts where x = iteration and y = metric, with one line per strategy.

Usage:
  PYTHONPATH=. .venv/bin/python \
    experiments/track3_automated_selection/iterative_selection/visualize_ultrafeedback_pruning_trajectories.py \
    --root results/ultrafeedback_pruning_metric_compare_20260127_093649
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt


def _load_strategy_trajectory(selection_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for iteration_dir in sorted(selection_dir.glob("iteration_*")):
        if not iteration_dir.is_dir():
            continue
        result_path = iteration_dir / "result.json"
        if not result_path.exists():
            continue
        data = json.loads(result_path.read_text())
        iteration = data.get("iteration")
        test = data.get("test_metrics") or {}
        val = data.get("val_metrics") or {}
        train = data.get("train_metrics") or {}
        rows.append(
            {
                "iteration": int(iteration) if iteration is not None else None,
                "test_r2": test.get("r2"),
                "test_mae": test.get("mae"),
                "val_r2": val.get("r2"),
                "val_mae": val.get("mae"),
                "train_r2": train.get("r2"),
                "train_mae": train.get("mae"),
                "n_judges": data.get("n_judges"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("iteration")
    return df


def _plot_metric(
    trajectories: Dict[str, pd.DataFrame],
    metric: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, df in trajectories.items():
        if df.empty or metric not in df:
            continue
        ax.plot(df["iteration"], df[metric], marker="o", linewidth=1.5, label=label)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-iteration metric trajectories for UF pruning strategies"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing strategy subfolders with selection/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write plots (default: root)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["test_r2", "test_mae"],
        help="Metrics to plot (columns from iteration results)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories: Dict[str, pd.DataFrame] = {}
    for strategy_dir in sorted(root.iterdir()):
        selection_dir = strategy_dir / "selection"
        if not selection_dir.is_dir():
            continue
        df = _load_strategy_trajectory(selection_dir)
        if not df.empty:
            trajectories[strategy_dir.name] = df

    if not trajectories:
        raise SystemExit(f"No selection runs found under: {root}")

    for metric in args.metrics:
        title = f"UltraFeedback Pruning: {metric} over iterations"
        out_path = output_dir / f"{metric}_trajectories.png"
        _plot_metric(trajectories, metric, title, out_path)

    # Write a combined CSV for convenience
    combined_rows = []
    for label, df in trajectories.items():
        for _, row in df.iterrows():
            combined_rows.append({"strategy": label, **row.to_dict()})
    combined = pd.DataFrame(combined_rows)
    combined.to_csv(output_dir / "trajectory_data.csv", index=False)

    print("Wrote trajectory plots and data to:")
    for metric in args.metrics:
        print(f"- {output_dir / f'{metric}_trajectories.png'}")
    print(f"- {output_dir / 'trajectory_data.csv'}")


if __name__ == "__main__":
    main()

