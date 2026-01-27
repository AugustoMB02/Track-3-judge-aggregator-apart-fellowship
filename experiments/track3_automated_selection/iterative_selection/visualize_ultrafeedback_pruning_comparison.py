#!/usr/bin/env python3
"""
Visualize UltraFeedback pruning strategy comparison.

Reads comparison_summary.csv and writes:
- bar chart of final R²
- bar chart of final MAE
- bar chart of ΔR²
- markdown table

Usage:
  PYTHONPATH=. .venv/bin/python \
    experiments/track3_automated_selection/iterative_selection/visualize_ultrafeedback_pruning_comparison.py \
    --comparison-csv results/ultrafeedback_pruning_metric_compare_20260127_093649/comparison_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _bar_plot(df: pd.DataFrame, x: str, y: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df[x], df[y], color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("Strategy")
    ax.set_ylabel(y)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize UltraFeedback pruning comparison"
    )
    parser.add_argument(
        "--comparison-csv",
        type=str,
        required=True,
        help="Path to comparison_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write plots/tables (default: alongside CSV)",
    )
    args = parser.parse_args()

    csv_path = Path(args.comparison_csv)
    df = pd.read_csv(csv_path)

    # Keep a consistent order (by final_r2 descending)
    df = df.sort_values(["final_r2", "delta_r2"], ascending=[False, False])

    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    _bar_plot(
        df,
        x="label",
        y="final_r2",
        title="UltraFeedback: Final R² by Strategy",
        out_path=output_dir / "final_r2_by_strategy.png",
    )
    _bar_plot(
        df,
        x="label",
        y="final_mae",
        title="UltraFeedback: Final MAE by Strategy",
        out_path=output_dir / "final_mae_by_strategy.png",
    )
    _bar_plot(
        df,
        x="label",
        y="delta_r2",
        title="UltraFeedback: ΔR² (Final - Iter0) by Strategy",
        out_path=output_dir / "delta_r2_by_strategy.png",
    )

    # Markdown table
    table_cols = [
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
    ]
    # Write a simple markdown table without requiring tabulate.
    table_df = df[table_cols].copy()
    headers = list(table_df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in table_df.iterrows():
        values = []
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                values.append(f"{val:.6f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    (output_dir / "comparison_summary.md").write_text("\n".join(lines))

    print("Wrote:")
    print(f"- {output_dir / 'final_r2_by_strategy.png'}")
    print(f"- {output_dir / 'final_mae_by_strategy.png'}")
    print(f"- {output_dir / 'delta_r2_by_strategy.png'}")
    print(f"- {output_dir / 'comparison_summary.md'}")


if __name__ == "__main__":
    main()
