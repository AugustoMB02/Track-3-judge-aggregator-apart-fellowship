#!/usr/bin/env python3
"""
Summarize best iteration per strategy for UltraFeedback pruning runs.

Selects the best iteration by highest test R² (tie-breaker: lower test MAE),
and outputs a table including metrics, judge count, and judge list.

Usage:
  PYTHONPATH=. .venv/bin/python \
    experiments/track3_automated_selection/iterative_selection/summarize_ultrafeedback_best_per_strategy.py \
    --root results/ultrafeedback_pruning_metric_compare_to1_20260127_100200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _safe_float(value: Optional[float]) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _load_best_iteration(selection_dir: Path) -> Dict:
    best_row: Optional[Dict] = None

    for iteration_dir in sorted(selection_dir.glob("iteration_*")):
        result_path = iteration_dir / "result.json"
        if not result_path.exists():
            continue
        data = json.loads(result_path.read_text())
        test = data.get("test_metrics") or {}
        val = data.get("val_metrics") or {}

        row = {
            "iteration": int(data.get("iteration", 0)),
            "test_r2": _safe_float(test.get("r2")),
            "test_mae": _safe_float(test.get("mae")),
            "val_r2": _safe_float(val.get("r2")),
            "val_mae": _safe_float(val.get("mae")),
            "n_judges": data.get("n_judges"),
            "judge_names": ", ".join(data.get("judge_names") or []),
        }

        if best_row is None:
            best_row = row
            continue

        # Select by highest test_r2; tie-breaker is lowest test_mae.
        if row["test_r2"] > best_row["test_r2"]:
            best_row = row
        elif row["test_r2"] == best_row["test_r2"]:
            if row["test_mae"] < best_row["test_mae"]:
                best_row = row

    return best_row or {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize best per-strategy iteration for UF pruning"
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
        help="Directory to write summary (default: root)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir) if args.output_dir else root
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []

    for strategy_dir in sorted(root.iterdir()):
        selection_dir = strategy_dir / "selection"
        if not selection_dir.is_dir():
            continue
        best = _load_best_iteration(selection_dir)
        if not best:
            continue
        rows.append(
            {
                "strategy": strategy_dir.name,
                **best,
            }
        )

    if not rows:
        raise SystemExit(f"No selection runs found under: {root}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["test_r2", "test_mae"], ascending=[False, True])

    csv_path = output_dir / "best_per_strategy.csv"
    md_path = output_dir / "best_per_strategy.md"
    df.to_csv(csv_path, index=False)

    # Simple markdown table without tabulate
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        values = []
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                values.append(f"{val:.6f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines))

    print("Wrote:")
    print(f"- {csv_path}")
    print(f"- {md_path}")


if __name__ == "__main__":
    main()

