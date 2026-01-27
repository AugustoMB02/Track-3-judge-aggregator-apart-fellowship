#!/usr/bin/env python3
"""
Track 1: HelpSteer2 full dataset, all judges, GAM only, plus baselines.

Uses precomputed judge scores from:
  datasets/helpsteer2_full_30_judges_recomputed.pkl
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from pipeline.core.aggregator_training import GAMAggregator, compute_metrics


def _load_dataframe(path: Path) -> pd.DataFrame:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    required = ["judge_scores", "judge_ids", "target_human_aggregated"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    return data


def _infer_dimensions(df: pd.DataFrame) -> List[str]:
    if "dimensions" in df.columns:
        for val in df["dimensions"]:
            if isinstance(val, list) and val:
                return list(val)
    for val in df["target_human_aggregated"]:
        if isinstance(val, dict) and val:
            return list(val.keys())
    raise ValueError("Could not infer dimensions from dataset")


def _extract_targets(df: pd.DataFrame, dimension: str) -> np.ndarray:
    return np.array(
        [
            row["target_human_aggregated"].get(dimension, np.nan)
            if row["target_human_aggregated"] is not None
            else np.nan
            for _, row in df.iterrows()
        ],
        dtype=float,
    )


def _compute_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    judge_ids: List[str],
    judge_indices: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    baselines: Dict[str, Dict[str, Any]] = {}

    X_train_sub = X_train[:, judge_indices]
    X_test_sub = X_test[:, judge_indices]
    judge_ids_sub = [judge_ids[i] for i in judge_indices]

    # Mean / median / max across all judges
    baselines["mean"] = {
        "metrics": compute_metrics(y_test, X_test_sub.mean(axis=1))
    }
    baselines["median"] = {
        "metrics": compute_metrics(y_test, np.median(X_test_sub, axis=1))
    }
    baselines["max"] = {
        "metrics": compute_metrics(y_test, X_test_sub.max(axis=1))
    }

    # Best single judge picked by train R²
    train_r2 = [
        r2_score(y_train, X_train_sub[:, j]) for j in range(X_train_sub.shape[1])
    ]
    best_idx = int(np.argmax(train_r2))
    baselines["best_single_judge"] = {
        "best_judge_id": judge_ids_sub[best_idx],
        "train_r2": float(train_r2[best_idx]),
        "metrics": compute_metrics(y_test, X_test_sub[:, best_idx]),
    }

    return baselines


def main() -> None:
    parser = argparse.ArgumentParser(description="Track 1 HelpSteer2 full run (GAM + baselines)")
    parser.add_argument(
        "--input",
        default="datasets/helpsteer2_full_30_judges_recomputed.pkl",
        help="Path to precomputed HelpSteer2 data with judge scores",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        default="results/track1_helpsteer2_full",
        help="Output directory prefix",
    )
    parser.add_argument(
        "--baseline-scope",
        choices=["all", "dimension"],
        default="dimension",
        help="Which judges baselines should use (all or only dimension-matching)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    df = _load_dataframe(input_path)
    dimensions = _infer_dimensions(df)

    # Prepare features
    judge_ids = df.iloc[0]["judge_ids"]
    X_all = np.array(df["judge_scores"].tolist())

    # Shared split across all dimensions
    all_indices = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        all_indices, test_size=args.test_size, random_state=args.seed
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"{args.output}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    per_dim_results: Dict[str, Any] = {}

    for dim in dimensions:
        y_all = _extract_targets(df, dim)

        X_train_all = X_all[idx_train]
        y_train_all = y_all[idx_train]
        train_mask = ~np.isnan(y_train_all)
        X_train = X_train_all[train_mask]
        y_train = y_train_all[train_mask]

        X_test_all = X_all[idx_test]
        y_test_all = y_all[idx_test]
        test_mask = ~np.isnan(y_test_all)
        X_test = X_test_all[test_mask]
        y_test = y_test_all[test_mask]

        # Train GAM (all judges)
        gam = GAMAggregator(feature_names=judge_ids, n_splines=10, lam=0.6)
        gam.fit(X_train, y_train)
        gam_pred = gam.predict(X_test)
        gam_metrics = compute_metrics(y_test, gam_pred)

        # Baselines
        if args.baseline_scope == "dimension":
            prefix = f"helpsteer2-{dim}-judge"
            baseline_indices = np.array(
                [i for i, jid in enumerate(judge_ids) if jid.startswith(prefix)],
                dtype=int,
            )
        else:
            baseline_indices = np.arange(len(judge_ids), dtype=int)

        baselines = _compute_baselines(
            X_train, y_train, X_test, y_test, judge_ids, baseline_indices
        )

        per_dim_results[dim] = {
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "gam_metrics": gam_metrics,
            "baselines": baselines,
            "baseline_scope": args.baseline_scope,
            "baseline_n_judges": int(len(baseline_indices)),
        }

        # Flatten for CSV
        summary_rows.append({
            "dimension": dim,
            "method": "gam",
            **gam_metrics,
        })
        for name, info in baselines.items():
            row = {"dimension": dim, "method": name, **info["metrics"]}
            if name == "best_single_judge":
                row["best_judge_id"] = info["best_judge_id"]
                row["best_judge_train_r2"] = info["train_r2"]
            row["baseline_scope"] = args.baseline_scope
            row["baseline_n_judges"] = int(len(baseline_indices))
            summary_rows.append(row)

    # Save outputs
    (out_dir / "summary.json").write_text(json.dumps(per_dim_results, indent=2))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)

    config = {
        "input": str(input_path),
        "n_samples": int(len(df)),
        "n_judges": int(len(judge_ids)),
        "dimensions": dimensions,
        "test_size": args.test_size,
        "seed": args.seed,
        "model": "GAM",
        "baselines": ["mean", "median", "max", "best_single_judge"],
        "baseline_scope": args.baseline_scope,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"✅ Track 1 HelpSteer2 full run complete: {out_dir}")
    print(f"   Metrics: {out_dir / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()
