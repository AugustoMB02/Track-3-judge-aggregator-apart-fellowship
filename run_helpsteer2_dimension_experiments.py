#!/usr/bin/env python3
"""
Run per-dimension HelpSteer2 experiments:
- Parent-only baseline (single judge)
- Children-only (all)
- Children-only (pruned to 3)

Generates baseline-overlaid plots for children runs.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

from experiments.track3_automated_selection.iterative_selection.iterative_selection import (
    IterativeJudgeSelector,
    SelectionConfig,
)
from experiments.track3_automated_selection.iterative_selection import visualize_selection_results as viz


BASE_DATASET = Path("datasets/helpsteer2_full_30_judges_recomputed.pkl")
PARENT_YAML = Path("judges/helpsteer2/depth_0_parents.yaml")
CHILD_YAML = Path("judges/helpsteer2/depth_1_children.yaml")
OUTPUT_ROOT = Path("results/track3_full_dataset")
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1


def _load_judges(path: Path) -> List[Dict]:
    with path.open() as f:
        data = yaml.safe_load(f)
    return data.get("judges", [])


def _write_judges(judges: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"judges": judges}
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))


def _build_dataset(
    df: pd.DataFrame,
    judge_ids: List[str],
    subset_ids: List[str],
    dimension: str,
    out_path: Path,
) -> None:
    idxs = [judge_ids.index(jid) for jid in subset_ids]
    scores = [[row["judge_scores"][i] for i in idxs] for _, row in df.iterrows()]
    targets = [row["target_human_aggregated"].get(dimension) for _, row in df.iterrows()]
    out_df = pd.DataFrame(
        {
            "judge_scores": scores,
            "judge_ids": [list(subset_ids) for _ in range(len(df))],
            "target": targets,
        }
    )
    out_df.attrs["judge_ids"] = list(subset_ids)
    out_df.attrs["n_judges"] = len(subset_ids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(out_df, f)


def _run_selection(config: SelectionConfig) -> Path:
    selector = IterativeJudgeSelector(config)
    selector.run()
    return Path(config.output_dir)


def _plot_with_baseline(run_dir: Path, baseline_dir: Path) -> None:
    output_dir = run_dir / "visualizations"
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return
    series = viz._extract_series(viz._load_iteration_results(run_dir))
    baseline = viz._extract_baseline_metrics(baseline_dir)
    summary = viz._load_json(summary_path)
    viz._plot_series(run_dir, series, output_dir, baseline=baseline)
    viz._write_removals(run_dir, summary, output_dir)


def main() -> None:
    if not BASE_DATASET.exists():
        raise FileNotFoundError(f"Missing dataset: {BASE_DATASET}")

    df = pickle.load(BASE_DATASET.open("rb"))
    judge_ids = list(df.iloc[0]["judge_ids"])

    parent_judges = _load_judges(PARENT_YAML)
    child_judges = _load_judges(CHILD_YAML)

    dimensions = sorted(
        {dim for row in df["target_human_aggregated"] for dim in row.keys()}
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for dim in dimensions:
        parent_id = f"helpsteer2-{dim}-judge"
        parent = [j for j in parent_judges if j.get("id") == parent_id]
        if not parent:
            print(f"Skipping {dim}: missing parent judge {parent_id}")
            continue

        children = [j for j in child_judges if j.get("dimension") == dim]
        if not children:
            print(f"Skipping {dim}: no child judges found")
            continue

        child_ids = {j["id"] for j in children}
        parent_ids_ordered = [jid for jid in judge_ids if jid == parent_id]
        child_ids_ordered = [jid for jid in judge_ids if jid in child_ids]

        if not parent_ids_ordered or not child_ids_ordered:
            print(f"Skipping {dim}: missing judge IDs in dataset")
            continue

        dataset_root = Path("datasets")
        judges_root = Path("judges/helpsteer2")
        parent_dataset = dataset_root / f"helpsteer2_{dim}_parent_dataset.pkl"
        children_dataset = dataset_root / f"helpsteer2_{dim}_children_dataset.pkl"
        parent_yaml = judges_root / f"{dim}_parent.yaml"
        children_yaml = judges_root / f"{dim}_children.yaml"

        _build_dataset(df, judge_ids, parent_ids_ordered, dim, parent_dataset)
        _build_dataset(df, judge_ids, child_ids_ordered, dim, children_dataset)
        _write_judges(parent, parent_yaml)
        _write_judges(children, children_yaml)

        parent_run = OUTPUT_ROOT / f"helpsteer2_{dim}_parent_{timestamp}"
        children_all_run = OUTPUT_ROOT / f"helpsteer2_{dim}_children_all_{timestamp}"
        children_prune_run = OUTPUT_ROOT / f"helpsteer2_{dim}_children_prune_{timestamp}"

        parent_config = SelectionConfig(
            name=f"helpsteer2-{dim}-parent-only",
            description=f"{dim} parent-only baseline",
            initial_judge_file=str(parent_yaml),
            protected_judges=[],
            data_file=str(parent_dataset),
            target_column="target",
            train_test_split=TEST_SPLIT,
            validation_split=VAL_SPLIT,
            max_iterations=1,
            min_judges=1,
            target_judges=1,
            max_judges=1,
            r2_improvement_threshold=0.0,
            plateau_patience=1,
            use_llm_suggestions=False,
            output_dir=str(parent_run),
            save_intermediate=True,
        )
        parent_dir = _run_selection(parent_config)

        children_all_config = SelectionConfig(
            name=f"helpsteer2-{dim}-children-all",
            description=f"{dim} children-only baseline",
            initial_judge_file=str(children_yaml),
            protected_judges=[],
            data_file=str(children_dataset),
            target_column="target",
            train_test_split=TEST_SPLIT,
            validation_split=VAL_SPLIT,
            max_iterations=1,
            min_judges=len(child_ids_ordered),
            target_judges=len(child_ids_ordered),
            max_judges=len(child_ids_ordered),
            r2_improvement_threshold=0.0,
            plateau_patience=1,
            use_llm_suggestions=False,
            output_dir=str(children_all_run),
            save_intermediate=True,
        )
        children_all_dir = _run_selection(children_all_config)

        children_prune_config = SelectionConfig(
            name=f"helpsteer2-{dim}-children-prune",
            description=f"{dim} children-only prune to 3",
            initial_judge_file=str(children_yaml),
            protected_judges=[],
            data_file=str(children_dataset),
            target_column="target",
            train_test_split=TEST_SPLIT,
            validation_split=VAL_SPLIT,
            max_iterations=10,
            min_judges=3,
            target_judges=3,
            max_judges=len(child_ids_ordered),
            r2_improvement_threshold=0.0,
            plateau_patience=50,
            use_llm_suggestions=False,
            output_dir=str(children_prune_run),
            save_intermediate=True,
        )
        children_prune_dir = _run_selection(children_prune_config)

        _plot_with_baseline(children_all_dir, parent_dir)
        _plot_with_baseline(children_prune_dir, parent_dir)

        print(f"Completed {dim}: parent={parent_dir.name}, children_all={children_all_dir.name}, children_prune={children_prune_dir.name}")


if __name__ == "__main__":
    main()
