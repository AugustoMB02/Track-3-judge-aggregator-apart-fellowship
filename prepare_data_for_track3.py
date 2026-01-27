#!/usr/bin/env python3
"""
Data Preparation Script for Track 3 Pruning Experiments

Converts the recomputed datasets into the format required by the iterative selection pipeline.
Creates flattened judge scores suitable for GAM training.

Usage:
    python3 prepare_data_for_track3.py
"""

import pickle
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

def _mean_target_from_dict(value):
    if not isinstance(value, dict):
        return np.nan
    values = [v for v in value.values() if v is not None]
    if not values:
        return np.nan
    return float(np.mean(values))


def _extract_feedback_score(value):
    if isinstance(value, dict):
        return float(value.get("score", value.get("average_score", 0.0)))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _load_parent_judge_ids() -> set:
    parent_path = Path("judges/helpsteer2/depth_0_parents.yaml")
    if not parent_path.exists():
        return set()
    with parent_path.open() as f:
        data = yaml.safe_load(f)
    return {judge["id"] for judge in data.get("judges", [])}


def prepare_helpsteer2_data():
    """Prepare HelpSteer2 full dataset for track 3 experiments."""
    print("Loading HelpSteer2 full dataset...")
    
    input_file = Path("datasets/helpsteer2_full_30_judges_recomputed.pkl")
    output_file = Path("datasets/helpsteer2_track3_full_dataset.pkl")
    
    with open(input_file, "rb") as f:
        df = pickle.load(f)
    
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Ensure judge_scores and judge_ids are properly formatted
    if 'judge_scores' in df.columns and 'judge_ids' in df.columns:
        n_judges = len(df['judge_ids'].iloc[0])
        judge_names = df['judge_ids'].iloc[0]

        print(f"Processing {n_judges} judges: {judge_names[:5]}...")

        targets = df["target_human_aggregated"].apply(_mean_target_from_dict).tolist()
        processed_df = pd.DataFrame(
            {
                "judge_scores": df["judge_scores"].tolist(),
                "judge_ids": df["judge_ids"].tolist(),
                "target": targets,
            }
        )

        # Store judge metadata
        processed_df.attrs["judge_ids"] = judge_names
        processed_df.attrs["n_judges"] = n_judges

        print(f"Processed dataset shape: {processed_df.shape}")
        print(f"Missing target values: {processed_df['target'].isna().sum()}")

        # Save processed dataset (parents + children)
        with open(output_file, "wb") as f:
            pickle.dump(processed_df, f)

        print(f"✓ Saved to {output_file}")

        # Save children-only dataset (exclude parent judges)
        parent_ids = _load_parent_judge_ids()
        child_indices = [i for i, jid in enumerate(judge_names) if jid not in parent_ids]
        child_judges = [judge_names[i] for i in child_indices]
        child_scores = [
            [scores[i] for i in child_indices]
            for scores in df["judge_scores"].tolist()
        ]
        children_df = pd.DataFrame(
            {
                "judge_scores": child_scores,
                "judge_ids": [list(child_judges) for _ in range(len(df))],
                "target": targets,
            }
        )
        children_df.attrs["judge_ids"] = child_judges
        children_df.attrs["n_judges"] = len(child_judges)
        children_file = Path("datasets/helpsteer2_track3_children_dataset.pkl")
        with children_file.open("wb") as f:
            pickle.dump(children_df, f)
        print(f"✓ Saved children-only dataset to {children_file}")
        return processed_df
    else:
        print("ERROR: Expected columns 'judge_scores' and 'judge_ids' not found")
        return None


def prepare_ultrafeedback_data():
    """Prepare UltraFeedback workshop dataset for track 3 experiments."""
    print("\nLoading UltraFeedback workshop dataset...")
    
    input_file = Path("datasets/ultrafeedback_workshop_55_judges_repaired_full.pkl")
    output_file = Path("datasets/ultrafeedback_track3_full_dataset.pkl")
    
    with open(input_file, "rb") as f:
        df = pickle.load(f)
    
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle UltraFeedback specific column names
    score_col = 'judge_scores_55'
    id_col = 'judge_ids_55'
    
    if score_col in df.columns and id_col in df.columns:
        n_judges = len(df[id_col].iloc[0])
        judge_names = df[id_col].iloc[0]

        print(f"Processing {n_judges} judges: {judge_names[:5]}...")

        targets = df["human_feedback"].apply(_extract_feedback_score).tolist()
        processed_df = pd.DataFrame(
            {
                "judge_scores": df[score_col].tolist(),
                "judge_ids": df[id_col].tolist(),
                "target": targets,
            }
        )

        # Store judge metadata
        processed_df.attrs["judge_ids"] = judge_names
        processed_df.attrs["n_judges"] = n_judges

        print(f"Processed dataset shape: {processed_df.shape}")
        print(f"Missing target values: {processed_df['target'].isna().sum()}")

        # Save processed dataset
        with open(output_file, "wb") as f:
            pickle.dump(processed_df, f)

        print(f"✓ Saved to {output_file}")
        return processed_df
    else:
        print(f"ERROR: Expected columns '{score_col}' and '{id_col}' not found")
        return None


def main():
    """Prepare all datasets for track 3 experiments."""
    print("="*70)
    print("Track 3: Data Preparation for Pruning Experiments")
    print("="*70)
    
    # Check if input files exist
    helpsteer2_file = Path("datasets/helpsteer2_full_30_judges_recomputed.pkl")
    ultrafeedback_file = Path("datasets/ultrafeedback_workshop_55_judges_repaired_full.pkl")
    
    if not helpsteer2_file.exists():
        print(f"ERROR: {helpsteer2_file} not found")
        return False
    
    if not ultrafeedback_file.exists():
        print(f"ERROR: {ultrafeedback_file} not found")
        return False
    
    # Prepare both datasets
    hs2_df = prepare_helpsteer2_data()
    uf_df = prepare_ultrafeedback_data()
    
    if hs2_df is not None and uf_df is not None:
        print("\n" + "="*70)
        print("✓ Data preparation complete!")
        print("="*70)
        hs2_judges = hs2_df.attrs.get("n_judges", len(hs2_df.iloc[0]["judge_scores"]))
        uf_judges = uf_df.attrs.get("n_judges", len(uf_df.iloc[0]["judge_scores"]))
        print(f"\nHelpSteer2 dataset: {len(hs2_df)} samples, {hs2_judges} judges")
        print(f"UltraFeedback dataset: {len(uf_df)} samples, {uf_judges} judges")
        print("\nReady for track 3 experiments. Run:")
        print("  python experiments/track3_automated_selection/iterative_selection/iterative_selection.py ...")
        return True
    else:
        print("\n" + "="*70)
        print("✗ Data preparation failed")
        print("="*70)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
