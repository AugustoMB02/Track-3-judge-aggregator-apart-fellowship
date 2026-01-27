#!/usr/bin/env python3
import pickle
import pandas as pd

# Check HelpSteer2 full dataset
print("=" * 60)
print("HelpSteer2 Full 30 Judges Dataset")
print("=" * 60)
with open("datasets/helpsteer2_full_30_judges_recomputed.pkl", "rb") as f:
    data = pickle.load(f)
    if isinstance(data, pd.DataFrame):
        print(f"Shape: {data.shape}")
        print(f"\nColumns:\n{data.columns.tolist()}")
        print(f"\nFirst row index and keys: {data.index[0] if len(data) > 0 else 'N/A'}")
        if 'judge_scores' in data.columns:
            print(f"\nJudge scores shape: {data['judge_scores'].iloc[0].shape if len(data) > 0 else 'N/A'}")
            print(f"Judge IDs: {data['judge_ids'].iloc[0][:5] if len(data) > 0 and 'judge_ids' in data.columns else 'N/A'}")

print("\n" + "=" * 60)
print("Workshop UltraFeedback Dataset")
print("=" * 60)
with open("datasets/ultrafeedback_workshop_55_judges_repaired_full.pkl", "rb") as f:
    data = pickle.load(f)
    if isinstance(data, pd.DataFrame):
        print(f"Shape: {data.shape}")
        print(f"\nColumns:\n{data.columns.tolist()}")
        if 'judge_scores_55' in data.columns:
            print(f"\nJudge scores shape: {data['judge_scores_55'].iloc[0].shape if len(data) > 0 else 'N/A'}")
            print(f"Judge IDs: {data['judge_ids_55'].iloc[0][:5] if len(data) > 0 and 'judge_ids_55' in data.columns else 'N/A'}")
