#!/usr/bin/env python3
"""
Prepare validation dataset for judge selection experiment.

Takes existing UltraFeedback data and creates a version suitable for
backward elimination with parent judges only.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("=" * 80)
    print("PREPARING VALIDATION DATASET")
    print("=" * 80)
    
    # Load existing data
    print("\n1. Loading existing data...")
    with open('datasets/data_with_judge_scores.pkl', 'rb') as f:
        df = pickle.load(f)
    print(f"   Loaded {len(df)} samples with {len(df['judge_scores'].iloc[0])} judge scores")
    
    # Extract target from human_feedback
    print("\n2. Extracting target scores...")
    targets = []
    for _, row in df.iterrows():
        if isinstance(row['human_feedback'], dict) and 'score' in row['human_feedback']:
            targets.append(row['human_feedback']['score'])
        elif isinstance(row['human_feedback'], dict) and 'average_score' in row['human_feedback']:
            targets.append(row['human_feedback']['average_score'])
        else:
            targets.append(np.nan)
    
    df['target'] = targets
    
    # Remove samples with missing targets
    original_len = len(df)
    df = df[df['target'].notna()].reset_index(drop=True)
    print(f"   Extracted {len(df)} samples with valid targets (removed {original_len - len(df)})")
    print(f"   Target range: [{df['target'].min():.2f}, {df['target'].max():.2f}]")
    print(f"   Target mean: {df['target'].mean():.2f}, std: {df['target'].std():.2f}")
    
    # Validate judge_scores match expected count
    print("\n3. Validating judge scores...")
    expected_judges = 10  # UltraFeedback has 10 parent judges
    judge_counts = df['judge_scores'].apply(len)
    if not all(judge_counts == expected_judges):
        print(f"   WARNING: Some samples have {judge_counts.value_counts().to_dict()} judges")
        df = df[judge_counts == expected_judges].reset_index(drop=True)
        print(f"   Filtered to {len(df)} samples with exactly {expected_judges} judges")
    else:
        print(f"   ✓ All samples have exactly {expected_judges} judge scores")
    
    # Save prepared dataset
    output_path = Path('datasets/processed/ultrafeedback_validation.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n4. Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(df, f)
    
    print(f"   ✓ Saved {len(df)} samples")
    print(f"\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\nDataset ready for backward elimination:")
    print(f"  - Path: {output_path}")
    print(f"  - Samples: {len(df)}")
    print(f"  - Judges: {expected_judges}")
    print(f"  - Target column: 'target'")
    print(f"  - Judge scores column: 'judge_scores'")
    
    return df

if __name__ == "__main__":
    main()
