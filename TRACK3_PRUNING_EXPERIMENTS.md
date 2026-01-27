# Track 3: Pruning Experiments with Full Datasets

This document provides instructions for running the pruning experiments on the GAN branch using the newly recomputed full datasets.

## Available Datasets

Two newly recomputed datasets have been added to `datasets/`:

1. **HelpSteer2 Full Dataset** (20,324 samples, 30 judges)
   - File: `datasets/helpsteer2_full_30_judges_recomputed.pkl`
   - Size: ~50MB
   - Contains: 5 parent judges + 25 child judges
   - Columns: `judge_scores`, `judge_ids`, `target`

2. **Workshop UltraFeedback Dataset** (2,000 samples, 55 judges)
   - File: `datasets/ultrafeedback_workshop_55_judges_repaired_full.pkl`
   - Size: ~10MB
   - Contains: 55 judges (decomposed from workshop dataset)
   - Columns: `judge_scores_55`, `judge_ids_55`, `target`

## Quick Start

### Step 1: Activate Virtual Environment

```bash
source .venv/bin/activate
```

### Step 2: Prepare Data for Track 3

The iterative selection pipeline requires data in a specific format. Use the provided data preparation script:

```bash
python3 prepare_data_for_track3.py
```

This will create:
- `datasets/helpsteer2_track3_full_dataset.pkl` - Reformatted for iterative selection
- `datasets/ultrafeedback_track3_full_dataset.pkl` - Reformatted for iterative selection

### Step 3: Run Pruning Experiments

#### HelpSteer2 Full Dataset (Backward Selection - Prune to 15 judges)

```bash
python experiments/track3_automated_selection/iterative_selection/iterative_selection.py \
    --config config/selection_experiment_prune_half.yaml \
    --data-file datasets/helpsteer2_track3_full_dataset.pkl \
    --output results/track3_full_dataset/helpsteer2_prune_half \
    --max-iterations 20 \
    --initial-judges 30
```

#### UltraFeedback Workshop Dataset (Backward Selection - Prune to 27 judges)

```bash
python experiments/track3_automated_selection/iterative_selection/iterative_selection.py \
    --config config/selection_experiment_prune_half_ultrafeedback.yaml \
    --data-file datasets/ultrafeedback_track3_full_dataset.pkl \
    --output results/track3_full_dataset/ultrafeedback_prune_half \
    --max-iterations 20 \
    --initial-judges 55
```

### Step 4: Generate Visualizations

After experiments complete, generate comprehensive visualizations:

```bash
python experiments/track3_automated_selection/iterative_selection/visualize_selection_results.py \
    --run-dir results/track3_full_dataset/helpsteer2_prune_half \
    --output-dir results/track3_full_dataset/helpsteer2_prune_half/visualizations

python experiments/track3_automated_selection/iterative_selection/visualize_selection_results.py \
    --run-dir results/track3_full_dataset/ultrafeedback_prune_half \
    --output-dir results/track3_full_dataset/ultrafeedback_prune_half/visualizations
```

### Step 5: View Results

Results will be saved in `Results track 3 full dataset/` with the following structure:

```
Results track 3 full dataset/
├── helpsteer2_prune_half/
│   ├── iteration_0/
│   ├── iteration_1/
│   ├── ...
│   ├── visualizations/
│   │   ├── metrics_over_iterations.png
│   │   ├── judge_count_over_time.png
│   │   ├── selected_judges_timeline.png
│   │   └── composite_score_evolution.png
│   └── config.yaml
├── ultrafeedback_prune_half/
│   ├── iteration_0/
│   ├── iteration_1/
│   ├── ...
│   ├── visualizations/
│   │   ├── metrics_over_iterations.png
│   │   ├── judge_count_over_time.png
│   │   ├── selected_judges_timeline.png
│   │   └── composite_score_evolution.png
│   └── config.yaml
└── SUMMARY.md
```

## Full Automated Pipeline

Run everything in one go:

```bash
#!/bin/bash
source .venv/bin/activate

# Prepare data
python3 prepare_data_for_track3.py

# Run both pruning experiments
python experiments/track3_automated_selection/iterative_selection/iterative_selection.py \
    --config config/selection_experiment_prune_half.yaml \
    --data-file datasets/helpsteer2_track3_full_dataset.pkl \
    --output results/track3_full_dataset/helpsteer2_prune_half \
    --max-iterations 20 --initial-judges 30

python experiments/track3_automated_selection/iterative_selection/iterative_selection.py \
    --config config/selection_experiment_prune_half_ultrafeedback.yaml \
    --data-file datasets/ultrafeedback_track3_full_dataset.pkl \
    --output results/track3_full_dataset/ultrafeedback_prune_half \
    --max-iterations 20 --initial-judges 55

# Generate visualizations
python experiments/track3_automated_selection/iterative_selection/visualize_selection_results.py \
    --run-dir results/track3_full_dataset/helpsteer2_prune_half \
    --output-dir results/track3_full_dataset/helpsteer2_prune_half/visualizations

python experiments/track3_automated_selection/iterative_selection/visualize_selection_results.py \
    --run-dir results/track3_full_dataset/ultrafeedback_prune_half \
    --output-dir results/track3_full_dataset/ultrafeedback_prune_half/visualizations

# Copy results to final location
mkdir -p "Results track 3 full dataset"
cp -r results/track3_full_dataset/* "Results track 3 full dataset/"

echo "✓ All experiments complete. Results in 'Results track 3 full dataset/'"
```

## Configuration Files

Three configuration files are available for different pruning strategies:

- `config/selection_experiment_prune_half.yaml` - HelpSteer2 (30→15 judges)
- `config/selection_experiment_prune_half_ultrafeedback.yaml` - UltraFeedback (55→27 judges)
- `config/selection_experiment.yaml` - Full iterative selection (no target size)

## Key Metrics in Results

Each iteration generates:

1. **Predictive Performance**
   - R² Score (coefficient of determination)
   - MAE (mean absolute error)
   - MSE (mean squared error)
   - Correlation metrics (Spearman, Kendall, Pearson)

2. **Judge Set Metrics**
   - Composite score (weighted combination of metrics)
   - Redundancy (pairwise judge correlations)
   - Diversity (coverage of different evaluation dimensions)

3. **Judge Selection Info**
   - Added/removed judge per iteration
   - Importance scores
   - Gap analysis results

## Troubleshooting

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Out of Memory
Reduce batch size in config files or process datasets in chunks.

### Data Format Issues
Run the data preparation script to ensure correct format:
```bash
python3 prepare_data_for_track3.py
```

## Expected Runtime

- **Data Preparation**: ~2-5 minutes
- **HelpSteer2 Pruning (20 iterations)**: ~30-60 minutes
- **UltraFeedback Pruning (20 iterations)**: ~15-30 minutes
- **Visualization Generation**: ~5-10 minutes
- **Total**: ~1-2 hours

## Next Steps

1. Review the generated visualizations in `Results track 3 full dataset/`
2. Analyze the selected judge sets and their importance scores
3. Compare results between HelpSteer2 and UltraFeedback datasets
4. Consider different pruning strategies (see Configuration Files section)

## References

- [Track 3 README](experiments/track3_automated_selection/README.md)
- [Iterative Selection Implementation](experiments/track3_automated_selection/iterative_selection/iterative_selection.py)
- [Visualization Script](experiments/track3_automated_selection/iterative_selection/visualize_selection_results.py)
