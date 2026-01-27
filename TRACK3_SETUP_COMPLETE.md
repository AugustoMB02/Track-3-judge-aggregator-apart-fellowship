# Track 3 Full Dataset Pruning Experiments - Setup Complete

**Date**: January 26, 2026  
**Branch**: `Gan` (has .venv with full environment)  
**Status**: ✓ Ready to Run

## What Was Done

1. **Moved Full Datasets to Gan Branch**
   - `datasets/helpsteer2_full_30_judges_recomputed.pkl` (53 MB)
     - 20,324 samples with 30 judges (5 parents + 25 children)
   - `datasets/ultrafeedback_workshop_55_judges_repaired_full.pkl` (12 MB)
     - 2,000 samples with 55 judges

2. **Created Data Preparation Script**
   - `prepare_data_for_track3.py`
   - Converts datasets into format required by iterative selection pipeline
   - Expands judge scores into individual columns for GAM training

3. **Created Comprehensive Instructions**
   - `TRACK3_PRUNING_EXPERIMENTS.md`
   - Step-by-step guide for running experiments
   - Configuration options and expected runtimes
   - Troubleshooting guide

## Quick Start

```bash
# 1. Switch to Gan branch (if not already there)
git checkout Gan

# 2. Activate environment
source .venv/bin/activate

# 3. Prepare data
python3 prepare_data_for_track3.py

# 4. Run pruning experiments
python experiments/track3_automated_selection/iterative_selection/iterative_selection.py \
    --config config/selection_experiment_prune_half.yaml \
    --data-file datasets/helpsteer2_track3_full_dataset.pkl \
    --output results/track3_full_dataset/helpsteer2_prune_half \
    --max-iterations 20

# 5. Generate visualizations
python experiments/track3_automated_selection/iterative_selection/visualize_selection_results.py \
    --run-dir results/track3_full_dataset/helpsteer2_prune_half \
    --output-dir results/track3_full_dataset/helpsteer2_prune_half/visualizations
```

## Files on Gan Branch

### New Files Added
- `datasets/helpsteer2_full_30_judges_recomputed.pkl` - Full HelpSteer2 dataset
- `datasets/ultrafeedback_workshop_55_judges_repaired_full.pkl` - Full UltraFeedback dataset
- `prepare_data_for_track3.py` - Data preparation script
- `TRACK3_PRUNING_EXPERIMENTS.md` - Detailed instructions

### Existing Infrastructure
- `.venv/` - Complete Python environment
- `experiments/track3_automated_selection/` - Track 3 pipeline code
- `config/` - Configuration files for pruning strategies
- `requirements.txt` - All dependencies

## Expected Outputs

Results will be generated in:
```
results/track3_full_dataset/
├── helpsteer2_prune_half/
│   ├── iteration_0/
│   ├── iteration_1/
│   ├── ...
│   ├── visualizations/
│   │   ├── metrics_over_iterations.png
│   │   ├── judge_count_over_time.png
│   │   └── ...
│   └── config.yaml
└── ultrafeedback_prune_half/
    ├── iteration_0/
    ├── ...
    └── visualizations/
```

Can then be copied to "Results track 3 full dataset/" folder as needed.

## System Requirements

- Python 3.8+
- ~100 GB disk space (for datasets + results)
- ~16 GB RAM recommended for GAM training

## Environment Status

The Gan branch already has:
- ✓ `.venv` virtual environment configured
- ✓ All dependencies installed
- ✓ Test infrastructure in place
- ✓ GAM and MLP analysis tools ready

No additional setup needed - ready to run experiments immediately!

## Next Steps

1. Review `TRACK3_PRUNING_EXPERIMENTS.md` for full documentation
2. Run `prepare_data_for_track3.py` to format datasets
3. Execute pruning experiments
4. Generate and review visualizations
5. Copy results to "Results track 3 full dataset/" folder

## Notes

- Datasets are fresh recomputed versions with full data (not just partial cache)
- UltraFeedback uses guided JSON output for reliable score extraction
- HelpSteer2 includes both parent and child judges for comprehensive evaluation
- All configurations preserve original judge ordering in `judge_ids` columns

For detailed instructions, see `TRACK3_PRUNING_EXPERIMENTS.md`
