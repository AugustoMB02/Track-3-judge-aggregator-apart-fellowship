# ✓ Track 3 Pruning Experiments - Ready to Run

**Status**: Complete Setup  
**Branch**: `Gan` (has `.venv` environment)  
**Date**: January 26, 2026

---

## What's Been Set Up

### 1. Datasets (Moved to Gan Branch)
✓ **HelpSteer2 Full Dataset**
- File: `datasets/helpsteer2_full_30_judges_recomputed.pkl`
- Size: 53 MB
- Samples: 20,324
- Judges: 30 (5 parents + 25 children)

✓ **UltraFeedback Workshop Dataset**  
- File: `datasets/ultrafeedback_workshop_55_judges_repaired_full.pkl`
- Size: 12 MB
- Samples: 2,000
- Judges: 55

### 2. Scripts & Tools
✓ `prepare_data_for_track3.py` - Prepares datasets for GAM training  
✓ `run_track3_experiments.sh` - Automated end-to-end pipeline  
✓ `experiments/track3_automated_selection/` - Track 3 core implementation  

### 3. Documentation
✓ `TRACK3_PRUNING_EXPERIMENTS.md` - Complete step-by-step guide  
✓ `TRACK3_SETUP_COMPLETE.md` - Setup confirmation  
✓ `BRANCH_SETUP.md` - Branch organization guide  

### 4. Environment
✓ `.venv` - Complete Python environment (already configured)  
✓ All dependencies installed  
✓ Ready to use immediately  

---

## Quick Start (3 Steps)

### Step 1: Prepare Data
```bash
python3 prepare_data_for_track3.py
```
Creates:
- `datasets/helpsteer2_track3_full_dataset.pkl`
- `datasets/ultrafeedback_track3_full_dataset.pkl`

### Step 2: Run Experiments
**Option A - Automated** (recommended):
```bash
./run_track3_experiments.sh
```

**Option B - Manual** (HelpSteer2 example):
```bash
python experiments/track3_automated_selection/iterative_selection/iterative_selection.py \
    --config config/selection_experiment_prune_half.yaml \
    --data-file datasets/helpsteer2_track3_full_dataset.pkl \
    --output results/track3_full_dataset/helpsteer2_prune_half \
    --max-iterations 20
```

### Step 3: View Results
Results automatically saved to:
- `results/track3_full_dataset/helpsteer2_prune_half/`
- `results/track3_full_dataset/ultrafeedback_prune_half/`

Visualizations:
- `results/track3_full_dataset/*/visualizations/`

---

## Experiment Details

### HelpSteer2 Pruning
- **Goal**: Reduce 30 judges → 15 judges
- **Data**: 20,324 samples
- **Strategy**: Backward selection (iterative removal)
- **Duration**: 30-60 minutes
- **Metrics Tracked**: R², MAE, MSE, Spearman, composite score

### UltraFeedback Pruning  
- **Goal**: Reduce 55 judges → 27 judges
- **Data**: 2,000 samples
- **Strategy**: Backward selection (iterative removal)
- **Duration**: 15-30 minutes
- **Metrics Tracked**: R², MAE, MSE, Spearman, composite score

---

## Experiment Output Structure

```
results/track3_full_dataset/
├── helpsteer2_prune_half/
│   ├── iteration_0/
│   │   ├── result.json          # Metrics for this iteration
│   │   ├── importance_scores.json
│   │   └── config_snapshot.yaml
│   ├── iteration_1/
│   ├── ...
│   ├── iteration_19/
│   ├── visualizations/
│   │   ├── metrics_over_iterations.png
│   │   ├── judge_count_over_time.png
│   │   ├── selected_judges_timeline.png
│   │   └── composite_score_evolution.png
│   └── config.yaml
│
└── ultrafeedback_prune_half/
    ├── iteration_0/
    ├── ...
    └── visualizations/
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `prepare_data_for_track3.py` | Convert raw datasets to GAM format |
| `run_track3_experiments.sh` | Run full pipeline automatically |
| `TRACK3_PRUNING_EXPERIMENTS.md` | Detailed step-by-step instructions |
| `experiments/track3_automated_selection/iterative_selection/iterative_selection.py` | Main experiment controller |
| `experiments/track3_automated_selection/iterative_selection/visualize_selection_results.py` | Generate plots |

---

## System Requirements

- Python 3.8+
- 100+ GB disk space (for datasets + results)
- 16+ GB RAM (recommended for GAM training)
- ~2 hours total runtime

---

## Troubleshooting

### Issue: "Module not found" errors
**Solution**: Activate .venv first
```bash
source .venv/bin/activate
```

### Issue: Out of memory
**Solution**: Reduce batch size or max_iterations in config files

### Issue: Data format errors
**Solution**: Run `prepare_data_for_track3.py` again
```bash
python3 prepare_data_for_track3.py
```

See `TRACK3_PRUNING_EXPERIMENTS.md` for more troubleshooting.

---

## Next Steps

1. ✓ Switch to Gan branch (already there)
2. ✓ Activate environment: `source .venv/bin/activate`
3. → Run `python3 prepare_data_for_track3.py`
4. → Run `./run_track3_experiments.sh` (or manual commands)
5. → Review results in `results/track3_full_dataset/`
6. → Copy to "Results track 3 full dataset/" folder if needed

---

## Documentation Structure

```
Available Documentation:
├── README.md (this file)          - Overview & quick start
├── TRACK3_PRUNING_EXPERIMENTS.md  - Detailed instructions
├── TRACK3_SETUP_COMPLETE.md       - Setup confirmation
├── BRANCH_SETUP.md                - Branch organization
└── REEVALUATION_SUMMARY.md (workshop-evaluation-track3 branch)
    - Dataset generation details
```

---

## Summary

Everything is ready! The Gan branch now has:
- ✓ Full datasets (53 MB + 12 MB)
- ✓ Complete Python environment
- ✓ All necessary scripts and tools
- ✓ Comprehensive documentation

**Ready to run pruning experiments immediately!**

For detailed instructions, see `TRACK3_PRUNING_EXPERIMENTS.md`

---

**Branch**: Gan  
**Environment**: `.venv` (configured)  
**Status**: ✓ Ready  
**Last Updated**: January 26, 2026
