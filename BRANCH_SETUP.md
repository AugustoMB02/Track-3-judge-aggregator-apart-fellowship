# Branch Setup Summary

## Current Status (January 26, 2026)

### Gan Branch ✓ READY
**Location**: `/home/augustomb/Documentos/Track-3-judge-aggregator-apart-fellowship` (branch: `Gan`)

**Status**: Ready to run Track 3 pruning experiments

**New Content**:
- ✓ `datasets/helpsteer2_full_30_judges_recomputed.pkl` (53 MB, 20,324 samples)
- ✓ `datasets/ultrafeedback_workshop_55_judges_repaired_full.pkl` (12 MB, 2,000 samples)
- ✓ `prepare_data_for_track3.py` - Data preparation script
- ✓ `TRACK3_PRUNING_EXPERIMENTS.md` - Detailed instructions
- ✓ `TRACK3_SETUP_COMPLETE.md` - Setup confirmation
- ✓ `run_track3_experiments.sh` - Automated pipeline script

**Environment**:
- ✓ `.venv/` virtual environment (complete setup)
- ✓ All dependencies installed via requirements.txt
- ✓ Track 3 pipeline code ready
- ✓ GAM and MLP analysis tools configured

**How to Use**:
```bash
git checkout Gan
cd /home/augustomb/Documentos/Track-3-judge-aggregator-apart-fellowship
source .venv/bin/activate
python3 prepare_data_for_track3.py
# Then run experiments with iterative_selection.py
```

### Workshop-Evaluation-Track3 Branch
**Location**: `/home/augustomb/Documentos/Track-3-judge-aggregator-apart-fellowship` (branch: `workshop-evaluation-track3`)

**Contains**:
- ✓ `REEVALUATION_SUMMARY.md` - Documents the full dataset recomputation
- ✓ Full datasets (before transfer to Gan):
  - `datasets/helpsteer2_full_30_judges_recomputed.pkl`
  - `datasets/ultrafeedback_workshop_55_judges_repaired_full.pkl`
- ✓ Re-evaluation scripts:
  - `re_eval_helpsteer2_missing_scores.py`
  - `re_eval_workshop_55_missing_scores.py`

**Status**: Reference branch - datasets have been copied to Gan branch

## Quick Navigation

### To Run Track 3 Experiments:
```bash
git checkout Gan
./run_track3_experiments.sh  # Automated
# OR manually:
source .venv/bin/activate
python3 prepare_data_for_track3.py
python experiments/track3_automated_selection/iterative_selection/iterative_selection.py ...
```

### To Reference Dataset Generation:
```bash
git checkout workshop-evaluation-track3
cat REEVALUATION_SUMMARY.md
```

### To View Data Details:
On Gan branch:
```bash
python3 prepare_data_for_track3.py
# Creates: datasets/helpsteer2_track3_full_dataset.pkl
#          datasets/ultrafeedback_track3_full_dataset.pkl
```

## Dataset Summary

### HelpSteer2 (Full)
- **Original**: 20,324 samples with 30 judges (5 parents + 25 children)
- **File**: `datasets/helpsteer2_full_30_judges_recomputed.pkl`
- **Size**: 53 MB
- **Format**: DataFrame with `judge_scores` (array), `judge_ids` (list), `target` (float)
- **Processing**: Expands to individual judge columns for GAM training

### UltraFeedback Workshop (Full)
- **Original**: 2,000 samples with 55 judges
- **File**: `datasets/ultrafeedback_workshop_55_judges_repaired_full.pkl`
- **Size**: 12 MB
- **Format**: DataFrame with `judge_scores_55` (array), `judge_ids_55` (list), `target` (float)
- **Processing**: Expands to individual judge columns for GAM training

## Key Files on Gan Branch

```
Gan branch root/
├── .venv/                          # Complete environment
├── datasets/
│   ├── data_with_judge_scores.pkl  # Original
│   ├── helpsteer2_full_30_judges_recomputed.pkl           # NEW
│   └── ultrafeedback_workshop_55_judges_repaired_full.pkl # NEW
├── prepare_data_for_track3.py               # NEW - Data preparation
├── run_track3_experiments.sh                # NEW - Automated pipeline
├── TRACK3_PRUNING_EXPERIMENTS.md            # NEW - Full instructions
├── TRACK3_SETUP_COMPLETE.md                 # NEW - Setup confirmation
├── experiments/track3_automated_selection/
│   └── iterative_selection/
│       ├── iterative_selection.py           # Main script
│       ├── visualize_selection_results.py   # Visualization
│       ├── gap_analyzer.py
│       └── judge_set_metrics.py
├── config/
│   ├── selection_experiment_prune_half.yaml
│   ├── selection_experiment_prune_half_ultrafeedback.yaml
│   └── selection_experiment.yaml
└── results/                         # Will contain experiment outputs
```

## Experiment Workflow

1. **Activate Environment**
   ```bash
   source .venv/bin/activate
   ```

2. **Prepare Data**
   ```bash
   python3 prepare_data_for_track3.py
   ```

3. **Run Experiments** (choose one or both)
   ```bash
   # HelpSteer2 - prune 30 judges to 15
   python experiments/track3_automated_selection/iterative_selection/iterative_selection.py \
       --config config/selection_experiment_prune_half.yaml \
       --data-file datasets/helpsteer2_track3_full_dataset.pkl \
       --output results/track3_full_dataset/helpsteer2_prune_half \
       --max-iterations 20
   
   # UltraFeedback - prune 55 judges to 27
   python experiments/track3_automated_selection/iterative_selection/iterative_selection.py \
       --config config/selection_experiment_prune_half_ultrafeedback.yaml \
       --data-file datasets/ultrafeedback_track3_full_dataset.pkl \
       --output results/track3_full_dataset/ultrafeedback_prune_half \
       --max-iterations 20
   ```

4. **Generate Visualizations**
   ```bash
   python experiments/track3_automated_selection/iterative_selection/visualize_selection_results.py \
       --run-dir results/track3_full_dataset/helpsteer2_prune_half \
       --output-dir results/track3_full_dataset/helpsteer2_prune_half/visualizations
   ```

5. **Copy to Results Folder**
   ```bash
   mkdir -p "Results track 3 full dataset"
   cp -r results/track3_full_dataset/* "Results track 3 full dataset/"
   ```

## Expected Duration

- Data Preparation: 2-5 minutes
- HelpSteer2 Pruning (20 iterations): 30-60 minutes
- UltraFeedback Pruning (20 iterations): 15-30 minutes
- Visualization Generation: 5-10 minutes
- **Total: ~1-2 hours**

## Troubleshooting

See `TRACK3_PRUNING_EXPERIMENTS.md` for detailed troubleshooting and configuration options.

## Summary

Everything is set up on the **Gan branch** and ready to use. The branch has:
- ✓ Full Python environment (.venv)
- ✓ Complete dataset files
- ✓ Data preparation scripts
- ✓ Track 3 experimental pipeline
- ✓ Comprehensive documentation

**Next Action**: Run `./run_track3_experiments.sh` or follow manual steps in `TRACK3_PRUNING_EXPERIMENTS.md`
