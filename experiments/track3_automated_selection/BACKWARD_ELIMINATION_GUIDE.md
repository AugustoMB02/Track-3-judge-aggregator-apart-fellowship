# Backward Elimination for Optimal Judge Selection

## Goal

**Select the best 10 judges from a pool of 50+ candidates** to maximize prediction accuracy for a specific task/dataset.

## Strategy: Backward Elimination

### How It Works

1. **Start with ALL candidate judges** (e.g., 50 judges covering different aspects)
2. **Train aggregator** using all 50 judges
3. **Compute importance** for each judge using:
   - Gradient-based attribution (global impact)
   - Variance-based attribution (specialist importance)
4. **Remove the least important judge**
5. **Retrain and re-evaluate**
6. **Repeat** until you have exactly 10 judges (or until performance degrades)

### Example Run

```
Iteration 0: 50 judges → R² = 0.65
  Least important: "punctuation" (importance: 0.02)
  Remove it
  
Iteration 1: 49 judges → R² = 0.66 (improved!)
  Least important: "word_count" (importance: 0.05)
  Remove it
  
...

Iteration 40: 10 judges → R² = 0.72
  Reached target_judges = 10
  STOP
  
Final selected judges: [truthfulness, helpfulness, clarity, creativity, 
                        logical_consistency, relevance, coherence, 
                        completeness, safety, factuality]
```

---

## Quick Start

### 1. Prepare Your Judge Pool

Create a YAML file with ALL your candidate judges:

```yaml
# judges/my_judge_pool.yaml
truthfulness:
  name: "Truthfulness"
  description: "Factual accuracy and correctness"
  scale: "1-5"
  
helpfulness:
  name: "Helpfulness"
  description: "Usefulness for the user"
  scale: "1-5"
  
# ... 48 more judges
```

### 2. Configure the Experiment

Edit `config/backward_selection_example.yaml`:

```yaml
initial_judge_file: "judges/my_judge_pool.yaml"  # Your pool
target_judges: 10  # Goal: select 10
max_iterations: 50  # Should be >= pool_size - target
data_file: "datasets/processed/my_task.pkl"
```

### 3. Run Selection

**With GPU (recommended):**
```bash
python run_mlp_selection.py \
    --config config/backward_selection_example.yaml \
    --gpu \
    --output results/select_best_10
```

**With CPU:**
```bash
python run_mlp_selection.py \
    --config config/backward_selection_example.yaml \
    --cpu \
    --output results/select_best_10
```

### 4. Review Results

```bash
# See final selected judges
cat results/select_best_10/final_judges.txt

# See performance metrics
cat results/select_best_10/final_results.json
```

---

## Understanding the Output

### Terminal Output

```
Iteration 0: Starting with 50 judges
  Test R²: 0.6523
  Removing: punctuation (importance: 0.0234)
  
Iteration 1: 49 judges remaining
  Test R²: 0.6587 (+0.0064)
  Removing: word_count (importance: 0.0456)
  
...

Iteration 40: 10 judges remaining
  Test R²: 0.7234
  Stop reason: target_judges_reached_10
  
✅ Selected 10 optimal judges!
```

### Final Judge Ranking

```
Rank  Judge                    Importance
  1   truthfulness                 0.8923
  2   helpfulness                  0.8567
  3   logical_consistency          0.8234
  4   clarity                      0.7891
  5   relevance                    0.7645
  6   creativity                   0.7123
  7   coherence                    0.6987
  8   completeness                 0.6754
  9   safety                       0.6432
 10   factuality                   0.6210
```

---

## Configuration Parameters

### Key Settings for Backward Elimination

```yaml
# Start with full pool
initial_judge_file: "judges/full_pool.yaml"  # All candidates

# Target selection
target_judges: 10  # Select exactly this many

# Safety limits
min_judges: 5  # Never go below this (safety)
max_iterations: 50  # Max removal steps

# Performance thresholds
r2_degradation_threshold: 0.02  # Stop if R² drops >2%
plateau_patience: 3  # Stop if 3 iterations without improvement

# Protected judges (optional)
protected_judges:  # These will never be removed
  - "truthfulness"  # If you MUST keep certain judges
```

---

## When to Use Backward Elimination

### Best For:
✅ Moderate-sized pools (20-100 candidates)
✅ When you have a clear target number (e.g., budget for 10 API calls)
✅ When you want the "optimal subset" not just "minimal viable"
✅ Datasets with 500+ samples (more stable importance estimates)

### Not Ideal For:
❌ Huge pools (500+ judges) → Too slow
❌ Very small pools (<15 judges) → Use forward selection or exhaustive search
❌ When you want the absolute minimum judges → Use reduction mode instead

---

## Advanced Usage

### Protecting Must-Have Judges

If you know certain judges MUST be included:

```yaml
protected_judges:
  - "truthfulness"
  - "safety"
```

These will never be removed, so the algorithm selects the best 8 additional judges to complement them.

### Custom Stopping Criteria

**Stop early if performance degrades:**
```yaml
r2_degradation_threshold: 0.01  # Stop if R² drops by 1%
```

**Allow more exploration:**
```yaml
plateau_patience: 5  # Give it 5 iterations to improve
```

### Task-Specific Selection

Run separate selections for different tasks:

```bash
# Code generation task
python run_mlp_selection.py \
    --config config/backward_selection_example.yaml \
    --output results/code_gen_judges
    
# Creative writing task
python run_mlp_selection.py \
    --config config/backward_selection_creative.yaml \
    --output results/creative_judges
    
# Compare which judges were selected
diff results/code_gen_judges/final_judges.txt \
     results/creative_judges/final_judges.txt
```

---

## Performance Expectations

### With GPU (RTX 3090)

| Pool Size | Target | Time (Total) | Time/Iteration |
|-----------|--------|--------------|----------------|
| 20 judges | 10     | ~5 minutes   | 30s            |
| 50 judges | 10     | ~20 minutes  | 30s            |
| 100 judges| 10     | ~45 minutes  | 30s            |

### With CPU (16 cores)

| Pool Size | Target | Time (Total) | Time/Iteration |
|-----------|--------|--------------|----------------|
| 20 judges | 10     | ~40 minutes  | 4 min          |
| 50 judges | 10     | ~2.5 hours   | 4 min          |
| 100 judges| 10     | ~6 hours     | 4 min          |

**Speedup with GPU:** ~8x faster

---

## Comparison with Other Approaches

### Backward Elimination (This)
- **Start:** All 50 judges
- **End:** Best 10 judges
- **Time:** O(pool_size - target) iterations
- **Advantage:** Finds globally optimal subset
- **Disadvantage:** Slow for huge pools

### Forward Selection
- **Start:** 0 or 1 judge
- **End:** Best 10 judges
- **Time:** O(target) iterations
- **Advantage:** Fast for small targets
- **Disadvantage:** Greedy, might miss combinations

### Genetic Algorithm
- **Start:** Random populations
- **End:** Best 10 judges
- **Time:** O(generations × population_size)
- **Advantage:** Explores many combinations
- **Disadvantage:** Stochastic, needs tuning

### Exhaustive Search
- **Tests:** C(50, 10) = 10 billion combinations
- **Time:** Infeasible for pools >15
- **Advantage:** Guaranteed optimal
- **Disadvantage:** Only works for tiny pools

---

## Troubleshooting

### "Performance keeps improving, never stops"

**Cause:** `r2_degradation_threshold` too strict

**Solution:**
```yaml
r2_degradation_threshold: 0.03  # Allow small drops
plateau_patience: 2  # Stop sooner
```

### "Removes important judges too early"

**Cause:** Importance calculation unstable

**Solution:**
```yaml
protected_judges:  # Manually protect
  - "truthfulness"
  
# Or use larger validation set
validation_split: 0.2  # Default is 0.15
```

### "Takes too long on CPU"

**Solutions:**
1. Use GPU: `--gpu`
2. Reduce MLP epochs: `--epochs 50`
3. Smaller hidden dim: `--hidden-dim 32`
4. Start with pre-filtered pool (e.g., top 30 judges from prior analysis)

---

## Integration with Track 1 & Track 2

### From Track 1: Multi-Task Selection

```bash
# For each JUDGE-BENCH task
for task in summeval newsroom recipe; do
    python run_mlp_selection.py \
        --config config/backward_${task}.yaml \
        --gpu \
        --output results/optimal_judges_${task}
done

# Analyze cross-task patterns
python analysis/compare_task_selections.py results/optimal_judges_*
```

### From Track 2: Importance Analysis

After selection, run Track 2 interpretability on the final 10:

```bash
python experiments/track2_judge_interpretability/explainability/interp_pipeline.py \
    --judges results/select_best_10/final_judges.txt \
    --output results/interpretability_analysis
```

This shows WHY each judge was selected.

---

## Summary

**Backward elimination is ideal when you:**
- Have 20-100 candidate judges
- Want to select a specific number (e.g., 10)
- Need the optimal subset, not just minimal
- Have GPU for faster iterations

**The algorithm guarantees:**
- You get exactly `target_judges` judges
- They are the best performing subset from your pool
- Importance is calculated using both global and local metrics
- Results are reproducible and interpretable
