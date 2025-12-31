#!/usr/bin/env python3
"""
Quick test of the iterative selection pipeline with minimal iterations.
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from experiments.track3_automated_selection.iterative_selection.iterative_selection import (
    SelectionConfig,
    IterativeJudgeSelector,
)

def main():
    print("=" * 80)
    print("ITERATIVE JUDGE SELECTION PIPELINE - TEST RUN")
    print("=" * 80)
    
    # Load config
    config = SelectionConfig.from_yaml('config/selection_experiment.yaml')
    
    # Override for quick test
    config.max_iterations = 3  # Just 3 iterations for testing
    config.use_llm_suggestions = False  # Disable LLM to avoid API calls
    config.save_intermediate = True
    config.output_dir = 'results/test_selection_run'
    
    print(f"\nConfiguration:")
    print(f"  Data file: {config.data_file}")
    print(f"  Target: {config.target_column}")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Min judges: {config.min_judges}")
    print(f"  Output: {config.output_dir}")
    
    # Initialize selector
    print("\n" + "=" * 80)
    print("INITIALIZATION")
    print("=" * 80)
    selector = IterativeJudgeSelector(config)
    
    # Run selection
    print("\n" + "=" * 80)
    print("RUNNING ITERATIVE SELECTION")
    print("=" * 80)
    
    try:
        results = selector.run()
        
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"Total iterations: {len(results)}")
        print(f"Final judge count: {results[-1].n_judges if results else 0}")
        print(f"Final test R²: {results[-1].test_metrics.get('r2', 0):.4f}" if results else "N/A")
        print(f"Stop reason: {results[-1].stop_reason if results else 'N/A'}")
        
        print("\nIteration progression:")
        for r in results:
            print(f"  Iter {r.iteration}: {r.n_judges} judges, R²={r.test_metrics.get('r2', 0):.4f}, "
                  f"removed={r.removed_judge or 'none'}")
        
        print(f"\n✅ Test complete! Results saved to {config.output_dir}")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
