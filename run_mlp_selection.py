#!/usr/bin/env python3
"""
GPU-Accelerated Judge Selection Runner

Quick script to run iterative judge selection with MLP on GPU.

Usage:
    python run_mlp_selection.py --config config/selection_experiment.yaml --gpu
    python run_mlp_selection.py --config config/selection_experiment.yaml --cpu
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from experiments.track3_automated_selection.iterative_selection.mlp_selector import (
    MLPJudgeSelector,
    SelectionConfig,
)

def main():
    parser = argparse.ArgumentParser(description="Run GPU-accelerated judge selection")
    parser.add_argument(
        "--config",
        type=str,
        default="config/selection_experiment.yaml",
        help="Path to selection config YAML",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration (default if CUDA available)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage even if GPU available",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override max iterations from config",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="MLP hidden dimension (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max training epochs (default: 100)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate (default: 0.2)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GPU-ACCELERATED JUDGE SELECTION - BACKWARD ELIMINATION")
    print("=" * 80)
    print("\nStrategy: Start with ALL candidate judges, iteratively remove")
    print("          the least important until reaching target number.")
    print("=" * 80)
    
    # Load config
    config = SelectionConfig.from_yaml(args.config)
    
    # Apply overrides
    if args.max_iterations is not None:
        config.max_iterations = args.max_iterations
    if args.output is not None:
        config.output_dir = args.output
    
    # Determine device
    device = "cpu" if args.cpu else "cuda"
    
    print(f"\nConfiguration:")
    print(f"  Config file: {args.config}")
    print(f"  Data file: {config.data_file}")
    print(f"  Target: {config.target_column}")
    print(f"  Starting judges: {len(config.initial_judge_file.split(',')) if ',' in config.initial_judge_file else 'all in file'}")
    print(f"  Target judges: {config.target_judges}")
    print(f"  Max iterations: {config.max_iterations}")
    print(f"  Output: {config.output_dir}")
    print(f"\nMLP Settings:")
    print(f"  Device: {device}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max epochs: {args.epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Dropout: {args.dropout}")
    
    # Initialize selector
    print("\n" + "=" * 80)
    print("INITIALIZATION")
    print("=" * 80)
    
    selector = MLPJudgeSelector(
        config=config,
        device=device,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        dropout=args.dropout,
        l2_reg=0.001,
        early_stopping_patience=15,
    )
    
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
        print(f"Final test Spearman ρ: {results[-1].test_metrics.get('spearman_rho', 0):.4f}" if results else "N/A")
        print(f"Stop reason: {results[-1].stop_reason if results else 'N/A'}")
        
        print("\nIteration progression:")
        print(f"{'Iter':>4} {'Judges':>7} {'R²':>8} {'Spearman':>10} {'Removed':>20}")
        print("-" * 60)
        for r in results:
            removed = r.removed_judge or "none"
            print(f"{r.iteration:4d} {r.n_judges:7d} {r.test_metrics.get('r2', 0):8.4f} "
                  f"{r.test_metrics.get('spearman_rho', 0):10.4f} {removed:>20}")
        
        print(f"\n✅ Selection complete! Results saved to {config.output_dir}")
        
        # Print final judge set
        if results:
            print("\nFinal selected judges:")
            final_judges = results[-1].judge_names
            final_importance = results[-1].importance_scores
            
            # Sort by importance
            sorted_judges = sorted(
                final_judges,
                key=lambda j: final_importance.get(j, 0),
                reverse=True,
            )
            
            print(f"{'Rank':>4} {'Judge':>30} {'Importance':>12}")
            print("-" * 50)
            for i, judge in enumerate(sorted_judges, 1):
                print(f"{i:4d} {judge:>30} {final_importance.get(judge, 0):12.4f}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
