#!/usr/bin/env python3
"""
Plot Track 1 HelpSteer2 full baseline results.
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HelpSteer2 full baseline results")
    parser.add_argument("results_dir", help="Path to results/track1_helpsteer2_full_<timestamp>")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    csv_path = results_dir / "summary_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)
    pivot = df.pivot_table(index="dimension", columns="method", values="r2")
    order = ["gam", "mean", "median", "max", "best_single_judge"]
    pivot = pivot[[c for c in order if c in pivot.columns]]

    plt.figure(figsize=(8, 4.8))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker="o", label=col)
    plt.axhline(0, color="gray", linewidth=1, alpha=0.4)
    plt.ylabel("Test R²")
    plt.title("HelpSteer2 Full: GAM vs Baselines (All Judges)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_dir = results_dir / "plots"
    out_dir.mkdir(exist_ok=True)
    out_png = out_dir / "helpsteer2_full_gam_vs_baselines.png"
    out_pdf = out_dir / "helpsteer2_full_gam_vs_baselines.pdf"
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()
