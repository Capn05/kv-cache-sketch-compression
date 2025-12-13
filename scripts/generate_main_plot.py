"""Generate the *main plot* used in `final_report.md`.

The report references:
- `results/figures/sketch_cache_mb_vs_max_cache_size.png`

This script reproduces that figure from a sketch experiment summary JSON.

Default input matches the plotting notebook (`notebooks/final_plots.ipynb`):
- `results/sketch/targeted_summary.json`

Usage:
    python scripts/generate_main_plot.py
    python scripts/generate_main_plot.py --input results/sketch_targeted_topk_full/targeted_summary.json

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def build_df(summary: dict) -> pd.DataFrame:
    rows = []
    for r in summary.get("results", []):
        cfg = r.get("config", {})
        rows.append(
            {
                "total_length": int(r.get("total_length")),
                "max_cache_size": int(cfg.get("max_cache_size")),
                "cache_memory_mb": float(r.get("cache_memory_mb")),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No rows found in summary['results']")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="results/sketch/targeted_summary.json",
        help="Path to a targeted summary JSON (produced by experiments/sketch_experiments.py).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/figures/sketch_cache_mb_vs_max_cache_size.png",
        help="Output PNG path.",
    )
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    summary = read_json(inp)
    df = build_df(summary)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7, 4))
    sns.lineplot(
        data=df,
        x="max_cache_size",
        y="cache_memory_mb",
        hue="total_length",
        marker="o",
    )
    plt.title("Sketch eviction: cache MB vs max_cache_size")
    plt.ylabel("cache MB (all layers)")
    plt.xlabel("max_cache_size (tokens)")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
