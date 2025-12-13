## Quick Start

This repo is set up so you can:
- run a quick sanity test,
- run the experiment pipeline,
- regenerate the **main plot** referenced by `final_report.md`.

### 1) Install

From the repo root:

```bash
python3 -m venv venv
source venv/bin/activate
make install
```

### 2) Quick sanity test

```bash
make test
```

This downloads GPT-2 on first run and verifies:
- sketch data structures
- `SketchGPT2LMHeadModel` generation + cache capping
- experiment modules import

### 3) Run experiments

Quick mode (few samples):

```bash
make experiments-quick DEVICE=cuda
```

CPU mode:

```bash
make experiments-quick DEVICE=cpu
```

Full mode (slower):

```bash
make experiments-full DEVICE=cuda
```

### 4) Regenerate the main plot used in the report

The main plot in `final_report.md` is:
- `results/figures/sketch_cache_mb_vs_max_cache_size.png`

Generate it with:

```bash
make plot-main
```

or directly:

```bash
python scripts/generate_main_plot.py \
  --input results/sketch/targeted_summary.json \
  --output results/figures/sketch_cache_mb_vs_max_cache_size.png
```

### Outputs

- **Experiment JSON**: `results/**.json`
- **Figures**: `results/figures/*.png`
