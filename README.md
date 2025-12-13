## Probabilistic Sketching for KV-Cache Compression in LLMs

This repo contains a runnable prototype that **bounds the GPT-2 KV cache** during decoding, using simple **sketch-driven retention** to decide which cached tokens to keep under a fixed cap.

The final report is in `final_report.md` (and `final_report.pdf`).

### Quickstart (recommended)

```bash
make install
make test
make plot-main
```

- `make test` runs `test_implementation.py` (includes a small generation run).
- `make plot-main` regenerates the **main plot** referenced by `final_report.md`:
  - output: `results/figures/sketch_cache_mb_vs_max_cache_size.png`
  - input (default): `results/sketch/targeted_summary.json`

### Run experiments

```bash
# quick mode (few samples)
make experiments-quick DEVICE=cuda

# CPU fallback
make experiments-quick DEVICE=cpu

# full suite (slower)
make experiments-full DEVICE=cuda
```

### Regenerate the main report plot directly

```bash
python scripts/generate_main_plot.py \
  --input results/sketch/targeted_summary.json \
  --output results/figures/sketch_cache_mb_vs_max_cache_size.png
```

### Repo layout

```
.
├── experiments/                  # runnable experiment scripts
├── scripts/                      # plot-generation scripts
├── src/
│   ├── models/                   # SketchGPT2LMHeadModel (bounded KV cache)
│   ├── sketches/                 # Count-Min Sketch, Count-Sketch, RACE
│   └── evaluation/               # memory/latency/quality utilities
├── results/                      # JSON summaries + figures used in report
├── run_all_experiments.py        # orchestrates the full pipeline
├── test_implementation.py        # quick sanity checks
├── Makefile
└── requirements.txt
```
