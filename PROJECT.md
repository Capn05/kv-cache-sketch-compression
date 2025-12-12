## Probabilistic Sketching for KV-Cache Compression in LLMs – Final Report (GPT-2)

## Problem
Transformer inference commonly uses a key–value (KV) cache so each new token can attend over prior tokens without recomputing prior hidden states. The KV cache grows linearly with sequence length, increasing memory usage and attention cost.

This project explores whether probabilistic sketches (specifically **Count-Min Sketch**) can support KV-cache **compression via eviction** while remaining compatible with Hugging Face `generate()`.

## What we implemented
- **Model**: GPT-2 small (`gpt2`, 117M) via Hugging Face Transformers.
- **Sketch**: Count-Min Sketch (`src/sketches/count_min_sketch.py`).
- **Compression**: eviction-based KV-cache cap by physically capping `past_key_values` to `max_cache_size` during generation.
- **Integration**: Transformers can represent `past_key_values` as a `DynamicCache`. We cap the cache in `SketchGPT2LMHeadModel.forward()` by updating each layer’s `keys/values` so the next generation step uses the compressed cache (`src/models/sketch_gpt2.py`).

## Method (eviction-only)
For each transformer layer:
1. Maintain a Count-Min Sketch per head; update it with each new token’s key vector.
2. When KV length exceeds `max_cache_size`, compute an importance score per position (sketch query over stored keys), keep the top positions, and **return a capped KV cache**.

This final implementation focuses on KV eviction. We do **not** claim a sketch-based attention approximation method (e.g., aggregate attention) in the final evaluation.

## Experimental setup
### Hardware
- **GPU**: NVIDIA A100-SXM4-40GB (Lambda `gpu_1x_a100_sxm4`)

### Dataset / prompts
- WikiText-2 test split; prompts truncated to ~50 tokens

### Total lengths
- Results reported at **total lengths 256 and 512** (prompt length + generated tokens).

### Baselines
- **Full cache**: vanilla GPT-2 with standard KV cache (`experiments/baseline.py`)
- **Sliding window** and **quantization**: included as comparisons (`experiments/baselines_comparison.py`), but note this script states these are simplified measurement proxies.

### Targeted sketch grid (12 configs)
- `sketch_width ∈ {256, 512}`
- `sketch_depth ∈ {2, 4}`
- `max_cache_size ∈ {64, 128, 256}`

## Results
### Full-cache baseline (GPT-2)
From `results/baseline/baseline_summary.json`:
- **Total length 256**
  - Peak memory: **279.07 MB**
  - Latency: **7.23 ms/token**
  - Throughput: **139.45 tokens/sec**
- **Total length 512**
  - Peak memory: **288.07 MB**
  - Latency: **7.03 ms/token**
  - Throughput: **142.33 tokens/sec**
- **Perplexity** (WikiText-2 subset): **49.10**

### KV-cache compression works (cache size scales with `max_cache_size`)
From `results/sketch/targeted_summary.json`:
- **Total length 256**
  - `max_cache_size=64`: cache **2.25 MB**
  - `max_cache_size=128`: cache **4.50 MB**
  - `max_cache_size=256`: cache **8.96 MB**
- **Total length 512**
  - `max_cache_size=64`: cache **2.25 MB**
  - `max_cache_size=128`: cache **4.50 MB**
  - `max_cache_size=256`: cache **9.00 MB**

Figures:
- `results/figures/sketch_cache_mb_vs_max_cache_size.png`
- `results/figures/sketch_total_memory_vs_max_cache_size.png`

### Performance trade-off (latency overhead)
Throughput is lower than the full-cache baseline due to per-step eviction scoring (per layer, per generation step).

Best sketch throughput configs observed:
- **Total length 256**: **28.36 tokens/sec** (`width=512, depth=2, max_cache_size=256`)
- **Total length 512**: **19.65 tokens/sec** (`width=256, depth=2, max_cache_size=256`)

Figure:
- `results/figures/throughput_baseline_vs_best_sketch.png`

## Discussion
### What worked
- End-to-end **integration with `generate()`**: cache is populated and capped during generation.
- Cache memory **scales with the compression knob** `max_cache_size`.

### Limitations / caveats
- This is **eviction**, not attention approximation (no aggregate/top-k attention approximation claims).
- The current eviction scoring is expensive; throughput drops substantially vs full cache.
- `sketch_memory_mb` includes bookkeeping used for analysis, not just the CM sketch table.
- Sliding-window and quantization baselines here are simplified measurement proxies.

## Reproduction (A100)
### Install
```bash
python -m pip install -r requirements.txt
```

### Run tests
```bash
python test_implementation.py
```

### Baseline (full cache)
```bash
python experiments/baseline.py --device cuda --seq-lengths 256 512 --num-samples 10 --output-dir results/baseline
```

### Sketch targeted grid (eviction)
```bash
python experiments/sketch_experiments.py --device cuda --num-samples 10 --total-lengths 256 512 --grid targeted --output-dir results/sketch
```

### Baseline comparisons
```bash
python experiments/baselines_comparison.py --device cuda --num-samples 10 --total-lengths 256 512 --output-dir results/baselines
```

### Plots
See `notebooks/final_plots.ipynb` and generated figures under `results/figures/`.

## References
- Taming the Fragility of KV Cache Eviction in LLM Inference: `https://arxiv.org/html/2510.13334v1`
- KV Cache Eviction in Transformer LLMs: `https://www.emergentmind.com/topics/kv-cache-eviction`
