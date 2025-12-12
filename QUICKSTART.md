# Quick Start Guide

This guide will help you get started with the KV-Cache Sketch Compression project quickly.

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended: 8GB+ VRAM)
- 16GB+ system RAM

## Installation

### 1. Clone/Navigate to Project Directory

```bash
cd "C:\Users\gxalv\coding_projects\comp480\Final project"
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the environment:
- Windows: `venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

### 3. Install Dependencies

```bash
pip install torch transformers datasets numpy matplotlib seaborn scipy tqdm pandas scikit-learn accelerate
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## Quick Test

Verify the implementation is working:

```bash
python test_implementation.py
```

This will test all core components without running full experiments (takes ~1-2 minutes).

## Running Experiments

### Option 1: Quick Mode (Recommended for First Run)

Run a minimal experiment to verify everything works:

```bash
python run_all_experiments.py --quick --device cuda
```

This uses only 3 samples per configuration and takes ~10-15 minutes.

### Option 2: Full Experiments

Run the complete experimental suite:

```bash
python run_all_experiments.py --device cuda
```

This takes 1-2 hours depending on your GPU.

### Option 3: Individual Experiments

Run specific components:

```bash
# Baseline only (full KV-cache)
python experiments/baseline.py --device cuda --num-samples 10

# Sketch compression experiments
python experiments/sketch_experiments.py --device cuda --num-samples 5

# Baseline comparisons (sliding window, quantization)
python experiments/baselines_comparison.py --device cuda --num-samples 10

# Attention distribution analysis
python experiments/attention_analysis.py --device cuda --num-samples 5
```

### CPU Mode

If you don't have a GPU:

```bash
python run_all_experiments.py --quick --device cpu
```

**Note:** CPU mode is much slower and not recommended for full experiments.

## Analyzing Results

### 1. Open Analysis Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

### 2. Run All Cells

The notebook will:
- Load experimental results
- Generate comparison plots
- Print summary statistics
- Save visualizations to `results/`

### 3. View Results

Results are saved in the `results/` directory:

```
results/
├── baseline/               # Full KV-cache results
│   ├── baseline_summary.json
│   └── baseline_seq*.json
├── sketch/                 # Sketch compression results
│   ├── experiment_summary.json
│   └── config_*.json
├── baselines/              # Comparison baselines
│   ├── sliding_window.json
│   └── quantization.json
├── attention_analysis/     # Attention distribution analysis
│   └── analysis_summary.json
└── *.png                   # Generated plots
```

## Expected Outputs

After running experiments, you should see:

### 1. Memory Comparison
- Full cache baseline: ~300-500 MB (varies by sequence length)
- Sketch compression: ~100-200 MB (depending on configuration)
- **Expected savings: 40-70%**

### 2. Latency
- Full cache: Increases linearly with sequence length
- Sketch cache: More stable, slight overhead from sketch operations

### 3. Quality Metrics
- Perplexity should be close to baseline (within 1-5%)
- BLEU scores for generation tasks

### 4. Attention Analysis
- KL divergence between full and sketch attention
- Cosine similarity metrics

## Troubleshooting

### Out of Memory Error

If you run out of GPU memory:

1. Use quick mode: `--quick`
2. Reduce samples: `--num-samples 3`
3. Use CPU mode: `--device cpu`
4. Skip some experiments: `--skip-attention --skip-comparisons`

### Import Errors

Make sure you're in the project root directory and have installed all dependencies.

### Slow Performance

- Ensure you're using GPU mode (`--device cuda`)
- Check that CUDA is properly installed: `python -c "import torch; print(torch.cuda.is_available())"`
- Start with quick mode to verify setup

### No Results Files

Make sure experiments completed successfully. Check console output for errors.

## Next Steps

1. **Run Quick Test**: `python test_implementation.py`
2. **Run Quick Experiments**: `python run_all_experiments.py --quick --device cuda`
3. **Analyze Results**: Open `notebooks/analysis.ipynb`
4. **Read Full Results**: See `PROJECT.md` for detailed analysis

## Configuration Options

### Sketch Parameters

Edit experiment scripts to modify:

- **Sketch Width**: `[128, 256, 512, 1024]` - Controls memory/accuracy trade-off
- **Sketch Depth**: `[2, 4, 6]` - Number of hash functions
- **Strategy**: `'topk'` or `'aggregate'` - Attention approximation method
- **Top-K**: `[32, 64, 128]` - Number of tokens to attend to (for top-k strategy)

### Experimental Settings

Modify in `run_all_experiments.py`:

- `num_samples`: Number of test samples per configuration
- `seq_lengths`: Sequence lengths to test `[128, 256, 512, 1024]`
- `max_tokens`: Maximum tokens to generate per sample

## Getting Help

- **Implementation Details**: See `PROJECT.md`
- **Code Documentation**: Check docstrings in source files
- **Architecture Overview**: See `README.md`

## Summary of Files

| File | Purpose |
|------|---------|
| `test_implementation.py` | Quick verification tests |
| `run_all_experiments.py` | Master experiment runner |
| `requirements.txt` | Python dependencies |
| `PROJECT.md` | Full project documentation |
| `README.md` | Project overview |
| `QUICKSTART.md` | This guide |

## Estimated Time Requirements

| Task | Time (GPU) | Time (CPU) |
|------|-----------|------------|
| Quick test | 1-2 min | 5 min |
| Quick experiments | 10-15 min | 1-2 hours |
| Full experiments | 1-2 hours | 6-12 hours |
| Analysis | 5 min | 5 min |

**Recommendation**: Start with quick test, then quick experiments, then full experiments.

