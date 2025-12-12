# Probabilistic Sketching for KV-Cache Compression in LLMs

This project implements Count-Min Sketch-based compression of the Key-Value cache in transformer language models to reduce memory footprint and improve inference latency while maintaining generation quality.

## Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- 16GB+ system RAM

### Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download models and datasets (automatic on first run):
```bash
python experiments/baseline.py
```

## Project Structure

```
.
├── src/
│   ├── sketches/          # Probabilistic sketch implementations
│   ├── models/            # Modified GPT-2 with sketch integration
│   └── evaluation/        # Metrics and benchmarking tools
├── experiments/           # Experiment scripts
├── notebooks/            # Analysis and visualization notebooks
├── requirements.txt      # Python dependencies
└── PROJECT.md           # Project proposal and report
```

## Usage

### Run Baseline Experiments
```bash
python experiments/baseline.py
```

### Run Sketch Compression Experiments
```bash
python experiments/sketch_experiments.py
```

### Analyze Results
```bash
jupyter notebook notebooks/analysis.ipynb
```

## Results

See `PROJECT.md` for detailed experimental results and analysis.

## Authors

Jay Fu and Gabriel Alvarado

