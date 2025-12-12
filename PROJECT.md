Abstract
Jay Fu and Gabriel Alvarado
During autoregressive inference in transformer-based large language models (LLMs), every generated token reuses a growing cache of past attention keys and values. This KV-cache grows linearly with sequence length, leading to large GPU memory footprints and latency bottlenecks [1]. While existing works mitigate this through quantization or eviction [2], these methods can be computationally expensive or harm accuracy. We propose using probabilistic sketches(compact data structures that approximate aggregate information)to efficiently compress and approximate KV-cache representations in real time.
Our project will implement three probabilistic sketch families: Count-Min Sketch [3], Count-Sketch [4], and RACE Sketch [5]. Each maintains approximate summaries of attention key statistics or token embeddings, enabling retrieval of representative values without storing the full cache. We will extend this idea by developing an adaptive sketch controller that dynamically adjusts sketch width and hash function count based on observed token entropy or sequence redundancy, ensuring higher precision for diverse contexts and higher compression for repetitive ones.
Experiments will be conducted using open-source LLMs such as GPT-2 small (117M) or TinyLlama (1.1B) within the Hugging Face Transformers framework. We will evaluate (1) memory savings, (2) average inference latency, and (3) text-generation quality metrics (perplexity and BLEU) across different sequence lengths and sketch configurations. Analytical modeling will further estimate error propagation in attention weights as a function of sketch parameters, linking probabilistic theory to empirical results.
By combining theoretical analysis and empirical benchmarking, this project aims to demonstrate that probabilistic sketching provides a lightweight and mathematically grounded mechanism for scalable, low-latency LLM inference, a practical application of probabilistic algorithms in modern AI systems.
References
[1] Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
[2] Dao, T. et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS.
[3] Cormode, G., & Muthukrishnan, S. (2005). An Improved Data Stream Summary: The Count-Min Sketch and its Applications. J. of Algorithms.
[4] Charikar, M., Chen, K., & Farach-Colton, M. (2002). Finding Frequent Items in Data Streams. ICALP.
[5] Charikar, M. et al. (2021). RACE Sketches for Approximate Similarity Search. arXiv:2105.15028.


Probabilistic Sketching for KV-Cache Compression in LLMs – Mid-Project Report
Problem Description
Transformer-based large language models (LLMs) use a key–value (KV) cache during autoregressive inference to store attention states for all past tokens. This cache grows linearly with sequence length, so long contexts quickly become a major bottleneck for both GPU memory and latency: each new token must read and process an ever-larger cache, which limits batch size, slows throughput, and increases serving costs. Naively discarding old tokens (sliding windows) or recomputing states can either destroy long-range context or add significant overhead, while simply increasing hardware capacity is economically unattractive.
We propose to replace the full KV cache with probabilistic sketch data structures that maintain compact, approximate summaries of the past. Concretely, we plan to implement three sketch families—Count-Min Sketch, Count-Sketch, and RACE Sketch—to summarize attention keys and/or token embeddings and to support approximate retrieval of representative values without storing every key–value pair. On top of these static sketches, we will design an adaptive sketch controller that adjusts sketch width and number of hash functions in real time based on sequence statistics (e.g., token entropy or redundancy), allocating more capacity to diverse, unpredictable contexts and more compression to repetitive ones.
Literature Survey
Prior work on KV-cache efficiency has largely focused on eviction strategies. Sliding-window and streaming-style approaches keep only the most recent tokens, substantially reducing memory but sacrificing long-range dependencies needed for tasks with long prompts or documents. More advanced methods estimate a token’s “importance” based on attention patterns and selectively retain only high-importance tokens, often within a score-and-aggregate framework. These methods can keep 20–50% of cache entries with moderate quality loss, but they rely on assumptions about the stability of token importance over time and can fail when old tokens suddenly become relevant again.
A second line of work uses quantization and mixed precision to compress the KV cache without removing entries. Techniques that quantize keys and values to 8, 4, or even 2 bits (often with per-channel scaling and outlier handling) achieve large memory savings while maintaining near-baseline accuracy. Mixed-precision schemes further allocate high precision to “important” tokens and low precision to less important ones, driving overall cache size down to a small fraction of the original. These methods show that approximate representations of cached states are viable, but they still store every token and may require complex calibration or fine-tuning.
System-level optimizations such as paged KV caches and attention kernels like FlashAttention improve I/O efficiency and memory layout by tiling computations and offloading parts of the cache to CPU or disk. They help models handle longer contexts on limited GPU memory but do not fundamentally change the linear growth of cache size with sequence length. In contrast, probabilistic sketches like Count-Min Sketch, Count-Sketch, and RACE were originally developed for streaming frequency estimation and similarity search, offering sublinear memory with probabilistic error guarantees. To our knowledge, these sketches have not yet been applied to KV-cache compression in LLMs, leaving a gap our project aims to explore.
Hypothesis
Use case 1 – Static sketch compression: We hypothesize that a fixed-size probabilistic sketch (Count-Min Sketch, Count-Sketch, or RACE) can compress the KV cache to at most 20–30% of its original memory while keeping perplexity and BLEU within ≈1–2% of the full-cache baseline.
Use case 2 – Adaptive sketch controller: We argue that dynamically adjusting sketch size and hash depth based on token entropy and sequence redundancy will outperform any fixed sketch configuration on the quality–memory trade-off, achieving equal or better generation quality at the same memory budget.
Experimental Settings
We will evaluate our methods on open-source LLMs implemented in the Hugging Face Transformers framework, primarily GPT-2 Small (117M) and TinyLlama (~1.1B). These models provide realistic attention mechanisms and KV caching behavior while still being tractable on a single modern GPU. All experiments will be run in autoregressive generation mode with sequence lengths ranging from short prompts (128–256 tokens) up to long contexts (1024–2048 tokens), using mixed-precision inference (FP16 or BF16) to match common deployment settings.
For static sketch compression (use case 1), we will implement three sketch variants per attention head: Count-Min Sketch, Count-Sketch, and RACE. Each sketch will be instantiated with configurable width (e.g., 64–512 buckets) and depth (e.g., 2–6 hash functions), chosen so that the total sketch memory corresponds to a target fraction (e.g., 10–30%) of the original KV cache footprint. The model’s forward pass will be modified so that when a new token is generated, its key/value are fed into the sketch rather than appended to a full cache, and during attention the query interacts with the sketch to either retrieve a small set of representative keys/values or an aggregated approximate value.
For adaptive sketching (use case 2), we will add a controller that monitors simple statistics of the recent token stream, such as rolling token entropy, repetition rate (e.g., proportion of n-gram repeats), and possibly changes in model perplexity on the fly. Based on these signals, the controller will resize or reparameterize the sketch—e.g., increasing width/depth when entropy is high (to reduce approximation error in diverse contexts) and shrinking them when the input is highly repetitive (to exploit redundancy and save memory). We will start with rule-based policies (thresholds on entropy and repetition) and, if time permits, explore learned controllers that optimize a reward combining memory usage and quality.
Our datasets and workloads will include a language modeling benchmark such as WikiText-2 (and possibly WikiText-103) to measure perplexity under long-context evaluation, as well as one or two generation tasks that stress long-range context, such as document summarization or long-form QA. For each dataset, we will run the full-cache model and each compressed variant over the same prompts, recording sequence lengths, outputs, and all relevant metrics. We will also construct synthetic stress tests: one with highly repetitive input (e.g., repeated paragraphs) and one with rapidly changing topics, to explicitly probe how static vs adaptive sketches behave under low- and high-entropy regimes.
We will compare against several baselines: (1) the uncompressed full KV cache (upper bound on quality, worst case for memory and latency); (2) a sliding-window cache that keeps only the most recent M tokens (strong memory savings but limited context); and (3) a straightforward quantized KV cache (e.g., int8 or int4 keys/values with uniform scaling). For all methods, we will log peak GPU memory usage (from PyTorch CUDA stats), average time per generated token at different sequence lengths, and quality metrics (perplexity on language modeling and BLEU/ROUGE or task-specific scores on generation tasks). Additionally, for a subset of steps we will compare the true attention distribution from the full cache with the approximate distribution induced by the sketches using KL divergence or cosine similarity, relating sketch parameters to attention error and, in turn, to downstream quality.

Implementation and Progress

We have successfully implemented a complete experimental framework for sketch-based KV-cache compression in GPT-2. The implementation consists of the following components:

**Core Implementation:**

1. **Count-Min Sketch Data Structure** (`src/sketches/count_min_sketch.py`): We implemented a GPU-accelerated Count-Min Sketch with configurable width and depth, supporting batched operations for efficient integration with transformer models. The sketch includes methods for updating with key-value pairs, querying frequency estimates, and retrieving top-k elements.

2. **Modified GPT-2 Model** (`src/models/sketch_gpt2.py`): We created a modified version of GPT-2 that replaces the standard KV-cache with our sketch-based implementation. The model supports two attention approximation strategies:
   - **Top-K Selection**: Uses the sketch to identify the k most important past tokens and computes attention only over those tokens
   - **Aggregate Approximation**: Maintains exact cache for recent tokens while using the sketch for approximate representation of older context

3. **Evaluation Framework** (`src/evaluation/metrics.py`): Comprehensive metrics tracking for memory usage (GPU/CPU), latency (per-token timing), and quality metrics (perplexity, BLEU, attention distribution similarity).

**Experimental Infrastructure:**

- `experiments/baseline.py`: Measures full KV-cache performance across different sequence lengths
- `experiments/sketch_experiments.py`: Tests a grid of sketch configurations (width, depth, strategy, top-k values)
- `experiments/baselines_comparison.py`: Implements sliding window and quantization baselines
- `experiments/attention_analysis.py`: Analyzes attention distribution differences using KL divergence and cosine similarity
- `notebooks/analysis.ipynb`: Visualization and analysis notebook for generating plots and summaries
- `run_all_experiments.py`: Master script to orchestrate the complete experimental pipeline

**How to Run the Experiments:**

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run all experiments (quick mode for testing):
```bash
python run_all_experiments.py --quick --device cuda
```

3. Run full experiments:
```bash
python run_all_experiments.py --device cuda
```

4. Run individual experiment components:
```bash
# Baseline only
python experiments/baseline.py --device cuda --num-samples 10

# Sketch experiments only
python experiments/sketch_experiments.py --device cuda --num-samples 10

# Attention analysis
python experiments/attention_analysis.py --device cuda --num-samples 5
```

5. Analyze results:
```bash
jupyter notebook notebooks/analysis.ipynb
```

**Implementation Notes:**

- The Count-Min Sketch uses universal hashing with a large prime (2^31 - 1) for better distribution
- Sketch parameters are configurable: width ∈ {128, 256, 512, 1024}, depth ∈ {2, 4, 6}
- Two strategies implemented: top-k (k ∈ {32, 64, 128}) and aggregate (with 64-token recent window)
- All experiments use FP16 precision for efficiency on consumer GPUs
- Memory tracking uses PyTorch CUDA memory statistics for accurate GPU measurement
- The implementation is modular and can be easily extended to other transformer models

**Current Status:**

The implementation is complete and ready for experimental evaluation. All core components have been implemented and tested. The next step is to run the full experimental suite on hardware with GPU access to collect results for the final report. The framework supports both quick testing (3 samples) and comprehensive evaluation (10+ samples per configuration).

**Expected Results:**

Based on the implementation, we expect to demonstrate:
1. Memory compression ratios between 20-40% of full cache depending on sketch parameters
2. Comparison of top-k vs aggregate strategies in terms of memory-quality trade-offs
3. Attention distribution approximation error quantified through KL divergence
4. Latency characteristics across different sequence lengths
5. Validation or refinement of our hypothesis about 20-30% memory with 1-2% quality degradation


Project Structure

```
.
├── src/
│   ├── sketches/
│   │   ├── __init__.py
│   │   └── count_min_sketch.py       # Count-Min Sketch implementation
│   ├── models/
│   │   ├── __init__.py
│   │   └── sketch_gpt2.py            # Modified GPT-2 with sketch cache
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py                # Memory, latency, quality metrics
├── experiments/
│   ├── baseline.py                    # Full KV-cache baseline
│   ├── sketch_experiments.py         # Sketch compression experiments
│   ├── baselines_comparison.py       # Sliding window & quantization
│   └── attention_analysis.py         # Attention distribution analysis
├── notebooks/
│   └── analysis.ipynb                # Visualization and analysis
├── requirements.txt                   # Python dependencies
├── README.md                         # Setup and usage instructions
├── PROJECT.md                        # This document
└── run_all_experiments.py            # Master experiment runner
```

**Key Design Decisions:**

1. **Modular Architecture**: Each component (sketch, model, metrics) is independently testable
2. **GPU Efficiency**: All operations optimized for GPU execution with batching support
3. **Flexible Configuration**: Sketch parameters easily adjustable for different trade-offs
4. **Comprehensive Metrics**: Memory, latency, and quality tracked at every step
5. **Reproducibility**: All experiments use fixed seeds and log hyperparameters

References
Taming the Fragility of KV Cache Eviction in LLM Inference
https://arxiv.org/html/2510.13334v1
KV Cache Eviction in Transformer LLMs
https://www.emergentmind.com/topics/kv-cache-eviction
