"""
Attention distribution analysis for sketch-based compression.

This script compares attention distributions between full cache
and sketch-based eviction/capping during cached token-by-token decoding.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import json
import numpy as np
from tqdm import tqdm

from src.models.sketch_gpt2 import SketchGPT2LMHeadModel
from src.evaluation.metrics import QualityMetrics


def compare_attention_distributions(
    full_attention: torch.Tensor,
    sketch_attention: torch.Tensor,
    head_idx: int = 0
):
    """
    Compare two attention distributions.
    
    Args:
        full_attention: Attention from full cache
        sketch_attention: Attention from sketch cache
        head_idx: Which attention head to analyze
        
    Returns:
        Dictionary of comparison metrics
    """
    # Extract attention for specific head and last query token
    # Shape: (seq_len,) - attention over all keys for last query
    full_attn = full_attention[0, head_idx, -1, :]
    sketch_attn = sketch_attention[0, head_idx, -1, :]

    # Make metrics robust under fp16 / eager attention (avoid NaNs/negatives).
    full_attn = torch.nan_to_num(full_attn.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    sketch_attn = torch.nan_to_num(sketch_attn.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    
    # Ensure same length (pad if needed)
    max_len = max(full_attn.shape[0], sketch_attn.shape[0])
    if full_attn.shape[0] < max_len:
        full_attn = F.pad(full_attn, (0, max_len - full_attn.shape[0]))
    if sketch_attn.shape[0] < max_len:
        sketch_attn = F.pad(sketch_attn, (0, max_len - sketch_attn.shape[0]))
    
    # Compute metrics
    metrics = {}
    
    # KL divergence
    metrics['kl_divergence'] = QualityMetrics.compute_kl_divergence(
        full_attn, sketch_attn
    )
    
    # Cosine similarity
    metrics['cosine_similarity'] = QualityMetrics.compute_cosine_similarity(
        full_attn, sketch_attn
    )
    
    # L1 distance
    metrics['l1_distance'] = (full_attn - sketch_attn).abs().mean().item()
    
    # L2 distance
    metrics['l2_distance'] = ((full_attn - sketch_attn) ** 2).mean().sqrt().item()
    
    # Peak attention position difference
    full_peak = full_attn.argmax().item()
    sketch_peak = sketch_attn.argmax().item()
    metrics['peak_position_diff'] = abs(full_peak - sketch_peak)
    
    return metrics


def analyze_sketch_config(
    model_name: str,
    sketch_config: dict,
    test_texts: list,
    num_samples: int,
    device: str
):
    """
    Analyze attention distributions for a sketch configuration.
    
    Args:
        model_name: Model identifier
        sketch_config: Sketch configuration
        test_texts: Test prompts
        num_samples: Number of samples
        device: Device to run on
        
    Returns:
        Analysis results
    """
    # Load models
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Full cache model (force eager attention so attentions are materialized)
    try:
        full_model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
    except TypeError:
        full_model = GPT2LMHeadModel.from_pretrained(model_name)
    if device == 'cuda' and torch.cuda.is_available():
        full_model = full_model.to(device).half()
    else:
        full_model = full_model.to(device)
    
    # Sketch model (force eager attention so attentions are materialized)
    try:
        sketch_model = SketchGPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
    except TypeError:
        sketch_model = SketchGPT2LMHeadModel.from_pretrained(model_name)
    if device == 'cuda' and torch.cuda.is_available():
        sketch_model = sketch_model.to(device).half()
    else:
        sketch_model = sketch_model.to(device)
    
    # Filter out analysis-only keys before enabling cache.
    allowed_cache_keys = {
        "sketch_type",
        "sketch_width",
        "sketch_depth",
        "strategy",
        "topk",
        "recent_window",
        "max_cache_size",
        "race_bins",
        "race_num_projections",
        "race_seed",
    }
    cache_cfg = {k: v for k, v in sketch_config.items() if k in allowed_cache_keys}
    sketch_model.enable_sketch_cache(**cache_cfg)
    
    # Results storage
    results = {
        'config': sketch_config,
        'kl_divergences': [],
        'cosine_similarities': [],
        'l1_distances': [],
        'l2_distances': [],
        'peak_diffs': []
    }
    
    # Analysis params
    layer_idx = int(sketch_config.get("layer_idx", 6))
    warmup = int(sketch_config.get("warmup_tokens", 128))
    stride = int(sketch_config.get("stride", 16))

    # Analyze samples
    for i in range(min(num_samples, len(test_texts))):
        text = test_texts[i]
        
        # Tokenize (limit length for speed/memory)
        max_len = int(sketch_config.get("max_length", 256))
        input_ids = tokenizer.encode(
            text, return_tensors="pt", max_length=max_len, truncation=True
        )
        input_ids = input_ids.to(device)
        
        if input_ids.shape[1] < 10:
            continue
        
        try:
            # Token-by-token cached decoding (teacher forcing) so capping affects future steps.
            past_full = None
            past_sketch = None

            # Feed tokens sequentially; compare attentions at selected steps.
            for t in range(input_ids.shape[1] - 1):
                curr = input_ids[:, t : t + 1]

                with torch.no_grad():
                    out_full = full_model(
                        curr,
                        past_key_values=past_full,
                        use_cache=True,
                        output_attentions=True,
                    )
                    past_full = out_full.past_key_values

                    out_sketch = sketch_model(
                        curr,
                        past_key_values=past_sketch,
                        use_cache=True,
                        output_attentions=True,
                    )
                    past_sketch = out_sketch.past_key_values

                if t < warmup or (stride > 0 and (t % stride) != 0):
                    continue

                full_attn = out_full.attentions[layer_idx]  # (b, h, 1, kv)
                sketch_attn = out_sketch.attentions[layer_idx]  # (b, h, 1, kv')

                # Compare distributions across a few heads
                for head_idx in range(min(4, full_attn.shape[1])):
                    metrics = compare_attention_distributions(
                        full_attn, sketch_attn, head_idx=head_idx
                    )
                    results["kl_divergences"].append(metrics["kl_divergence"])
                    results["cosine_similarities"].append(metrics["cosine_similarity"])
                    results["l1_distances"].append(metrics["l1_distance"])
                    results["l2_distances"].append(metrics["l2_distance"])
                    results["peak_diffs"].append(metrics["peak_position_diff"])
        
        except Exception as e:
            print(f"Error analyzing sample {i}: {e}")
            continue
    
    # Compute statistics
    if results['kl_divergences']:
        results['avg_kl_divergence'] = np.mean(results['kl_divergences'])
        results['avg_cosine_similarity'] = np.mean(results['cosine_similarities'])
        results['avg_l1_distance'] = np.mean(results['l1_distances'])
        results['avg_l2_distance'] = np.mean(results['l2_distances'])
        results['avg_peak_diff'] = np.mean(results['peak_diffs'])
    
    # Cleanup
    del full_model
    del sketch_model
    torch.cuda.empty_cache()
    
    return results


def run_attention_analysis(
    model_name: str = 'gpt2',
    device: str = 'cuda',
    num_samples: int = 10,
    output_dir: str = 'results/attention_analysis'
):
    """
    Run attention analysis across different sketch configurations.
    
    Args:
        model_name: Model name
        device: Device to run on
        num_samples: Number of samples to analyze
        output_dir: Output directory
    """
    print("="*60)
    print("ATTENTION DISTRIBUTION ANALYSIS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    print("\nLoading WikiText-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100]
    texts = texts[:num_samples]
    
    # Define configurations to analyze
    # Use a small set of configs that actually change cache behavior.
    configs = [
        {
            "sketch_type": "cms",
            "sketch_width": 512,
            "sketch_depth": 4,
            "strategy": "topk",
            "topk": 64,
            "recent_window": 64,
            "max_cache_size": 64,
            "layer_idx": 6,
            "warmup_tokens": 64,
            "stride": 16,
            "max_length": 256,
        },
        {
            "sketch_type": "count_sketch",
            "sketch_width": 512,
            "sketch_depth": 4,
            "strategy": "topk",
            "topk": 64,
            "recent_window": 64,
            "max_cache_size": 64,
            "layer_idx": 6,
            "warmup_tokens": 64,
            "stride": 16,
            "max_length": 256,
        },
        {
            "sketch_type": "race",
            "race_bins": 512,
            "race_num_projections": 16,
            "race_seed": 42,
            "sketch_width": 512,  # unused by race, kept for uniformity
            "sketch_depth": 4,  # unused by race, kept for uniformity
            "strategy": "topk",
            "topk": 64,
            "recent_window": 64,
            "max_cache_size": 64,
            "layer_idx": 6,
            "warmup_tokens": 64,
            "stride": 16,
            "max_length": 256,
        },
    ]
    
    all_results = []
    
    # Analyze each configuration
    for config_idx, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Configuration {config_idx+1}/{len(configs)}")
        print(f"{'='*60}")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        try:
            results = analyze_sketch_config(
                model_name=model_name,
                sketch_config=config,
                test_texts=texts,
                num_samples=num_samples,
                device=device
            )
            
            # Print results
            if 'avg_kl_divergence' in results:
                print(f"\nResults:")
                print(f"  Avg KL divergence: {results['avg_kl_divergence']:.4f}")
                print(f"  Avg cosine similarity: {results['avg_cosine_similarity']:.4f}")
                print(f"  Avg L1 distance: {results['avg_l1_distance']:.4f}")
                print(f"  Avg L2 distance: {results['avg_l2_distance']:.4f}")
                print(f"  Avg peak position diff: {results['avg_peak_diff']:.2f}")
            
            all_results.append(results)
            
            # Save individual results
            config_name = f"config_w{config['sketch_width']}_d{config['sketch_depth']}_{config['strategy']}"
            with open(os.path.join(output_dir, f'{config_name}.json'), 'w') as f:
                # Convert numpy types for JSON
                json_results = {
                    k: (v.tolist() if isinstance(v, np.ndarray) else 
                        float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in results.items()
                }
                json.dump(json_results, f, indent=2)
        
        except Exception as e:
            print(f"Error with config: {e}")
            continue
    
    # Save summary
    summary = {
        'model_name': model_name,
        'num_samples': num_samples,
        'num_configs': len(all_results),
        'results': all_results
    }
    
    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
        # Convert numpy types
        json_summary = {
            'model_name': summary['model_name'],
            'num_samples': summary['num_samples'],
            'num_configs': summary['num_configs'],
            'results': [
                {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                 for k, v in r.items() if not isinstance(v, list)}
                for r in summary['results']
            ]
        }
        json.dump(json_summary, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")
    
    return all_results


def main():
    """Run attention analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze attention distributions')
    parser.add_argument('--model', type=str, default='gpt2', help='Model name')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--output-dir', type=str, default='results/attention_analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    run_attention_analysis(
        model_name=args.model,
        device=args.device,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

