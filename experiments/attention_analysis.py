"""
Attention distribution analysis for sketch-based compression.

This script compares attention distributions between full cache
and sketch-based approximations.
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


def extract_attention_weights(model, input_ids, layer_idx=0):
    """
    Extract attention weights from a specific layer.
    
    Args:
        model: GPT2 model
        input_ids: Input token IDs
        layer_idx: Which layer to analyze
        
    Returns:
        Attention weights tensor
    """
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_attentions=True,
            use_cache=False
        )
    
    # Get attention weights for specified layer
    # Shape: (batch, num_heads, seq_len, seq_len)
    attention_weights = outputs.attentions[layer_idx]
    
    return attention_weights


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
    
    # Full cache model
    full_model = GPT2LMHeadModel.from_pretrained(model_name)
    if device == 'cuda' and torch.cuda.is_available():
        full_model = full_model.to(device).half()
    else:
        full_model = full_model.to(device)
    
    # Sketch model
    sketch_model = SketchGPT2LMHeadModel.from_pretrained(model_name)
    if device == 'cuda' and torch.cuda.is_available():
        sketch_model = sketch_model.to(device).half()
    else:
        sketch_model = sketch_model.to(device)
    
    sketch_model.enable_sketch_cache(**sketch_config)
    
    # Results storage
    results = {
        'config': sketch_config,
        'kl_divergences': [],
        'cosine_similarities': [],
        'l1_distances': [],
        'l2_distances': [],
        'peak_diffs': []
    }
    
    # Analyze samples
    for i in range(min(num_samples, len(test_texts))):
        text = test_texts[i]
        
        # Tokenize (limit length for memory)
        input_ids = tokenizer.encode(text, return_tensors='pt', max_length=256, truncation=True)
        input_ids = input_ids.to(device)
        
        if input_ids.shape[1] < 10:
            continue
        
        try:
            # Get attention from full model
            full_attention = extract_attention_weights(full_model, input_ids, layer_idx=6)
            
            # Get attention from sketch model (Note: this is simplified)
            # In practice, we'd need to modify the model to return sketch-based attention
            sketch_attention = extract_attention_weights(sketch_model, input_ids, layer_idx=6)
            
            # Compare distributions across multiple heads
            for head_idx in range(min(4, full_attention.shape[1])):
                metrics = compare_attention_distributions(
                    full_attention,
                    sketch_attention,
                    head_idx=head_idx
                )
                
                results['kl_divergences'].append(metrics['kl_divergence'])
                results['cosine_similarities'].append(metrics['cosine_similarity'])
                results['l1_distances'].append(metrics['l1_distance'])
                results['l2_distances'].append(metrics['l2_distance'])
                results['peak_diffs'].append(metrics['peak_position_diff'])
        
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
    configs = [
        {
            'sketch_width': 256,
            'sketch_depth': 4,
            'strategy': 'topk',
            'topk': 64,
            'recent_window': 64
        },
        {
            'sketch_width': 512,
            'sketch_depth': 4,
            'strategy': 'topk',
            'topk': 128,
            'recent_window': 64
        },
        {
            'sketch_width': 256,
            'sketch_depth': 4,
            'strategy': 'aggregate',
            'topk': 64,
            'recent_window': 64
        }
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

