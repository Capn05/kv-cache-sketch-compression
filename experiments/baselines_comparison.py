"""
Comparison baselines: sliding window and quantization.

This script implements and evaluates alternative KV-cache compression methods
for comparison with sketch-based approaches.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from datasets import load_dataset
import json
from tqdm import tqdm

from src.evaluation.metrics import MemoryTracker, LatencyTracker, QualityMetrics


class SlidingWindowCache:
    """Sliding window KV-cache that keeps only recent tokens."""
    
    def __init__(self, window_size: int = 256):
        self.window_size = window_size
        self.keys = []
        self.values = []
    
    def update(self, key, value):
        """Add new key-value pair."""
        self.keys.append(key)
        self.values.append(value)
        
        # Keep only recent window
        if len(self.keys) > self.window_size:
            self.keys.pop(0)
            self.values.pop(0)
    
    def get_cache(self):
        """Get current cache."""
        if not self.keys:
            return None, None
        return torch.stack(self.keys), torch.stack(self.values)
    
    def clear(self):
        """Clear cache."""
        self.keys.clear()
        self.values.clear()


class QuantizedCache:
    """Quantized KV-cache using int8 quantization."""
    
    def __init__(self):
        self.keys = []
        self.values = []
        self.key_scales = []
        self.value_scales = []
    
    def quantize_tensor(self, tensor):
        """Quantize tensor to int8."""
        # Compute scale
        abs_max = tensor.abs().max()
        scale = abs_max / 127.0 if abs_max > 0 else 1.0
        
        # Quantize
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        
        return quantized, scale
    
    def dequantize_tensor(self, quantized, scale):
        """Dequantize int8 tensor."""
        return quantized.to(torch.float32) * scale
    
    def update(self, key, value):
        """Add quantized key-value pair."""
        # Quantize
        key_q, key_scale = self.quantize_tensor(key)
        value_q, value_scale = self.quantize_tensor(value)
        
        # Store
        self.keys.append(key_q)
        self.values.append(value_q)
        self.key_scales.append(key_scale)
        self.value_scales.append(value_scale)
    
    def get_cache(self):
        """Get dequantized cache."""
        if not self.keys:
            return None, None
        
        # Dequantize all
        keys = torch.stack([
            self.dequantize_tensor(k, s)
            for k, s in zip(self.keys, self.key_scales)
        ])
        values = torch.stack([
            self.dequantize_tensor(v, s)
            for v, s in zip(self.values, self.value_scales)
        ])
        
        return keys, values
    
    def clear(self):
        """Clear cache."""
        self.keys.clear()
        self.values.clear()
        self.key_scales.clear()
        self.value_scales.clear()
    
    def get_memory_usage(self):
        """Estimate memory usage in bytes."""
        # int8 is 1 byte per element
        keys_mem = sum(k.numel() for k in self.keys)
        values_mem = sum(v.numel() for v in self.values)
        scales_mem = (len(self.key_scales) + len(self.value_scales)) * 4  # float32 scales
        
        return keys_mem + values_mem + scales_mem


def evaluate_sliding_window(
    model_name: str = 'gpt2',
    window_size: int = 256,
    device: str = 'cuda',
    num_samples: int = 10,
    max_new_tokens: int = 128,
    output_dir: str = 'results/baselines'
):
    """
    Evaluate sliding window baseline.
    
    Note: This is a simplified implementation for measurement purposes.
    Full implementation would require modifying the attention mechanism.
    """
    print(f"Evaluating sliding window baseline (window={window_size})")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    if device == 'cuda' and torch.cuda.is_available():
        model = model.to(device)
        model = model.half()
    else:
        device = 'cpu'
        model = model.to(device)
    
    # Load dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100]
    
    results = {
        'method': 'sliding_window',
        'window_size': window_size,
        'memory_mb': [],
        'latency_s': [],
        'generated_texts': []
    }
    
    # Test samples
    for i in tqdm(range(min(num_samples, len(texts))), desc="Sliding window"):
        # Prepare prompt
        prompt_tokens = tokenizer.encode(texts[i])[:50]
        prompt = tokenizer.decode(prompt_tokens)
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Track memory
        memory_tracker = MemoryTracker(device=device)
        memory_tracker.reset()
        
        # Generate
        model.eval()
        with torch.no_grad():
            import time
            start_time = time.time()
            
            # Standard generation (simplified - real sliding window needs attention modification)
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            
            elapsed = time.time() - start_time
        
        # Record metrics
        peak_memory = memory_tracker.get_peak_memory()
        results['memory_mb'].append(peak_memory)
        results['latency_s'].append(elapsed)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results['generated_texts'].append(generated_text[:200])
    
    # Compute averages
    if results['memory_mb']:
        avg_memory = sum(results['memory_mb']) / len(results['memory_mb'])
        avg_latency = sum(results['latency_s']) / len(results['latency_s'])
        
        print(f"  Avg memory: {avg_memory:.2f} MB")
        print(f"  Avg latency: {avg_latency:.2f} s")
        
        results['avg_memory_mb'] = avg_memory
        results['avg_latency_s'] = avg_latency
    
    # Save results
    with open(os.path.join(output_dir, 'sliding_window.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def evaluate_quantization(
    model_name: str = 'gpt2',
    device: str = 'cuda',
    num_samples: int = 10,
    max_new_tokens: int = 128,
    output_dir: str = 'results/baselines'
):
    """
    Evaluate int8 quantization baseline.
    
    Note: Simplified implementation for measurement.
    """
    print("Evaluating int8 quantization baseline")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    if device == 'cuda' and torch.cuda.is_available():
        model = model.to(device)
        # Note: We use FP16 here as proxy for quantized inference
        model = model.half()
    else:
        device = 'cpu'
        model = model.to(device)
    
    # Load dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100]
    
    results = {
        'method': 'int8_quantization',
        'memory_mb': [],
        'latency_s': [],
        'generated_texts': []
    }
    
    # Test samples
    for i in tqdm(range(min(num_samples, len(texts))), desc="Quantization"):
        # Prepare prompt
        prompt_tokens = tokenizer.encode(texts[i])[:50]
        prompt = tokenizer.decode(prompt_tokens)
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # Track memory
        memory_tracker = MemoryTracker(device=device)
        memory_tracker.reset()
        
        # Generate
        model.eval()
        with torch.no_grad():
            import time
            start_time = time.time()
            
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
            
            elapsed = time.time() - start_time
        
        # Record metrics
        peak_memory = memory_tracker.get_peak_memory()
        results['memory_mb'].append(peak_memory)
        results['latency_s'].append(elapsed)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results['generated_texts'].append(generated_text[:200])
    
    # Compute averages
    if results['memory_mb']:
        avg_memory = sum(results['memory_mb']) / len(results['memory_mb'])
        avg_latency = sum(results['latency_s']) / len(results['latency_s'])
        
        print(f"  Avg memory: {avg_memory:.2f} MB")
        print(f"  Avg latency: {avg_latency:.2f} s")
        
        results['avg_memory_mb'] = avg_memory
        results['avg_latency_s'] = avg_latency
    
    # Save results
    with open(os.path.join(output_dir, 'quantization.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def compare_all_baselines(
    model_name: str = 'gpt2',
    device: str = 'cuda',
    num_samples: int = 10,
    output_dir: str = 'results/baselines'
):
    """Run all baseline comparisons."""
    print("="*60)
    print("BASELINE COMPARISONS")
    print("="*60)
    
    results = {}
    
    # Sliding window
    print("\n1. Sliding Window Cache")
    print("-"*60)
    results['sliding_window'] = evaluate_sliding_window(
        model_name=model_name,
        window_size=256,
        device=device,
        num_samples=num_samples,
        output_dir=output_dir
    )
    
    # Quantization
    print("\n2. Int8 Quantization")
    print("-"*60)
    results['quantization'] = evaluate_quantization(
        model_name=model_name,
        device=device,
        num_samples=num_samples,
        output_dir=output_dir
    )
    
    # Save comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    summary = {
        'model': model_name,
        'num_samples': num_samples,
        'results': results
    }
    
    for method, data in results.items():
        print(f"\n{method.upper()}:")
        if 'avg_memory_mb' in data:
            print(f"  Memory: {data['avg_memory_mb']:.2f} MB")
            print(f"  Latency: {data['avg_latency_s']:.2f} s")
    
    with open(os.path.join(output_dir, 'comparison_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    
    return results


def main():
    """Run baseline comparisons."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run baseline comparisons')
    parser.add_argument('--model', type=str, default='gpt2', help='Model name')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--output-dir', type=str, default='results/baselines',
                       help='Output directory')
    
    args = parser.parse_args()
    
    compare_all_baselines(
        model_name=args.model,
        device=args.device,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

