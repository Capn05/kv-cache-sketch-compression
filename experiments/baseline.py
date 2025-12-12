"""
Baseline experiments with full KV-cache GPT-2.

This script measures memory usage, latency, and perplexity for
unmodified GPT-2 to establish baseline metrics.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import time

from src.evaluation.metrics import MemoryTracker, LatencyTracker, QualityMetrics, ExperimentLogger


def run_baseline_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
    memory_tracker: MemoryTracker,
    latency_tracker: LatencyTracker
):
    """
    Run generation with memory and latency tracking.
    
    Returns generated text and per-token times.
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Reset trackers
    memory_tracker.reset()
    memory_tracker.record('start')
    
    # Generate tokens one at a time to track per-token metrics
    generated_ids = input_ids.clone()
    token_times = []
    
    model.eval()
    with torch.no_grad():
        past_key_values = None
        
        for step in range(max_new_tokens):
            # Time this token generation
            start_time = time.time()
            
            # Forward pass
            if past_key_values is None:
                outputs = model(generated_ids, use_cache=True)
            else:
                outputs = model(generated_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            
            past_key_values = outputs.past_key_values
            
            # Get next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Record timing
            elapsed = time.time() - start_time
            token_times.append(elapsed)
            latency_tracker.record_token_time(elapsed)
            
            # Record memory after some tokens
            if step % 10 == 0:
                memory_tracker.record(f'step_{step}')
            
            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Final memory measurement
    memory_tracker.record('end')
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text, token_times


def evaluate_baseline(
    model_name: str = 'gpt2',
    device: str = 'cuda',
    sequence_lengths: list = [128, 256, 512, 1024],
    num_samples: int = 10,
    output_dir: str = 'results'
):
    """
    Run comprehensive baseline evaluation.
    
    Args:
        model_name: Hugging Face model name
        device: Device to run on
        sequence_lengths: List of sequence lengths to test
        num_samples: Number of samples per length
        output_dir: Directory to save results
    """
    print(f"Running baseline evaluation for {model_name}")
    print(f"Device: {device}")
    print(f"Sequence lengths: {sequence_lengths}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    if device == 'cuda' and torch.cuda.is_available():
        model = model.to(device)
        model = model.half()  # Use FP16 for efficiency
    else:
        device = 'cpu'
        model = model.to(device)
    
    print(f"Model loaded on {device}")
    
    # Load WikiText-2 dataset
    print("\nLoading WikiText-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100]
    
    # Initialize experiment logger
    logger = ExperimentLogger()
    logger.set_metadata('model_name', model_name)
    logger.set_metadata('device', device)
    logger.set_metadata('cache_type', 'full')
    
    # Test each sequence length
    for seq_len in sequence_lengths:
        print(f"\n{'='*60}")
        print(f"Testing sequence length: {seq_len}")
        print(f"{'='*60}")
        
        seq_results = {
            'sequence_length': seq_len,
            'memory_mb': [],
            'latency_ms': [],
            'tokens_per_sec': [],
            'generated_texts': []
        }
        
        # Test on multiple samples
        for sample_idx in tqdm(range(min(num_samples, len(texts))), desc="Samples"):
            # Get prompt (limit to reasonable length)
            prompt_text = texts[sample_idx]
            prompt_tokens = tokenizer.encode(prompt_text)[:50]  # Limit prompt
            prompt = tokenizer.decode(prompt_tokens)
            
            # Calculate tokens to generate
            max_new_tokens = seq_len - len(prompt_tokens)
            if max_new_tokens <= 0:
                continue
            
            # Initialize trackers
            memory_tracker = MemoryTracker(device=device)
            latency_tracker = LatencyTracker()
            
            try:
                # Run generation
                generated_text, token_times = run_baseline_generation(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    device=device,
                    memory_tracker=memory_tracker,
                    latency_tracker=latency_tracker
                )
                
                # Collect metrics
                peak_memory = memory_tracker.get_peak_memory()
                avg_latency = latency_tracker.get_avg_time_per_token() * 1000  # Convert to ms
                tokens_per_sec = latency_tracker.get_tokens_per_second()
                
                seq_results['memory_mb'].append(peak_memory)
                seq_results['latency_ms'].append(avg_latency)
                seq_results['tokens_per_sec'].append(tokens_per_sec)
                seq_results['generated_texts'].append(generated_text[:200])  # Store snippet
                
                # Log individual run
                logger.log(f'memory_mb_seq{seq_len}', peak_memory)
                logger.log(f'latency_ms_seq{seq_len}', avg_latency)
                logger.log(f'tokens_per_sec_seq{seq_len}', tokens_per_sec)
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"\nOut of memory at sequence length {seq_len}")
                    break
                else:
                    raise e
        
        # Compute aggregates for this sequence length
        if seq_results['memory_mb']:
            print(f"\nResults for sequence length {seq_len}:")
            print(f"  Avg memory: {sum(seq_results['memory_mb'])/len(seq_results['memory_mb']):.2f} MB")
            print(f"  Avg latency: {sum(seq_results['latency_ms'])/len(seq_results['latency_ms']):.2f} ms/token")
            print(f"  Avg throughput: {sum(seq_results['tokens_per_sec'])/len(seq_results['tokens_per_sec']):.2f} tokens/sec")
            
            # Save sequence-specific results
            with open(f'{output_dir}/baseline_seq{seq_len}.json', 'w') as f:
                json.dump(seq_results, f, indent=2)
    
    # Compute perplexity on subset of texts
    print("\n" + "="*60)
    print("Computing perplexity...")
    print("="*60)
    
    try:
        perplexity_texts = texts[:20]  # Use subset for speed
        perplexity = QualityMetrics.compute_perplexity(
            model=model,
            tokenizer=tokenizer,
            texts=perplexity_texts,
            device=device,
            max_length=512
        )
        print(f"Perplexity: {perplexity:.2f}")
        logger.set_metadata('perplexity', perplexity)
    except Exception as e:
        print(f"Could not compute perplexity: {e}")
    
    # Save overall results
    print("\nSaving results...")
    logger.save(f'{output_dir}/baseline_summary.json')
    
    print(f"\nBaseline evaluation complete!")
    print(f"Results saved to {output_dir}/")
    
    return logger.get_summary()


def main():
    """Run baseline experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run baseline GPT-2 experiments')
    parser.add_argument('--model', type=str, default='gpt2', help='Model name')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seq-lengths', type=int, nargs='+', default=[128, 256, 512],
                       help='Sequence lengths to test')
    parser.add_argument('--num-samples', type=int, default=10, help='Samples per length')
    parser.add_argument('--output-dir', type=str, default='results/baseline',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_baseline(
        model_name=args.model,
        device=args.device,
        sequence_lengths=args.seq_lengths,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.2f}")


if __name__ == '__main__':
    main()

