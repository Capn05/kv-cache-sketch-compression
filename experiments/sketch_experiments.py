"""
Sketch-based KV-cache compression experiments.

This script evaluates Count-Min Sketch compression across different
configurations and strategies, and supports multiple sketch types.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import json
import time
import itertools

from src.models.sketch_gpt2 import SketchGPT2LMHeadModel
from src.evaluation.metrics import MemoryTracker, LatencyTracker, QualityMetrics, ExperimentLogger


def _entropy_from_logits(logits: torch.Tensor) -> float:
    """
    Args:
        logits: (vocab,) or (1, vocab)
    """
    if logits.ndim == 2:
        logits = logits[0]
    logp = F.log_softmax(logits, dim=-1)
    p = torch.exp(logp)
    return float(-(p * logp).sum().item())


def _repetition_rate_token_window(token_ids: torch.Tensor, window: int = 128) -> float:
    """
    Simple repetition heuristic: fraction of tokens in the last `window` that
    have occurred before within that window.
    """
    ids = token_ids.view(-1).tolist()
    if not ids:
        return 0.0
    tail = ids[-int(window) :] if len(ids) > window else ids
    seen = set()
    repeats = 0
    for t in tail:
        if t in seen:
            repeats += 1
        else:
            seen.add(t)
    return float(repeats / max(1, len(tail)))


def run_sketch_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
    memory_tracker: MemoryTracker,
    latency_tracker: LatencyTracker
):
    """
    Run generation with sketch-based cache.
    
    Returns generated text and metrics.
    """
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)
    
    # Reset trackers
    memory_tracker.reset()
    memory_tracker.record('start')
    
    # Generate
    model.eval()
    with torch.no_grad():
        latency_tracker.start()
        
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
            use_cache=True
        )
        
        latency_tracker.stop()
    
    # Record memory
    memory_tracker.record('end')
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def run_sketch_generation_manual(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str,
    memory_tracker: MemoryTracker,
    latency_tracker: LatencyTracker,
    adaptive: bool = False,
):
    """
    Manual token-by-token generation loop (so we can drive adaptive control).
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)

    memory_tracker.reset()
    memory_tracker.record("start")

    model.eval()
    generated_ids = input_ids.clone()
    past_key_values = None
    adaptive_log = []

    with torch.no_grad():
        for step in range(int(max_new_tokens)):
            t0 = time.time()
            if past_key_values is None:
                outputs = model(
                    generated_ids,
                    attention_mask=torch.ones_like(generated_ids),
                    use_cache=True,
                )
            else:
                outputs = model(
                    generated_ids[:, -1:],
                    attention_mask=torch.ones_like(generated_ids[:, -1:]),
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

            entropy = _entropy_from_logits(next_logits)
            repetition = _repetition_rate_token_window(generated_ids, window=128)
            if adaptive:
                new_size = model.adaptive_maybe_update(
                    step_idx=step, entropy=entropy, repetition=repetition
                )
            else:
                new_size = None
            adaptive_log.append(
                {
                    "step": int(step),
                    "entropy": float(entropy),
                    "repetition": float(repetition),
                    "new_max_cache_size": int(new_size) if new_size is not None else None,
                }
            )

            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            elapsed = time.time() - t0
            latency_tracker.record_token_time(elapsed)

            if next_token.item() == tokenizer.eos_token_id:
                break

    memory_tracker.record("end")
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text, adaptive_log


def run_sketch_generation_to_total_length(
    model,
    tokenizer,
    prompt: str,
    total_length: int,
    device: str,
    memory_tracker: MemoryTracker,
    latency_tracker: LatencyTracker,
):
    """
    Generate until reaching a target total sequence length (prompt + new tokens).
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    max_new_tokens = max(0, int(total_length) - int(prompt_len))
    return run_sketch_generation(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
        memory_tracker=memory_tracker,
        latency_tracker=latency_tracker,
    )


def run_sketch_generation_to_total_length_manual(
    model,
    tokenizer,
    prompt: str,
    total_length: int,
    device: str,
    memory_tracker: MemoryTracker,
    latency_tracker: LatencyTracker,
    adaptive: bool = False,
):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    max_new_tokens = max(0, int(total_length) - int(prompt_len))
    return run_sketch_generation_manual(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
        memory_tracker=memory_tracker,
        latency_tracker=latency_tracker,
        adaptive=adaptive,
    )


def evaluate_sketch_config(
    model_name: str,
    sketch_config: dict,
    device: str,
    test_texts: list,
    num_samples: int,
    max_new_tokens: int = 128,
    total_length: int = None,
    reference_texts: list = None,
    compute_ppl: bool = False,
    ppl_num_texts: int = 1,
    ppl_max_length: int = 256,
):
    """
    Evaluate a specific sketch configuration.
    
    Args:
        model_name: Model identifier
        sketch_config: Configuration dict with sketch parameters
        device: Device to run on
        test_texts: List of prompt texts
        num_samples: Number of samples to test
        max_new_tokens: Tokens to generate
        reference_texts: Optional reference texts for BLEU
        
    Returns:
        Dictionary of results
    """
    # Load model
    model = SketchGPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    if device == 'cuda' and torch.cuda.is_available():
        model = model.to(device)
        model = model.half()
    else:
        device = 'cpu'
        model = model.to(device)
    
    # Enable sketch cache with compression
    # Defaults for backward compatibility / missing fields
    if "sketch_type" not in sketch_config:
        sketch_config["sketch_type"] = "cms"
    if 'max_cache_size' not in sketch_config:
        sketch_config['max_cache_size'] = 128  # Default: compress to 128 tokens
    if "race_bins" not in sketch_config:
        sketch_config["race_bins"] = 512
    if "race_num_projections" not in sketch_config:
        sketch_config["race_num_projections"] = 16
    if "race_seed" not in sketch_config:
        sketch_config["race_seed"] = 42
    
    model.enable_sketch_cache(**sketch_config)
    if sketch_config.get("adaptive", False):
        model.enable_adaptive_cache()
    
    # Results storage
    results = {
        'config': sketch_config,
        'memory_mb': [],
        'latency_s': [],
        'tokens_per_sec': [],
        'generated_texts': [],
        'bleu_scores': [],
        'adaptive_log': [],
    }
    
    # Run samples
    for i in range(min(num_samples, len(test_texts))):
        prompt = test_texts[i]
        
        # Initialize trackers
        memory_tracker = MemoryTracker(device=device)
        latency_tracker = LatencyTracker()
        
        try:
            # Generate
            if total_length is not None:
                if sketch_config.get("adaptive", False):
                    generated_text, adapt_log = run_sketch_generation_to_total_length_manual(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        total_length=total_length,
                        device=device,
                        memory_tracker=memory_tracker,
                        latency_tracker=latency_tracker,
                        adaptive=True,
                    )
                    results["adaptive_log"].append(adapt_log)
                else:
                    generated_text = run_sketch_generation_to_total_length(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        total_length=total_length,
                        device=device,
                        memory_tracker=memory_tracker,
                        latency_tracker=latency_tracker,
                    )
                effective_new_tokens = max(
                    0,
                    int(total_length)
                    - int(tokenizer.encode(prompt, return_tensors="pt").shape[1]),
                )
            else:
                if sketch_config.get("adaptive", False):
                    generated_text, adapt_log = run_sketch_generation_manual(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        device=device,
                        memory_tracker=memory_tracker,
                        latency_tracker=latency_tracker,
                        adaptive=True,
                    )
                    results["adaptive_log"].append(adapt_log)
                else:
                    generated_text = run_sketch_generation(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        device=device,
                        memory_tracker=memory_tracker,
                        latency_tracker=latency_tracker,
                    )
                effective_new_tokens = int(max_new_tokens)
            
            # Collect metrics
            peak_memory = memory_tracker.get_peak_memory()
            total_time = latency_tracker.total_time
            tokens_per_sec = effective_new_tokens / total_time if total_time > 0 else 0
            
            results['memory_mb'].append(peak_memory)
            results['latency_s'].append(total_time)
            results['tokens_per_sec'].append(tokens_per_sec)
            results['generated_texts'].append(generated_text[:200])
            
            # Compute BLEU if reference provided
            if reference_texts and i < len(reference_texts):
                bleu = QualityMetrics.compute_bleu(reference_texts[i], generated_text)
                results['bleu_scores'].append(bleu)
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"OOM with config: {sketch_config}")
                break
            else:
                raise e
    
    # Get detailed sketch memory breakdown
    sketch_mem_info = model.get_sketch_memory_usage()
    results['sketch_memory_mb'] = sketch_mem_info['sketch_mb']
    results['cache_memory_mb'] = sketch_mem_info['cache_mb']
    results['total_sketch_cache_mb'] = sketch_mem_info['total_mb']
    results['avg_tokens_per_layer'] = sketch_mem_info['avg_tokens_per_layer']

    # Autoregressive perplexity (optional; computed once per config for speed)
    if compute_ppl:
        try:
            ppl_texts = test_texts[: max(1, int(ppl_num_texts))]
            results["perplexity_ar"] = QualityMetrics.compute_perplexity_autoregressive(
                model=model,
                tokenizer=tokenizer,
                texts=ppl_texts,
                device=device,
                max_length=int(ppl_max_length),
            )
        except Exception as e:
            results["perplexity_ar"] = None
            results["perplexity_ar_error"] = str(e)
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return results


def run_experiment_grid(
    model_name: str = 'gpt2',
    device: str = 'cuda',
    output_dir: str = 'results/sketch',
    num_samples: int = 10,
    max_new_tokens: int = 128,
    total_lengths: list = None,
    grid: str = "full",
):
    """
    Run experiments across a grid of sketch configurations.
    
    Args:
        model_name: Model name
        device: Device to run on
        output_dir: Output directory
        num_samples: Samples per config
        max_new_tokens: Tokens to generate
    """
    print("="*60)
    print("SKETCH COMPRESSION EXPERIMENTS")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    print("\nLoading WikiText-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 100]
    
    # Prepare prompts (first 50 tokens)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    test_prompts = []
    for text in texts[:num_samples]:
        tokens = tokenizer.encode(text)[:50]
        prompt = tokenizer.decode(tokens)
        test_prompts.append(prompt)
    
    def build_configs(grid_name: str):
        if grid_name == "targeted":
            # Targeted (time-boxed): sketch_type × max_cache_size with fixed sketch params
            sketch_types = ["cms", "count_sketch", "race"]
            sketch_widths = [512]
            sketch_depths = [4]
            max_cache_sizes = [64, 128, 256]
            configs = []
            for sketch_type, width, depth, cache_size in itertools.product(
                sketch_types, sketch_widths, sketch_depths, max_cache_sizes
            ):
                recent_window = 64
                # For the top-k strategy, pick topk so (recent_window + topk) ~= max_cache_size.
                topk = max(0, int(cache_size) - int(recent_window))
                cfg = {
                    "sketch_type": sketch_type,
                    "sketch_width": width,
                    "sketch_depth": depth,
                    "strategy": "topk",  # accepted for compatibility
                    "topk": topk,
                    "recent_window": recent_window,
                    "max_cache_size": cache_size,
                }
                if sketch_type == "race":
                    cfg.update({"race_bins": 512, "race_num_projections": 16, "race_seed": 42})
                configs.append(cfg)
            return configs

        # Default: original full grid
        sketch_widths = [128, 256, 512, 1024]
        sketch_depths = [2, 4, 6]
        strategies = ["topk", "aggregate"]
        topk_values = [32, 64, 128]
        max_cache_sizes = [64, 128, 256]

        configs = []
        for width, depth, strategy, cache_size in itertools.product(
            sketch_widths, sketch_depths, strategies, max_cache_sizes
        ):
            if strategy == "topk":
                for topk in topk_values:
                    configs.append(
                        {
                            "sketch_width": width,
                            "sketch_depth": depth,
                            "strategy": strategy,
                            "topk": topk,
                            "recent_window": 64,
                            "max_cache_size": cache_size,
                        }
                    )
            else:
                configs.append(
                    {
                        "sketch_width": width,
                        "sketch_depth": depth,
                        "strategy": strategy,
                        "topk": 64,
                        "recent_window": 64,
                        "max_cache_size": cache_size,
                    }
                )
        return configs
    
    configs = build_configs(grid)
    
    print(f"\nTotal configurations to test: {len(configs)}")
    
    # Run experiments
    all_results = []
    targeted_summary = {
        "model_name": model_name,
        "device": device,
        "grid": grid,
        "num_samples": num_samples,
        "max_new_tokens": max_new_tokens,
        "total_lengths": total_lengths,
        "results": [],
    }
    
    for config_idx, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Configuration {config_idx+1}/{len(configs)}")
        print(f"{'='*60}")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        try:
            if total_lengths:
                for total_len in total_lengths:
                    results = evaluate_sketch_config(
                        model_name=model_name,
                        sketch_config=config,
                        device=device,
                        test_texts=test_prompts,
                        num_samples=num_samples,
                        max_new_tokens=max_new_tokens,
                        total_length=int(total_len),
                        compute_ppl=(bool(total_lengths) and int(total_len) == int(total_lengths[0])),
                        ppl_num_texts=1,
                        ppl_max_length=min(512, int(total_len)),
                    )
                    results["total_length"] = int(total_len)

                    # Compute averages
                    if results["memory_mb"]:
                        avg_memory = sum(results["memory_mb"]) / len(results["memory_mb"])
                        avg_latency = sum(results["latency_s"]) / len(results["latency_s"])
                        avg_throughput = (
                            sum(results["tokens_per_sec"]) / len(results["tokens_per_sec"])
                        )

                        print(f"\nResults (total_length={total_len}):")
                        print(f"  Total GPU memory: {avg_memory:.2f} MB")
                        print(
                            f"  Cache memory: {results['cache_memory_mb']:.2f} MB (max_size={config['max_cache_size']})"
                        )
                        print(
                            f"  Avg tokens stored/layer: {results['avg_tokens_per_layer']:.1f}"
                        )
                        print(f"  Sketch overhead: {results['sketch_memory_mb']:.2f} MB")
                        print(f"  Avg latency: {avg_latency:.2f} s")
                        print(f"  Avg throughput: {avg_throughput:.2f} tokens/s")

                        all_results.append(
                            {
                                "total_length": int(total_len),
                                "config": config,
                                "avg_memory_mb": avg_memory,
                                "sketch_memory_mb": results["sketch_memory_mb"],
                                "cache_memory_mb": results["cache_memory_mb"],
                                "avg_tokens_per_layer": results["avg_tokens_per_layer"],
                                "avg_latency_s": avg_latency,
                                "avg_throughput": avg_throughput,
                            }
                        )
                        targeted_summary["results"].append(
                            {
                                "total_length": int(total_len),
                                "config": config,
                                "avg_memory_mb": avg_memory,
                                "sketch_memory_mb": results["sketch_memory_mb"],
                                "cache_memory_mb": results["cache_memory_mb"],
                                "avg_tokens_per_layer": results["avg_tokens_per_layer"],
                                "avg_latency_s": avg_latency,
                                "avg_throughput": avg_throughput,
                            }
                        )

                        # Save per-config-per-length results
                        config_filename = (
                            f"config_{config.get('sketch_type','cms')}_"
                            f"w{config['sketch_width']}_d{config['sketch_depth']}_{config['strategy']}"
                        )
                        if config["strategy"] == "topk":
                            config_filename += f"_k{config['topk']}"
                        config_filename += f"_len{int(total_len)}.json"

                        with open(os.path.join(output_dir, config_filename), "w") as f:
                            json.dump(results, f, indent=2)
            else:
                results = evaluate_sketch_config(
                    model_name=model_name,
                    sketch_config=config,
                    device=device,
                    test_texts=test_prompts,
                    num_samples=num_samples,
                    max_new_tokens=max_new_tokens,
                )
            
                # Compute averages
                if results["memory_mb"]:
                    avg_memory = sum(results["memory_mb"]) / len(results["memory_mb"])
                    avg_latency = sum(results["latency_s"]) / len(results["latency_s"])
                    avg_throughput = sum(results["tokens_per_sec"]) / len(
                        results["tokens_per_sec"]
                    )

                    print(f"\nResults:")
                    print(f"  Total GPU memory: {avg_memory:.2f} MB")
                    print(
                        f"  Cache memory: {results['cache_memory_mb']:.2f} MB (max_size={config['max_cache_size']})"
                    )
                    print(
                        f"  Avg tokens stored/layer: {results['avg_tokens_per_layer']:.1f}"
                    )
                    print(f"  Sketch overhead: {results['sketch_memory_mb']:.2f} MB")
                    print(f"  Avg latency: {avg_latency:.2f} s")
                    print(f"  Avg throughput: {avg_throughput:.2f} tokens/s")

                    if results["bleu_scores"]:
                        avg_bleu = sum(results["bleu_scores"]) / len(results["bleu_scores"])
                        print(f"  Avg BLEU: {avg_bleu:.4f}")

                    all_results.append(
                        {
                            "config": config,
                            "avg_memory_mb": avg_memory,
                            "sketch_memory_mb": results["sketch_memory_mb"],
                            "avg_latency_s": avg_latency,
                            "avg_throughput": avg_throughput,
                            "full_results": results,
                        }
                    )

                    # Save individual config results
                    config_filename = (
                        f"config_{config.get('sketch_type','cms')}_"
                        f"w{config['sketch_width']}_d{config['sketch_depth']}_{config['strategy']}"
                    )
                    if config["strategy"] == "topk":
                        config_filename += f"_k{config['topk']}"
                    config_filename += ".json"

                    with open(os.path.join(output_dir, config_filename), "w") as f:
                        json.dump(results, f, indent=2)
            
        except Exception as e:
            print(f"Error with config: {e}")
            continue
    
    # Save summary
    print("\n" + "="*60)
    print("Saving aggregate results...")
    print("="*60)
    
    summary = {
        'model_name': model_name,
        'num_configs': len(all_results),
        'results': all_results
    }
    
    with open(os.path.join(output_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save targeted summary if applicable
    if grid == "targeted" or total_lengths:
        with open(os.path.join(output_dir, "targeted_summary.json"), "w") as f:
            json.dump(targeted_summary, f, indent=2)
    
    print(f"\nExperiments complete! Results saved to {output_dir}/")
    
    # Print best configurations
    if all_results:
        print("\n" + "="*60)
        print("TOP CONFIGURATIONS")
        print("="*60)
        
        # Best memory
        best_memory = min(all_results, key=lambda x: x['avg_memory_mb'])
        print(f"\nBest Memory:")
        print(f"  Config: {best_memory['config']}")
        print(f"  Memory: {best_memory['avg_memory_mb']:.2f} MB")
        
        # Best throughput
        best_throughput = max(all_results, key=lambda x: x['avg_throughput'])
        print(f"\nBest Throughput:")
        print(f"  Config: {best_throughput['config']}")
        print(f"  Throughput: {best_throughput['avg_throughput']:.2f} tokens/s")
    
    return all_results


def main():
    """Run sketch experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run sketch compression experiments')
    parser.add_argument('--model', type=str, default='gpt2', help='Model name')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output-dir', type=str, default='results/sketch',
                       help='Output directory')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples per config')
    parser.add_argument('--max-tokens', type=int, default=128,
                       help='Max tokens to generate')
    parser.add_argument(
        "--total-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Target total sequence lengths (prompt + generated). If provided, overrides --max-tokens per sample.",
    )
    parser.add_argument(
        "--grid",
        type=str,
        choices=["full", "targeted"],
        default="full",
        help="Which configuration grid to run.",
    )
    
    args = parser.parse_args()
    
    results = run_experiment_grid(
        model_name=args.model,
        device=args.device,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        max_new_tokens=args.max_tokens,
        total_lengths=args.total_lengths,
        grid=args.grid,
    )


if __name__ == '__main__':
    main()

