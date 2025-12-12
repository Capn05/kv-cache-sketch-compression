"""
Master script to run all experiments in sequence.

This script orchestrates the complete experimental pipeline:
1. Baseline experiments (full KV-cache)
2. Sketch compression experiments
3. Baseline comparisons (sliding window, quantization)
4. Attention distribution analysis
5. Results visualization

Usage:
    python run_all_experiments.py [--quick] [--device cuda]
"""

import sys
import os
import argparse
from pathlib import Path


def run_baseline(device='cuda', num_samples=10, seq_lengths=None):
    """Run baseline experiments."""
    print("\n" + "="*60)
    print("STEP 1: BASELINE EXPERIMENTS")
    print("="*60)
    
    if seq_lengths is None:
        seq_lengths = [128, 256, 512]
    
    from experiments.baseline import evaluate_baseline
    
    try:
        evaluate_baseline(
            model_name='gpt2',
            device=device,
            sequence_lengths=seq_lengths,
            num_samples=num_samples,
            output_dir='results/baseline'
        )
        print("\n✓ Baseline experiments completed")
        return True
    except Exception as e:
        print(f"\n✗ Baseline experiments failed: {e}")
        return False


def run_sketch_experiments(device='cuda', num_samples=5, max_tokens=128):
    """Run sketch compression experiments."""
    print("\n" + "="*60)
    print("STEP 2: SKETCH COMPRESSION EXPERIMENTS")
    print("="*60)
    
    from experiments.sketch_experiments import run_experiment_grid
    
    try:
        run_experiment_grid(
            model_name='gpt2',
            device=device,
            output_dir='results/sketch',
            num_samples=num_samples,
            max_new_tokens=max_tokens
        )
        print("\n✓ Sketch experiments completed")
        return True
    except Exception as e:
        print(f"\n✗ Sketch experiments failed: {e}")
        return False


def run_baseline_comparisons(device='cuda', num_samples=10):
    """Run baseline comparison experiments."""
    print("\n" + "="*60)
    print("STEP 3: BASELINE COMPARISONS")
    print("="*60)
    
    from experiments.baselines_comparison import compare_all_baselines
    
    try:
        compare_all_baselines(
            model_name='gpt2',
            device=device,
            num_samples=num_samples,
            output_dir='results/baselines'
        )
        print("\n✓ Baseline comparisons completed")
        return True
    except Exception as e:
        print(f"\n✗ Baseline comparisons failed: {e}")
        return False


def run_attention_analysis(device='cuda', num_samples=5):
    """Run attention distribution analysis."""
    print("\n" + "="*60)
    print("STEP 4: ATTENTION DISTRIBUTION ANALYSIS")
    print("="*60)
    
    from experiments.attention_analysis import run_attention_analysis
    
    try:
        run_attention_analysis(
            model_name='gpt2',
            device=device,
            num_samples=num_samples,
            output_dir='results/attention_analysis'
        )
        print("\n✓ Attention analysis completed")
        return True
    except Exception as e:
        print(f"\n✗ Attention analysis failed: {e}")
        return False


def main():
    """Run complete experimental pipeline."""
    parser = argparse.ArgumentParser(
        description='Run all KV-cache sketch compression experiments'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run experiments on'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick mode with fewer samples'
    )
    parser.add_argument(
        '--skip-baseline',
        action='store_true',
        help='Skip baseline experiments'
    )
    parser.add_argument(
        '--skip-sketch',
        action='store_true',
        help='Skip sketch experiments'
    )
    parser.add_argument(
        '--skip-comparisons',
        action='store_true',
        help='Skip baseline comparisons'
    )
    parser.add_argument(
        '--skip-attention',
        action='store_true',
        help='Skip attention analysis'
    )
    
    args = parser.parse_args()
    
    # Adjust parameters based on mode
    if args.quick:
        num_samples = 3
        seq_lengths = [256]
        max_tokens = 64
        print("\n*** QUICK MODE: Running with reduced samples ***\n")
    else:
        num_samples = 10
        seq_lengths = [128, 256, 512]
        max_tokens = 128
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    print("="*60)
    print("KV-CACHE SKETCH COMPRESSION - FULL EXPERIMENTAL PIPELINE")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Samples: {num_samples}")
    
    # Track results
    results = {
        'baseline': False,
        'sketch': False,
        'comparisons': False,
        'attention': False
    }
    
    # Run experiments
    if not args.skip_baseline:
        results['baseline'] = run_baseline(
            device=args.device,
            num_samples=num_samples,
            seq_lengths=seq_lengths
        )
    else:
        print("\nSkipping baseline experiments")
    
    if not args.skip_sketch:
        results['sketch'] = run_sketch_experiments(
            device=args.device,
            num_samples=num_samples,
            max_tokens=max_tokens
        )
    else:
        print("\nSkipping sketch experiments")
    
    if not args.skip_comparisons:
        results['comparisons'] = run_baseline_comparisons(
            device=args.device,
            num_samples=num_samples
        )
    else:
        print("\nSkipping baseline comparisons")
    
    if not args.skip_attention:
        results['attention'] = run_attention_analysis(
            device=args.device,
            num_samples=max(3, num_samples // 2)
        )
    else:
        print("\nSkipping attention analysis")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENTAL PIPELINE COMPLETE")
    print("="*60)
    
    for experiment, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED/SKIPPED"
        print(f"{experiment.upper()}: {status}")
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Run the analysis notebook:")
    print("   jupyter notebook notebooks/analysis.ipynb")
    print("\n2. Results are saved in the 'results/' directory")
    print("\n3. Update PROJECT.md with findings")
    print("="*60)
    
    # Return success if at least some experiments completed
    return any(results.values())


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

