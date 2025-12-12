"""
Quick test script to verify implementation is working.

This script runs minimal tests to ensure all components are functional
before running full experiments.
"""

import torch
import sys
import os

def test_count_min_sketch():
    """Test Count-Min Sketch implementation."""
    print("\n" + "="*60)
    print("Testing Count-Min Sketch")
    print("="*60)
    
    try:
        from src.sketches.count_min_sketch import CountMinSketch
        
        # Create sketch
        sketch = CountMinSketch(width=256, depth=4, device='cpu')
        print(f"✓ Created sketch: {sketch}")
        
        # Test update and query
        key = torch.randn(64)
        sketch.update(key, 1.0)
        freq = sketch.query(key)
        print(f"✓ Update and query working (freq: {freq})")
        
        # Test top-k
        for i in range(10):
            sketch.update(torch.randn(64), float(i))
        topk = sketch.get_topk(5)
        print(f"✓ Top-k retrieval working ({len(topk)} items)")
        
        # Test memory tracking
        mem = sketch.get_memory_usage()
        print(f"✓ Memory tracking: {mem / 1024:.2f} KB")
        
        return True
    except Exception as e:
        print(f"✗ Count-Min Sketch test failed: {e}")
        return False


def test_sketch_gpt2():
    """Test modified GPT-2 model."""
    print("\n" + "="*60)
    print("Testing Sketch GPT-2 Model")
    print("="*60)
    
    try:
        from src.models.sketch_gpt2 import SketchGPT2LMHeadModel
        from transformers import GPT2Tokenizer
        
        # Load model
        print("Loading GPT-2...")
        model = SketchGPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("✓ Model loaded successfully")
        
        # Enable sketch cache
        model.enable_sketch_cache(
            sketch_width=256,
            sketch_depth=4,
            strategy='topk',
            topk=32,
            max_cache_size=32
        )
        print("✓ Sketch cache enabled")
        
        # Test generation
        input_text = "The quick brown fox"
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        
        print("Testing generation...")
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False
            )
        
        generated = tokenizer.decode(outputs[0])
        print(f"✓ Generation working: '{generated[:50]}...'")
        
        # Test memory tracking
        mem = model.get_sketch_memory_usage()
        print(
            "✓ Sketch/cache memory (MB): "
            f"total={mem['total_mb']:.4f}, "
            f"sketch={mem['sketch_mb']:.4f}, "
            f"cache={mem['cache_mb']:.4f}, "
            f"avg_tokens/layer={mem['avg_tokens_per_layer']:.1f}"
        )
        # Smoke assertions: cache should be populated and capped
        assert mem["avg_tokens_per_layer"] > 0.0, "Cache did not populate during generate()"
        assert mem["avg_tokens_per_layer"] <= 32.0, "Cache exceeded max_cache_size during generate()"
        
        return True
    except Exception as e:
        print(f"✗ Sketch GPT-2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test evaluation metrics."""
    print("\n" + "="*60)
    print("Testing Evaluation Metrics")
    print("="*60)
    
    try:
        from src.evaluation.metrics import (
            MemoryTracker, LatencyTracker, QualityMetrics
        )
        
        # Test memory tracker
        memory_tracker = MemoryTracker(device='cpu')
        memory_tracker.record('test')
        summary = memory_tracker.get_summary()
        print(f"✓ MemoryTracker working")
        
        # Test latency tracker
        latency_tracker = LatencyTracker()
        latency_tracker.record_token_time(0.1)
        latency_tracker.record_token_time(0.2)
        tokens_per_sec = latency_tracker.get_tokens_per_second()
        print(f"✓ LatencyTracker working ({tokens_per_sec:.2f} tokens/s)")
        
        # Test BLEU
        ref = "The quick brown fox jumps over the lazy dog"
        hyp = "The quick brown fox jumps over a lazy dog"
        bleu = QualityMetrics.compute_bleu(ref, hyp)
        print(f"✓ BLEU computation working (score: {bleu:.4f})")
        
        # Test KL divergence
        p = torch.softmax(torch.randn(10), dim=0)
        q = torch.softmax(torch.randn(10), dim=0)
        kl = QualityMetrics.compute_kl_divergence(p, q)
        print(f"✓ KL divergence working (value: {kl:.4f})")
        
        return True
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False


def test_experiments_import():
    """Test that experiment scripts can be imported."""
    print("\n" + "="*60)
    print("Testing Experiment Scripts")
    print("="*60)
    
    try:
        # Test baseline
        from experiments import baseline
        print("✓ baseline.py imports successfully")
        
        # Test sketch experiments
        from experiments import sketch_experiments
        print("✓ sketch_experiments.py imports successfully")
        
        # Test baselines comparison
        from experiments import baselines_comparison
        print("✓ baselines_comparison.py imports successfully")
        
        # Test attention analysis
        from experiments import attention_analysis
        print("✓ attention_analysis.py imports successfully")
        
        return True
    except Exception as e:
        print(f"✗ Experiment import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("KV-CACHE SKETCH COMPRESSION - IMPLEMENTATION TESTS")
    print("="*60)
    print("\nThis script verifies that all components are working correctly.")
    print("Note: Full experiments require GPU and will take longer.")
    
    results = {
        'Count-Min Sketch': test_count_min_sketch(),
        'Sketch GPT-2': test_sketch_gpt2(),
        'Metrics': test_metrics(),
        'Experiments': test_experiments_import()
    }
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed! Implementation is ready.")
        print("\nNext steps:")
        print("  1. Run quick test: python run_all_experiments.py --quick")
        print("  2. Run full experiments: python run_all_experiments.py")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1
    
    print("="*60)
    return 0


if __name__ == '__main__':
    sys.exit(main())

