"""
Evaluation metrics for KV-cache compression experiments.

This module provides utilities for measuring memory usage, latency,
and text generation quality metrics.
"""

import torch
import time
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import math


class MemoryTracker:
    """Track GPU and CPU memory usage during model inference."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.is_cuda = device.startswith('cuda')
        self.reset()
    
    def reset(self):
        """Reset memory tracking."""
        if self.is_cuda and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
        self.measurements = []
    
    def record(self, label: str = ''):
        """Record current memory usage."""
        if self.is_cuda and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            peak = torch.cuda.max_memory_allocated(self.device)
            
            self.measurements.append({
                'label': label,
                'allocated_mb': allocated / (1024 ** 2),
                'reserved_mb': reserved / (1024 ** 2),
                'peak_mb': peak / (1024 ** 2)
            })
        else:
            # CPU memory tracking (approximate)
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            self.measurements.append({
                'label': label,
                'allocated_mb': mem_info.rss / (1024 ** 2),
                'reserved_mb': 0,
                'peak_mb': mem_info.rss / (1024 ** 2)
            })
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if self.is_cuda and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        return 0.0
    
    def get_current_memory(self) -> float:
        """Get current allocated memory in MB."""
        if self.is_cuda and torch.cuda.is_available():
            return torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        return 0.0
    
    def get_summary(self) -> Dict:
        """Get summary statistics of memory usage."""
        if not self.measurements:
            return {}
        
        peak_memories = [m['peak_mb'] for m in self.measurements]
        allocated_memories = [m['allocated_mb'] for m in self.measurements]
        
        return {
            'peak_memory_mb': max(peak_memories),
            'avg_allocated_mb': np.mean(allocated_memories),
            'final_allocated_mb': allocated_memories[-1] if allocated_memories else 0
        }


class LatencyTracker:
    """Track inference latency and throughput."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset latency tracking."""
        self.token_times = []
        self.total_time = 0.0
        self.start_time = None
    
    def start(self):
        """Start timing."""
        self.start_time = time.time()
    
    def stop(self):
        """Stop timing and record."""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.token_times.append(elapsed)
            self.total_time += elapsed
            self.start_time = None
    
    def record_token_time(self, elapsed: float):
        """Manually record token generation time."""
        self.token_times.append(elapsed)
        self.total_time += elapsed
    
    def get_tokens_per_second(self) -> float:
        """Calculate tokens per second."""
        if self.total_time == 0:
            return 0.0
        return len(self.token_times) / self.total_time
    
    def get_avg_time_per_token(self) -> float:
        """Get average time per token in seconds."""
        if not self.token_times:
            return 0.0
        return np.mean(self.token_times)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.token_times:
            return {}
        
        return {
            'total_time_s': self.total_time,
            'num_tokens': len(self.token_times),
            'tokens_per_second': self.get_tokens_per_second(),
            'avg_time_per_token_ms': self.get_avg_time_per_token() * 1000,
            'median_time_per_token_ms': np.median(self.token_times) * 1000,
            'std_time_per_token_ms': np.std(self.token_times) * 1000
        }


class QualityMetrics:
    """Compute text generation quality metrics."""
    
    @staticmethod
    def compute_perplexity(
        model,
        tokenizer,
        texts: List[str],
        device: str = 'cuda',
        max_length: int = 1024
    ) -> float:
        """
        Compute perplexity on a list of texts.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            texts: List of text strings
            device: Device to run on
            max_length: Maximum sequence length
            
        Returns:
            Average perplexity across texts
        """
        model.eval()
        total_nll = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                encodings = tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=max_length,
                    truncation=True
                )
                input_ids = encodings['input_ids'].to(device)
                
                # Forward pass
                outputs = model(input_ids, labels=input_ids)
                
                # Accumulate negative log-likelihood
                nll = outputs.loss.item() * input_ids.size(1)
                total_nll += nll
                total_tokens += input_ids.size(1)
        
        # Compute perplexity
        avg_nll = total_nll / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_nll)
        
        return perplexity
    
    @staticmethod
    def compute_bleu(
        reference: str,
        hypothesis: str,
        max_n: int = 4
    ) -> float:
        """
        Compute BLEU score (simplified implementation).
        
        Args:
            reference: Reference text
            hypothesis: Generated text
            max_n: Maximum n-gram size
            
        Returns:
            BLEU score (0-1)
        """
        from collections import Counter
        
        def get_ngrams(tokens: List[str], n: int) -> Counter:
            """Extract n-grams from token list."""
            return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
        
        # Tokenize (simple whitespace split)
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        if not hyp_tokens:
            return 0.0
        
        # Brevity penalty
        bp = 1.0 if len(hyp_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(hyp_tokens))
        
        # Compute precision for each n-gram size
        precisions = []
        for n in range(1, max_n + 1):
            ref_ngrams = get_ngrams(ref_tokens, n)
            hyp_ngrams = get_ngrams(hyp_tokens, n)
            
            if not hyp_ngrams:
                precisions.append(0.0)
                continue
            
            # Clipped counts
            matches = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
            total = sum(hyp_ngrams.values())
            
            precision = matches / total if total > 0 else 0.0
            precisions.append(precision)
        
        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            geo_mean = 0.0
        
        bleu = bp * geo_mean
        return bleu
    
    @staticmethod
    def compute_kl_divergence(
        p: torch.Tensor,
        q: torch.Tensor,
        eps: float = 1e-10
    ) -> float:
        """
        Compute KL divergence between two probability distributions.
        
        Args:
            p: True distribution
            q: Approximate distribution
            eps: Small constant for numerical stability
            
        Returns:
            KL divergence value
        """
        p = p + eps
        q = q + eps
        p = p / p.sum()
        q = q / q.sum()
        
        kl = (p * torch.log(p / q)).sum().item()
        return kl
    
    @staticmethod
    def compute_cosine_similarity(
        a: torch.Tensor,
        b: torch.Tensor
    ) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity (-1 to 1)
        """
        a_norm = torch.nn.functional.normalize(a.flatten(), dim=0)
        b_norm = torch.nn.functional.normalize(b.flatten(), dim=0)
        
        similarity = torch.dot(a_norm, b_norm).item()
        return similarity


class ExperimentLogger:
    """Log and aggregate experimental results."""
    
    def __init__(self):
        self.results = defaultdict(list)
        self.metadata = {}
    
    def log(self, key: str, value: any):
        """Log a metric value."""
        self.results[key].append(value)
    
    def log_dict(self, data: Dict):
        """Log multiple metrics at once."""
        for key, value in data.items():
            self.log(key, value)
    
    def set_metadata(self, key: str, value: any):
        """Set metadata (non-aggregated info)."""
        self.metadata[key] = value
    
    def get_summary(self) -> Dict:
        """Get summary statistics for all metrics."""
        summary = {}
        for key, values in self.results.items():
            if not values:
                continue
            
            # Handle numeric values
            if isinstance(values[0], (int, float)):
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
                summary[f'{key}_min'] = np.min(values)
                summary[f'{key}_max'] = np.max(values)
            else:
                # Just keep the list for non-numeric
                summary[key] = values
        
        # Add metadata
        summary.update(self.metadata)
        
        return summary
    
    def save(self, filepath: str):
        """Save results to file."""
        import json
        summary = self.get_summary()
        
        # Convert numpy types to native Python for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        summary = convert_to_native(summary)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

