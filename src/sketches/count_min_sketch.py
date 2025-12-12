"""
Count-Min Sketch implementation for KV-cache compression.

This module implements a Count-Min Sketch data structure optimized for
tracking key-value pairs in transformer attention mechanisms.
"""

import torch
import numpy as np
from typing import Optional, Tuple, List


class CountMinSketch:
    """
    Count-Min Sketch for approximate frequency estimation and top-k retrieval.
    
    This implementation is designed for GPU acceleration and batched operations,
    making it suitable for real-time inference in transformer models.
    
    Args:
        width: Number of buckets per hash function (sketch width)
        depth: Number of hash functions (sketch depth)
        device: Device to run computations on ('cpu' or 'cuda')
        dtype: Data type for sketch values (default: torch.float32)
    """
    
    def __init__(
        self,
        width: int,
        depth: int,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        self.width = width
        self.depth = depth
        self.device = device
        self.dtype = dtype
        
        # Initialize sketch table (depth x width)
        self.sketch = torch.zeros(depth, width, device=device, dtype=dtype)
        
        # Hash parameters for universal hashing: h(x) = ((a*x + b) mod p) mod width
        # Use large prime for better distribution
        self.prime = 2**31 - 1
        
        # Random hash parameters for each depth
        np.random.seed(42)  # For reproducibility
        self.hash_params = []
        for _ in range(depth):
            a = np.random.randint(1, self.prime)
            b = np.random.randint(0, self.prime)
            self.hash_params.append((a, b))
        
        # Track which keys have been inserted (for top-k retrieval)
        self.key_history = []
        self.max_history = 10000  # Limit history size
        
    def _hash(self, key: torch.Tensor, depth_idx: int) -> torch.Tensor:
        """
        Compute hash value for a key at a specific depth.
        
        Args:
            key: Input key tensor (can be multi-dimensional)
            depth_idx: Which hash function to use (0 to depth-1)
            
        Returns:
            Hash values in range [0, width-1]
        """
        a, b = self.hash_params[depth_idx]
        
        # Convert key to single scalar value for hashing
        # Use sum of all elements as the key representation
        key_val = int(key.sum().abs().item())
        
        # Universal hashing
        hash_val = ((a * key_val + b) % self.prime) % self.width
        return hash_val
    
    def update(self, key: torch.Tensor, value: float = 1.0):
        """
        Update sketch with a key-value pair.
        
        Args:
            key: Key tensor (e.g., attention key vector)
            value: Value to add (default: 1.0 for frequency counting)
        """
        # Ensure key is on correct device
        if key.device != self.device:
            key = key.to(self.device)
        
        # Update all depth rows
        for d in range(self.depth):
            bucket = self._hash(key, d)
            self.sketch[d, bucket] += value
        
        # Track key for top-k retrieval
        if len(self.key_history) < self.max_history:
            self.key_history.append(key.detach().cpu())
    
    def query(self, key: torch.Tensor) -> float:
        """
        Query estimated frequency/value for a key.
        
        Uses the minimum estimate across all hash functions (CM sketch property).
        
        Args:
            key: Key tensor to query
            
        Returns:
            Estimated frequency/value
        """
        # Ensure key is on correct device
        if key.device != self.device:
            key = key.to(self.device)
        
        # Get minimum estimate across all depths
        estimates = []
        for d in range(self.depth):
            bucket = self._hash(key, d)
            estimates.append(self.sketch[d, bucket].item())
        
        return min(estimates)
    
    def batch_query(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Query multiple keys in batch.
        
        Args:
            keys: Batch of key tensors, shape (batch_size, key_dim)
            
        Returns:
            Estimated values, shape (batch_size,)
        """
        batch_size = keys.shape[0]
        estimates = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
        
        for i in range(batch_size):
            estimates[i] = self.query(keys[i])
        
        return estimates
    
    def get_topk(self, k: int) -> List[Tuple[torch.Tensor, float]]:
        """
        Retrieve top-k keys by estimated frequency.
        
        Args:
            k: Number of top keys to retrieve
            
        Returns:
            List of (key, estimated_frequency) tuples
        """
        if not self.key_history:
            return []
        
        # Query all historical keys
        key_scores = []
        for key in self.key_history:
            score = self.query(key)
            key_scores.append((key, score))
        
        # Sort by score and return top-k
        key_scores.sort(key=lambda x: x[1], reverse=True)
        return key_scores[:k]
    
    def get_topk_indices(self, k: int, sequence_length: int) -> torch.Tensor:
        """
        Get indices of top-k tokens in the sequence.
        
        Args:
            k: Number of top tokens to retrieve
            sequence_length: Total length of sequence
            
        Returns:
            Tensor of shape (min(k, sequence_length),) with top-k indices
        """
        # Get scores for all positions
        if sequence_length == 0 or not self.key_history:
            return torch.tensor([], dtype=torch.long, device=self.device)
        
        # Limit to actual sequence length
        available_keys = min(len(self.key_history), sequence_length)
        scores = torch.zeros(available_keys, device=self.device)
        
        for i in range(available_keys):
            scores[i] = self.query(self.key_history[i])
        
        # Get top-k indices
        k_actual = min(k, available_keys)
        if k_actual == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)
        
        topk_scores, topk_indices = torch.topk(scores, k_actual)
        return topk_indices
    
    def reset(self):
        """Reset sketch to empty state."""
        self.sketch.zero_()
        self.key_history.clear()
    
    def get_memory_usage(self) -> int:
        """
        Get memory footprint in bytes.
        
        Returns:
            Memory usage in bytes
        """
        sketch_memory = self.sketch.element_size() * self.sketch.nelement()
        # Approximate history memory (each key stored as tensor)
        history_memory = len(self.key_history) * 4 * 64  # Assume 64-dim keys, 4 bytes per float
        return sketch_memory + history_memory
    
    def get_compression_ratio(self, full_cache_size: int) -> float:
        """
        Calculate compression ratio compared to full cache.
        
        Args:
            full_cache_size: Size of full KV cache in bytes
            
        Returns:
            Compression ratio (sketch_size / full_cache_size)
        """
        sketch_size = self.get_memory_usage()
        return sketch_size / full_cache_size if full_cache_size > 0 else 0.0
    
    def __repr__(self) -> str:
        return (f"CountMinSketch(width={self.width}, depth={self.depth}, "
                f"device={self.device}, memory={self.get_memory_usage() / 1024:.2f} KB)")

