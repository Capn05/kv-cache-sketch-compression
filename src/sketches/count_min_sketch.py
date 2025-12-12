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
        
        # Random hash parameters for each depth (stored as torch tensors on device)
        np.random.seed(42)  # For reproducibility
        a_list = []
        b_list = []
        for _ in range(depth):
            a_list.append(np.random.randint(1, self.prime))
            b_list.append(np.random.randint(0, self.prime))
        self.hash_a = torch.tensor(a_list, device=device, dtype=torch.int64)  # (depth,)
        self.hash_b = torch.tensor(b_list, device=device, dtype=torch.int64)  # (depth,)
        
        # Track which keys have been inserted (for top-k retrieval)
        self.key_history = []
        self.max_history = 10000  # Limit history size
        
    def _key_to_int(self, key: torch.Tensor) -> torch.Tensor:
        """
        Convert a key vector to an int64 scalar for hashing (stays on device).
        
        Args:
            key: Input key tensor (can be multi-dimensional)
            
        Returns:
            int64 scalar tensor on the sketch device
        """
        if key.device != self.sketch.device:
            key = key.to(self.sketch.device)
        # Use sum(abs(key)) as a deterministic scalar; keep it on GPU to avoid sync.
        return torch.sum(torch.abs(key)).to(torch.int64)

    def _hash_buckets(self, key_vals: torch.Tensor) -> torch.Tensor:
        """
        Compute bucket indices for a batch of key_vals.

        Args:
            key_vals: int64 tensor shape (n,)

        Returns:
            buckets: int64 tensor shape (depth, n) with values in [0, width-1]
        """
        if key_vals.device != self.sketch.device:
            key_vals = key_vals.to(self.sketch.device)
        key_vals = key_vals.to(torch.int64)
        # (depth, 1) * (1, n) -> (depth, n)
        hv = (self.hash_a.view(self.depth, 1) * key_vals.view(1, -1) + self.hash_b.view(self.depth, 1)) % self.prime
        return hv % self.width
    
    def update(self, key: torch.Tensor, value: float = 1.0):
        """
        Update sketch with a key-value pair.
        
        Args:
            key: Key tensor (e.g., attention key vector)
            value: Value to add (default: 1.0 for frequency counting)
        """
        # Ensure key is on correct device
        if key.device != self.sketch.device:
            key = key.to(self.sketch.device)

        key_val = self._key_to_int(key).view(1)  # (1,)
        buckets = self._hash_buckets(key_val)  # (depth, 1)
        val = torch.tensor([value], device=self.sketch.device, dtype=self.dtype)
        for d in range(self.depth):
            self.sketch[d].index_add_(0, buckets[d], val)
        
        # Track key for top-k retrieval
        if len(self.key_history) < self.max_history:
            self.key_history.append(key.detach().cpu())
    
    def query_tensor(self, key: torch.Tensor) -> torch.Tensor:
        """
        Query estimated frequency/value for a key (returns a scalar tensor on device).
        
        Uses the minimum estimate across all hash functions (CM sketch property).
        
        Args:
            key: Key tensor to query
            
        Returns:
            Estimated frequency/value
        """
        if key.device != self.sketch.device:
            key = key.to(self.sketch.device)
        key_val = self._key_to_int(key).view(1)  # (1,)
        buckets = self._hash_buckets(key_val)  # (depth, 1)
        # gather -> (depth, 1)
        gathered = torch.stack(
            [self.sketch[d].gather(0, buckets[d]) for d in range(self.depth)],
            dim=0,
        )
        return torch.min(gathered).view(())

    def query(self, key: torch.Tensor) -> float:
        """
        Query estimated frequency/value for a key (returns python float).
        """
        return float(self.query_tensor(key).item())
    
    def batch_query(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Query multiple keys in batch.
        
        Args:
            keys: Batch of key tensors, shape (batch_size, key_dim)
            
        Returns:
            Estimated values, shape (batch_size,)
        """
        return self.batch_query_tensor(keys)

    def batch_query_tensor(self, keys: torch.Tensor) -> torch.Tensor:
        """
        Query multiple keys in batch (returns tensor on device).

        Args:
            keys: shape (n, key_dim)
        Returns:
            estimates: shape (n,) on sketch device
        """
        if keys.device != self.sketch.device:
            keys = keys.to(self.sketch.device)
        # Convert each key to int64 scalar (n,)
        key_vals = torch.sum(torch.abs(keys), dim=1).to(torch.int64)
        buckets = self._hash_buckets(key_vals)  # (depth, n)
        gathered = []
        for d in range(self.depth):
            gathered.append(self.sketch[d].gather(0, buckets[d]))
        stacked = torch.stack(gathered, dim=0)  # (depth, n)
        return torch.min(stacked, dim=0).values.to(self.dtype)
    
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

