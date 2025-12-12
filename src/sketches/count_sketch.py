"""
Count-Sketch implementation for approximate frequency/importance estimation.

We use Count-Sketch as an alternative to Count-Min Sketch for scoring KV entries
for eviction. The model code expects the minimal interface:
- update(vec, value)
- batch_query_tensor(vecs) -> scores
- get_memory_usage()
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


class CountSketch:
    """
    Count-Sketch with universal hashing and sign hashing.

    Table shape: (depth, width).
    - bucket hash: h_d(x) in [0, width)
    - sign hash: s_d(x) in {-1, +1}

    Query estimate is the median over d of: s_d(x) * table[d, h_d(x)].
    """

    def __init__(
        self,
        width: int,
        depth: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ):
        self.width = int(width)
        self.depth = int(depth)
        self.device = device
        self.dtype = dtype

        self.table = torch.zeros(self.depth, self.width, device=device, dtype=dtype)

        # Large prime for universal hashing
        self.prime = 2**31 - 1

        # Deterministic parameters (keep stable across runs)
        rng = np.random.RandomState(seed)

        # Bucket hash params
        self.hash_a = torch.tensor(
            [rng.randint(1, self.prime) for _ in range(self.depth)],
            device=device,
            dtype=torch.int64,
        )
        self.hash_b = torch.tensor(
            [rng.randint(0, self.prime) for _ in range(self.depth)],
            device=device,
            dtype=torch.int64,
        )

        # Sign hash params (independent)
        self.sign_a = torch.tensor(
            [rng.randint(1, self.prime) for _ in range(self.depth)],
            device=device,
            dtype=torch.int64,
        )
        self.sign_b = torch.tensor(
            [rng.randint(0, self.prime) for _ in range(self.depth)],
            device=device,
            dtype=torch.int64,
        )

        # Optional history (debug/top-k is not required by the model)
        self.key_history = []
        self.max_history = 10_000

    def _key_to_int(self, key: torch.Tensor) -> torch.Tensor:
        if key.device != self.table.device:
            key = key.to(self.table.device)
        return torch.sum(torch.abs(key)).to(torch.int64)

    def _hash_buckets(self, key_vals: torch.Tensor) -> torch.Tensor:
        if key_vals.device != self.table.device:
            key_vals = key_vals.to(self.table.device)
        key_vals = key_vals.to(torch.int64)
        hv = (
            self.hash_a.view(self.depth, 1) * key_vals.view(1, -1)
            + self.hash_b.view(self.depth, 1)
        ) % self.prime
        return hv % self.width  # (depth, n)

    def _hash_signs(self, key_vals: torch.Tensor) -> torch.Tensor:
        if key_vals.device != self.table.device:
            key_vals = key_vals.to(self.table.device)
        key_vals = key_vals.to(torch.int64)
        hv = (
            self.sign_a.view(self.depth, 1) * key_vals.view(1, -1)
            + self.sign_b.view(self.depth, 1)
        ) % self.prime
        bits = hv & 1  # (depth, n) in {0,1}
        # map 0->-1, 1->+1
        return bits.to(torch.int8) * 2 - 1

    def update(self, vec: torch.Tensor, value: float = 1.0) -> None:
        if vec.device != self.table.device:
            vec = vec.to(self.table.device)

        key_val = self._key_to_int(vec).view(1)  # (1,)
        buckets = self._hash_buckets(key_val)  # (depth, 1)
        signs = self._hash_signs(key_val).to(torch.int64)  # (depth, 1)

        v = torch.tensor([float(value)], device=self.table.device, dtype=self.dtype)
        for d in range(self.depth):
            signed_v = v * signs[d].to(self.dtype)
            self.table[d].index_add_(0, buckets[d], signed_v)

        if len(self.key_history) < self.max_history:
            self.key_history.append(vec.detach().cpu())

    def batch_query_tensor(self, vecs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vecs: (n, dim)
        Returns:
            estimates: (n,) float tensor on sketch device
        """
        if vecs.numel() == 0:
            return torch.zeros(0, device=self.table.device, dtype=self.dtype)
        if vecs.device != self.table.device:
            vecs = vecs.to(self.table.device)

        key_vals = torch.sum(torch.abs(vecs), dim=1).to(torch.int64)  # (n,)
        buckets = self._hash_buckets(key_vals)  # (depth, n)
        signs = self._hash_signs(key_vals).to(torch.int64)  # (depth, n)

        gathered = []
        for d in range(self.depth):
            vals = self.table[d].gather(0, buckets[d])  # (n,)
            gathered.append(vals * signs[d].to(self.dtype))
        stacked = torch.stack(gathered, dim=0)  # (depth, n)

        # Use median (Count-Sketch property); torch.median returns values.
        return torch.median(stacked, dim=0).values.to(self.dtype)

    def get_memory_usage(self) -> int:
        table_bytes = self.table.element_size() * self.table.nelement()
        # rough history estimate (not used in core scoring)
        history_bytes = len(self.key_history) * 4 * 64
        return int(table_bytes + history_bytes)

    def reset(self) -> None:
        self.table.zero_()
        self.key_history.clear()

    def __repr__(self) -> str:
        return (
            f"CountSketch(width={self.width}, depth={self.depth}, "
            f"device={self.device}, memory={self.get_memory_usage() / 1024:.2f} KB)"
        )

