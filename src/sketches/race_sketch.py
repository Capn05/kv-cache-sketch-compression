"""
Minimal RACE-style sketch for similarity/density scoring.

This is a pragmatic implementation intended for KV eviction scoring:
it estimates how often we've seen vectors *similar* to the queried vector.

Design:
- Maintain R random projections (rows of W) and a (R, B) count table.
- For a vector x, compute projections p = x @ W^T.
- Normalize p to roughly [-1, 1] using (||x||*||w|| + eps) and clamp.
- Bucketize each projection into B bins.
- Update: increment counts for those (r, bucket_r) positions.
- Query: return mean counts across the R buckets.

The model expects:
- update(vec, value)
- batch_query_tensor(vecs) -> scores
- get_memory_usage()
"""

from __future__ import annotations

import math
from typing import Optional

import torch


class RaceSketch:
    def __init__(
        self,
        bins: int = 512,
        num_projections: int = 16,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ):
        self.bins = int(bins)
        self.num_projections = int(num_projections)
        self.device = device
        self.dtype = dtype
        self.seed = int(seed)

        # Count table (R, B)
        self.table = torch.zeros(
            self.num_projections, self.bins, device=device, dtype=dtype
        )

        # Lazy-init projection matrix when we see the first vector.
        self._dim: Optional[int] = None
        self._W: Optional[torch.Tensor] = None  # (R, dim)
        self._W_norm: Optional[torch.Tensor] = None  # (R,)

    def _ensure_W(self, dim: int) -> None:
        if self._W is not None:
            if self._dim != dim:
                raise ValueError(
                    f"RaceSketch dimensionality changed: was {self._dim}, now {dim}"
                )
            return

        self._dim = int(dim)
        g = torch.Generator(device=self.table.device)
        g.manual_seed(self.seed)
        # Standard normal random projections
        W = torch.randn(
            self.num_projections, self._dim, device=self.table.device, generator=g
        )
        self._W = W
        self._W_norm = torch.norm(W, dim=1).clamp_min(1e-12)  # (R,)

    def _bucketize(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj: (R, n) normalized to approximately [-1, 1]
        Returns:
            buckets: (R, n) int64 in [0, bins-1]
        """
        # map [-1,1] -> [0, bins)
        scaled = (proj + 1.0) * 0.5 * float(self.bins)
        buckets = torch.floor(scaled).to(torch.int64)
        return buckets.clamp_(0, self.bins - 1)

    def update(self, vec: torch.Tensor, value: float = 1.0) -> None:
        if vec.device != self.table.device:
            vec = vec.to(self.table.device)
        if vec.ndim != 1:
            vec = vec.view(-1)

        # Project in float32 for numerical stability / mixed precision compatibility.
        vec_f = vec.to(torch.float32)

        self._ensure_W(vec_f.shape[0])
        assert self._W is not None and self._W_norm is not None

        # (R,)
        p = torch.matmul(self._W.to(torch.float32), vec_f)  # (R,)
        denom = (torch.norm(vec_f).clamp_min(1e-12) * self._W_norm.to(torch.float32)).clamp_min(1e-12)
        pn = (p / denom).clamp(-1.0, 1.0)  # (R,)
        buckets = self._bucketize(pn.view(self.num_projections, 1))  # (R,1)

        v = torch.tensor([float(value)], device=self.table.device, dtype=self.dtype)
        for r in range(self.num_projections):
            self.table[r].index_add_(0, buckets[r].view(-1), v)

    def batch_query_tensor(self, vecs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vecs: (n, dim)
        Returns:
            scores: (n,) float on sketch device (higher => more frequent/similar)
        """
        if vecs.numel() == 0:
            return torch.zeros(0, device=self.table.device, dtype=self.dtype)
        if vecs.device != self.table.device:
            vecs = vecs.to(self.table.device)
        if vecs.ndim != 2:
            vecs = vecs.view(vecs.shape[0], -1)

        vecs_f = vecs.to(torch.float32)
        n, dim = int(vecs_f.shape[0]), int(vecs_f.shape[1])
        self._ensure_W(dim)
        assert self._W is not None and self._W_norm is not None

        # projections: (R, n)
        p = torch.matmul(self._W.to(torch.float32), vecs_f.t())  # (R, n)
        x_norm = torch.norm(vecs_f, dim=1).clamp_min(1e-12)  # (n,)
        denom = self._W_norm.to(torch.float32).view(-1, 1) * x_norm.view(1, -1)  # (R, n)
        pn = (p / denom.clamp_min(1e-12)).clamp(-1.0, 1.0)
        buckets = self._bucketize(pn)  # (R, n)

        gathered = []
        for r in range(self.num_projections):
            gathered.append(self.table[r].gather(0, buckets[r]))  # (n,)
        stacked = torch.stack(gathered, dim=0)  # (R, n)
        return torch.mean(stacked, dim=0).to(self.dtype)

    def get_memory_usage(self) -> int:
        table_bytes = self.table.element_size() * self.table.nelement()
        w_bytes = 0
        if self._W is not None:
            w_bytes = self._W.element_size() * self._W.nelement()
        return int(table_bytes + w_bytes)

    def reset(self) -> None:
        self.table.zero_()
        # keep W initialized for stable hashing

    def __repr__(self) -> str:
        return (
            f"RaceSketch(R={self.num_projections}, bins={self.bins}, "
            f"device={self.device}, memory={self.get_memory_usage() / 1024:.2f} KB)"
        )

