from __future__ import annotations

from typing import Protocol

import torch

from .count_min_sketch import CountMinSketch
from .count_sketch import CountSketch
from .race_sketch import RaceSketch


class SketchLike(Protocol):
    """
    Minimal interface the model expects from a sketch implementation.

    The sketch is used as an *importance estimator* for KV eviction:
    higher score => token/key is more likely to be kept.
    """

    def update(self, vec: torch.Tensor, value: float = 1.0) -> None: ...

    def batch_query_tensor(self, vecs: torch.Tensor) -> torch.Tensor: ...

    def get_memory_usage(self) -> int: ...


__all__ = ["CountMinSketch", "CountSketch", "RaceSketch", "SketchLike"]

