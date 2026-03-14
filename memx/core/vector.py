"""FAISS-backed vector index for nearest-neighbour search."""

import threading
from typing import Tuple

import faiss
import numpy as np


class VectorIndex:
    """Thread-safe wrapper around a FAISS ``IndexFlatIP`` (inner-product / cosine).

    Vectors **must** be L2-normalised before insertion so that inner-product
    equals cosine similarity.  All operations are guarded by a ``threading.Lock``
    because FAISS indexes are **not** thread-safe for concurrent add+search.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        with self._lock:
            return self.index.ntotal

    def add(self, vector: np.ndarray) -> None:
        """Add a single vector ``(dim,)`` or batch ``(N, dim)``."""
        vec = np.ascontiguousarray(vector, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        with self._lock:
            self.index.add(vec)

    def search(self, query: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, indices)`` arrays of shape ``(top_k,)``."""
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        with self._lock:
            n = self.index.ntotal
            k = min(top_k, n) if n > 0 else 0
            if k == 0:
                return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
            scores, indices = self.index.search(q, k)
        return scores[0], indices[0]

    def reset(self) -> None:
        """Remove all vectors."""
        with self._lock:
            self.index.reset()
