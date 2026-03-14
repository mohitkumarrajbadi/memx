"""Embedding abstraction with sentence-transformers and random fallback."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_DIMENSION = 384  # all-MiniLM-L6-v2 output dimension


class Embedder:
    """Embeds text into dense vectors.

    Uses ``sentence-transformers`` (``all-MiniLM-L6-v2``) when available,
    otherwise falls back to deterministic hash-based vectors so that the
    library works out-of-the-box without downloading a 90 MB model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.dim = _DIMENSION
        self._model = None
        self._model_name = model_name
        self._use_st = False
        self._try_load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _try_load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self._model_name)
            self._use_st = True
            logger.info("Loaded sentence-transformers model: %s", self._model_name)
        except ImportError:
            logger.info(
                "sentence-transformers not installed – using hash-based fallback embedder. "
                "Install with: pip install 'memx-ai[embeddings]'"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """Return a unit-norm float32 vector of shape ``(dim,)``."""
        if self._use_st and self._model is not None:
            vec = self._model.encode(text, convert_to_numpy=True).astype(np.float32)
        else:
            vec = self._hash_embed(text)
        # L2-normalise so dot-product == cosine similarity
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode multiple texts, returning ``(N, dim)`` float32 matrix."""
        if self._use_st and self._model is not None:
            vecs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        else:
            vecs = np.array([self._hash_embed(t) for t in texts], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _hash_embed(self, text: str) -> np.ndarray:
        """Deterministic hash-based embedding (no ML model needed)."""
        import hashlib

        digest = hashlib.sha512(text.encode("utf-8")).digest()
        # Expand hash to fill dimension
        repeats = (self.dim * 4 // len(digest)) + 1
        raw = (digest * repeats)[: self.dim * 4]
        vec = np.frombuffer(raw, dtype=np.float32).copy()
        return vec[: self.dim]
