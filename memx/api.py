"""MemX — the AI Memory Operating System.

One-class interface to the entire memory system:
add, update, merge, rag, compress, reflect, inspect, namespace, decay.
"""

import logging
from typing import Dict, List, Optional, Callable

from .types import Memory, MemoryType, RetrievalExplanation
from .core.braintrace import BrainTrace
from .backends.sqlite_backend import SQLiteBackend

logger = logging.getLogger(__name__)


class MemX:
    """AI Memory Operating System — one class, infinite memory.

    Usage::

        from memx import MemX

        m = MemX()
        m.add("User prefers Python", importance=0.9)
        m.add("User mentioned liking Python")
        m.compress()  # merges similar memories
        results = m.rag("programming language preference")
        m.inspect("programming")  # explain why results scored
    """

    def __init__(
        self,
        backend: str = "sqlite",
        db_path: Optional[str] = None,
        model: Optional[str] = None,
    ):
        if backend in ("sqlite", "memory"):
            self._backend = SQLiteBackend(db_path or ":memory:")
        else:
            raise ValueError(f"Unknown backend '{backend}'. Supported: sqlite, memory")

        model_name = model or "all-MiniLM-L6-v2"
        self.brain = BrainTrace(backend=self._backend, model_name=model_name)
        logger.info("MemX Memory OS initialised (backend=%s)", backend)

    # ══════════════════════════════════════════════════════════════
    # CORE — add, update, delete
    # ══════════════════════════════════════════════════════════════

    def add(
        self,
        content: str,
        mem_type: Optional[MemoryType] = None,
        metadata: Optional[Dict] = None,
        importance: Optional[float] = None,
        namespace: str = "default",
        source: str = "user",
    ) -> str:
        """Store a memory. Auto-classifies type and importance if not supplied."""
        return self.brain.add(
            content, mem_type=mem_type, metadata=metadata,
            importance=importance, namespace=namespace, source=source,
        )

    def update(self, memory_id: str, new_content: str, merge: bool = False) -> Optional[str]:
        """Update a memory's content. Returns new memory ID.

        Args:
            memory_id: ID of the memory to update.
            new_content: New content to replace or merge with.
            merge: If True, combines old+new. If False, replaces entirely.
        """
        return self.brain.update(memory_id, new_content, merge=merge)

    def delete(self, memory_id: str) -> bool:
        """Soft-delete a memory (mark as inactive, excluded from RAG)."""
        return self.brain.delete(memory_id)

    # ══════════════════════════════════════════════════════════════
    # RETRIEVE — hybrid RAG
    # ══════════════════════════════════════════════════════════════

    def rag(
        self,
        query: str,
        top_k: int = 5,
        namespace: Optional[str] = None,
    ) -> List[Memory]:
        """Importance-weighted hybrid retrieval.

        Combines: vector similarity + keyword match + importance + recency + frequency.
        """
        return self.brain.rag(query, top_k=top_k, namespace=namespace)

    def get(self, memory_id: str) -> Optional[Memory]:
        """Fetch a single memory by ID."""
        return self.brain.get(memory_id)

    # ══════════════════════════════════════════════════════════════
    # INTELLIGENCE — compress, reflect, decay
    # ══════════════════════════════════════════════════════════════

    def compress(
        self,
        namespace: Optional[str] = None,
        threshold: float = 0.75,
        summarizer: Optional[Callable] = None,
    ) -> Dict:
        """Compress similar memories into merged summaries.

        Pass a summarizer function ``f(texts) -> summary`` for LLM-quality compression.
        Without it, uses sentence-level deduplication.
        """
        return self.brain.compress(namespace, threshold, summarizer)

    def reflect(
        self,
        namespace: Optional[str] = None,
        summarizer: Optional[Callable] = None,
    ) -> List[Memory]:
        """Generate reflection insights from accumulated memories.

        Returns a list of new REFLECTION-type memories.
        """
        return self.brain.reflect(namespace, summarizer)

    def reflect_conversation(
        self,
        messages: List[str],
        summarizer: Optional[Callable] = None,
    ) -> Optional[Memory]:
        """Reflect on a conversation to produce a single insight memory."""
        return self.brain.reflect_conversation(messages, summarizer)

    def decay(self) -> List[str]:
        """Run decay sweep — deactivate memories that have expired. Returns deactivated IDs."""
        return self.brain.run_decay()

    # ══════════════════════════════════════════════════════════════
    # OBSERVABILITY — inspect, explain
    # ══════════════════════════════════════════════════════════════

    def inspect(
        self,
        query: str,
        top_k: int = 5,
        namespace: Optional[str] = None,
    ) -> List[RetrievalExplanation]:
        """Explain why memories are retrieved for a query.

        Returns breakdown: vector_score, keyword_score, importance, recency, frequency.
        """
        return self.brain.inspect(query, top_k, namespace)

    # ══════════════════════════════════════════════════════════════
    # MULTI-AGENT — namespaces
    # ══════════════════════════════════════════════════════════════

    def namespace_stats(self, namespace: str) -> Dict:
        """Get stats for a specific namespace."""
        return self.brain.stats(namespace)

    def namespaces(self) -> List[str]:
        """List all active namespaces."""
        return self.brain.stats()["namespaces"]

    # ══════════════════════════════════════════════════════════════
    # GRAPH
    # ══════════════════════════════════════════════════════════════

    def link(self, src_id: str, dst_id: str, weight: float = 1.0, label: str = "") -> None:
        """Add a causal link between two memories."""
        self.brain.link(src_id, dst_id, weight, label)

    # ══════════════════════════════════════════════════════════════
    # UTILITIES
    # ══════════════════════════════════════════════════════════════

    def stats(self) -> Dict:
        """Return memory counts, types, importance, namespaces."""
        return self.brain.stats()

    def all(self, namespace: Optional[str] = None, include_inactive: bool = False) -> List[Memory]:
        """Return all memories, optionally filtered by namespace."""
        return self.brain.all_memories(namespace, include_inactive)

    def clear(self, namespace: Optional[str] = None) -> None:
        """Delete all memories. If namespace given, only that namespace."""
        self.brain.clear(namespace)
        if namespace is None:
            self._backend.clear()

    def __repr__(self) -> str:
        s = self.brain.stats()
        ns = len(s["namespaces"])
        return f"MemX(memories={s['total']}, inactive={s['inactive']}, namespaces={ns})"
