"""BrainTrace — Memory Operating System engine.

Combines vector search, KV cache, causal graph, importance scoring,
compression, reflection, and namespace isolation into a unified
hybrid retrieval engine with full observability.
"""

import hashlib
import time
import logging
import re
import threading
from typing import List, Optional, Dict, Callable

import numpy as np

from ..types import Memory, MemoryType, RetrievalExplanation
from ..classify import auto_classify
from .embeddings import Embedder
from .vector import VectorIndex
from .kv import KVCache
from .graph import CausalGraph
from .importance import (
    estimate_importance,
    compute_recency_score,
    compute_frequency_score,
    run_decay_sweep,
)
from .updater import detect_contradiction, find_merge_candidates, create_updated_memory
from .inspector import explain_retrieval

logger = logging.getLogger(__name__)

# ── Hybrid scoring weights ──
_W_VECTOR = 0.35
_W_KEYWORD = 0.15
_W_IMPORTANCE = 0.25
_W_RECENCY = 0.15
_W_FREQUENCY = 0.10

_RECENCY_HALF_LIFE = 3600.0 * 24  # 24h


class BrainTrace:
    """Memory Operating System engine.

    Provides:
    - add/update/merge/delete lifecycle
    - Importance-weighted hybrid RAG
    - Multi-agent namespaces
    - Memory compression
    - Automatic reflection
    - Full retrieval observability
    """

    def __init__(
        self,
        backend: Optional[object] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.embedder = Embedder(model_name)
        self.vector_index = VectorIndex(dim=self.embedder.dim)
        self.kv = KVCache()
        self.graph = CausalGraph()
        self.backend = backend
        self._lock = threading.Lock()
        self._id_order: List[str] = []

    # ══════════════════════════════════════════════════════════════
    # WRITE — add, update, merge
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
        """Embed, classify, score importance, store, and index a new memory."""
        mem_id = hashlib.md5(content.encode("utf-8")).hexdigest()[:12]

        mem_type = mem_type or auto_classify(content)
        vector = self.embedder.encode(content)
        imp = importance if importance is not None else estimate_importance(content)
        now = time.time()

        with self._lock:
            if mem_id in self.kv:
                return mem_id

            memory = Memory(
                id=mem_id,
                type=mem_type,
                content=content,
                vector=vector,
                timestamp=now,
                score=0.0,
                metadata=metadata or {},
                importance=imp,
                namespace=namespace,
                source=source,
            )

            self.vector_index.add(vector)
            self._id_order.append(mem_id)
            self.kv.set(mem_id, memory)

        if self.backend is not None:
            try:
                self.backend.save(memory)
            except Exception:
                pass  # backend failures don't block in-memory ops

        logger.debug("Stored %s [%s, imp=%.2f, ns=%s]", mem_id, mem_type.name, imp, namespace)
        return mem_id

    def update(
        self,
        memory_id: str,
        new_content: str,
        merge: bool = False,
    ) -> Optional[str]:
        """Update an existing memory's content. Returns new memory ID.

        If merge=True, combines old+new. Otherwise replaces.
        Automatically handles contradiction detection.
        """
        old = self.kv.get(memory_id)
        if old is None:
            return None

        new_vector = self.embedder.encode(new_content)
        updated = create_updated_memory(old, new_content, new_vector, merge=merge)

        # Deactivate old
        with self._lock:
            old.active = False
            old.superseded_by = updated.id

            self.vector_index.add(updated.vector)
            self._id_order.append(updated.id)
            self.kv.set(updated.id, updated)

        if self.backend is not None:
            try:
                self.backend.save(updated)
            except Exception:
                pass

        logger.debug("Updated %s → %s (merge=%s)", memory_id, updated.id, merge)
        return updated.id

    def delete(self, memory_id: str) -> bool:
        """Soft-delete a memory (mark as inactive)."""
        mem = self.kv.get(memory_id)
        if mem is None:
            return False
        mem.active = False
        return True

    # ══════════════════════════════════════════════════════════════
    # READ — importance-weighted hybrid RAG
    # ══════════════════════════════════════════════════════════════

    def rag(
        self,
        query: str,
        top_k: int = 5,
        namespace: Optional[str] = None,
        include_inactive: bool = False,
    ) -> List[Memory]:
        """Importance-weighted hybrid retrieval.

        Scoring: vector similarity + keyword overlap + importance + recency + frequency.
        """
        query_vec = self.embedder.encode(query)

        with self._lock:
            if self.kv.size == 0:
                return []
            scores, indices = self.vector_index.search(query_vec, top_k=min(top_k * 5, self.kv.size))
            id_snap = list(self._id_order)

        now = time.time()
        query_tokens = set(re.findall(r"\w+", query.lower()))

        results: List[Memory] = []
        for idx, vec_score in zip(indices, scores):
            idx = int(idx)
            if idx < 0 or idx >= len(id_snap):
                continue
            mem = self.kv.get(id_snap[idx])
            if mem is None:
                continue
            if not include_inactive and not mem.active:
                continue
            if namespace and mem.namespace != namespace:
                continue

            # Keyword overlap
            content_tokens = set(re.findall(r"\w+", mem.content.lower()))
            matched = query_tokens & content_tokens
            keyword_score = min(len(matched) / max(len(query_tokens), 1), 1.0)

            # Recency & frequency
            recency = compute_recency_score(mem.timestamp, now, _RECENCY_HALF_LIFE)
            frequency = compute_frequency_score(mem.access_count)

            # Composite score
            combined = (
                _W_VECTOR * float(vec_score)
                + _W_KEYWORD * keyword_score
                + _W_IMPORTANCE * mem.importance
                + _W_RECENCY * recency
                + _W_FREQUENCY * frequency
            )

            scored = Memory(
                id=mem.id, type=mem.type, content=mem.content,
                vector=mem.vector, timestamp=mem.timestamp, score=combined,
                metadata=mem.metadata, importance=mem.importance,
                access_count=mem.access_count, last_accessed=mem.last_accessed,
                namespace=mem.namespace, source=mem.source,
                superseded_by=mem.superseded_by, active=mem.active,
            )
            results.append(scored)

        results.sort(key=lambda m: m.score, reverse=True)
        top_results = results[:top_k]

        # Update access counts for retrieved memories
        for r in top_results:
            original = self.kv.get(r.id)
            if original:
                original.access_count += 1
                original.last_accessed = now

        return top_results

    # ══════════════════════════════════════════════════════════════
    # INSPECT — observability
    # ══════════════════════════════════════════════════════════════

    def inspect(
        self,
        query: str,
        top_k: int = 5,
        namespace: Optional[str] = None,
    ) -> List[RetrievalExplanation]:
        """Explain why memories are retrieved for a query."""
        query_vec = self.embedder.encode(query)

        with self._lock:
            if self.kv.size == 0:
                return []
            scores, indices = self.vector_index.search(query_vec, top_k=min(top_k * 5, self.kv.size))
            id_snap = list(self._id_order)

        now = time.time()
        explanations = []

        for idx, vec_score in zip(indices, scores):
            idx = int(idx)
            if idx < 0 or idx >= len(id_snap):
                continue
            mem = self.kv.get(id_snap[idx])
            if mem is None or not mem.active:
                continue
            if namespace and mem.namespace != namespace:
                continue

            exp = explain_retrieval(query, mem, float(vec_score), now)
            explanations.append(exp)

        explanations.sort(key=lambda e: e.final_score, reverse=True)
        return explanations[:top_k]

    # ══════════════════════════════════════════════════════════════
    # COMPRESSION & REFLECTION
    # ══════════════════════════════════════════════════════════════

    def compress(
        self,
        namespace: Optional[str] = None,
        threshold: float = 0.75,
        summarizer: Optional[Callable] = None,
    ) -> Dict:
        """Run memory compression on a namespace."""
        from .compression import run_compression

        mems = [m for m in self.kv.all()
                if m.active and (namespace is None or m.namespace == namespace)]

        compressed, deactivated = run_compression(mems, threshold, summarizer=summarizer)

        # Store compressed memories
        for cm in compressed:
            self.vector_index.add(cm.vector)
            with self._lock:
                self._id_order.append(cm.id)
                self.kv.set(cm.id, cm)
            if self.backend:
                try:
                    self.backend.save(cm)
                except Exception:
                    pass

        return {"compressed": len(compressed), "deactivated": len(deactivated)}

    def reflect(
        self,
        namespace: Optional[str] = None,
        summarizer: Optional[Callable] = None,
    ) -> List[Memory]:
        """Generate reflection insights from accumulated memories."""
        from .reflection import reflect_on_memories

        mems = [m for m in self.kv.all()
                if m.active and (namespace is None or m.namespace == namespace)]

        reflections = reflect_on_memories(
            mems, embedder=self.embedder, summarizer=summarizer,
        )

        for ref in reflections:
            self.vector_index.add(ref.vector)
            with self._lock:
                self._id_order.append(ref.id)
                self.kv.set(ref.id, ref)
            if self.backend:
                try:
                    self.backend.save(ref)
                except Exception:
                    pass

        return reflections

    def reflect_conversation(
        self,
        messages: List[str],
        summarizer: Optional[Callable] = None,
    ) -> Optional[Memory]:
        """Reflect on a conversation to produce a single insight memory."""
        from .reflection import reflect_on_conversation

        ref = reflect_on_conversation(messages, self.embedder, summarizer)
        if ref:
            self.vector_index.add(ref.vector)
            with self._lock:
                self._id_order.append(ref.id)
                self.kv.set(ref.id, ref)
            if self.backend:
                try:
                    self.backend.save(ref)
                except Exception:
                    pass
        return ref

    # ══════════════════════════════════════════════════════════════
    # DECAY
    # ══════════════════════════════════════════════════════════════

    def run_decay(self) -> List[str]:
        """Sweep memories and deactivate those that have decayed."""
        active = [m for m in self.kv.all() if m.active]
        return run_decay_sweep(active)

    # ══════════════════════════════════════════════════════════════
    # GRAPH
    # ══════════════════════════════════════════════════════════════

    def link(self, src_id: str, dst_id: str, weight: float = 1.0, label: str = "") -> None:
        self.graph.add_link(src_id, dst_id, weight, label)

    # ══════════════════════════════════════════════════════════════
    # UTILS
    # ══════════════════════════════════════════════════════════════

    def get(self, mem_id: str) -> Optional[Memory]:
        return self.kv.get(mem_id)

    def all_memories(self, namespace: Optional[str] = None, include_inactive: bool = False) -> List[Memory]:
        mems = self.kv.all()
        if namespace:
            mems = [m for m in mems if m.namespace == namespace]
        if not include_inactive:
            mems = [m for m in mems if m.active]
        return mems

    def clear(self, namespace: Optional[str] = None) -> None:
        if namespace:
            for mem in self.kv.all():
                if mem.namespace == namespace:
                    mem.active = False
        else:
            with self._lock:
                self.kv.clear()
                self.vector_index.reset()
                self.graph.clear()
                self._id_order.clear()

    def stats(self, namespace: Optional[str] = None) -> Dict:
        mems = self.all_memories(namespace)
        type_counts: Dict[str, int] = {}
        total_importance = 0.0
        for mem in mems:
            type_counts[mem.type.name] = type_counts.get(mem.type.name, 0) + 1
            total_importance += mem.importance
        inactive = len([m for m in self.kv.all()
                        if not m.active and (namespace is None or m.namespace == namespace)])
        return {
            "total": len(mems),
            "inactive": inactive,
            "types": type_counts,
            "avg_importance": total_importance / max(len(mems), 1),
            "graph_edges": self.graph.num_edges,
            "namespaces": list(set(m.namespace for m in self.kv.all())),
        }
