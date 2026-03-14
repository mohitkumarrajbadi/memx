"""BrainTrace — Memory Operating System engine.

Combines vector search, KV cache, causal graph, importance scoring,
compression, reflection, and namespace isolation into a unified
hybrid retrieval engine with full observability.
"""

import hashlib
import time
import logging
import re
from typing import List, Optional, Dict, Callable

import numba
import numpy as np

from ..types import Memory, MemoryType, RetrievalExplanation
from ..classify import auto_classify
from .embeddings import Embedder
from .vector import VectorIndex
from .kv import KVCache
from .graph import CausalGraph
from .utils import RWLock, ReadContext, WriteContext, LRUCache
from .importance import (
    estimate_importance,
    compute_recency_score,
    compute_frequency_score,
    run_decay_sweep,
    _FREQUENCY_LOG_BASE,
    _MAX_FREQUENCY_BONUS,
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

@numba.njit(fastmath=True)
def fast_hybrid_score(
    v_vec: np.ndarray,
    v_imp: np.ndarray,
    v_ts: np.ndarray,
    v_ac: np.ndarray,
    v_kw: np.ndarray,
    now: float,
    w_vector: float,
    w_keyword: float,
    w_importance: float,
    w_recency: float,
    w_frequency: float,
    recency_half_life: float,
    frequency_log_base: float,
    max_frequency_bonus: float,
) -> np.ndarray:
    n = len(v_vec)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        age = max(now - v_ts[i], 0.0)
        recency = 2.0 ** (-age / recency_half_life)
        
        freq = min(np.log1p(v_ac[i]) / np.log(frequency_log_base), max_frequency_bonus)
        
        out[i] = (
            w_vector * v_vec[i]
            + w_keyword * v_kw[i]
            + w_importance * v_imp[i]
            + w_recency * recency
            + w_frequency * freq
        )
    return out


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
        self._lock = RWLock()
        self.working_memory = LRUCache(capacity=100)
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

        with WriteContext(self._lock):
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
                tokens=set(re.findall(r"\w+", content.lower())),
            )

            self.vector_index.add(vector)
            self._id_order.append(mem_id)
            self.kv.set(mem_id, memory)
            self.working_memory.clear()

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
        with WriteContext(self._lock):
            old.active = False
            old.superseded_by = updated.id

            self.vector_index.add(updated.vector)
            self._id_order.append(updated.id)
            self.kv.set(updated.id, updated)
            self.working_memory.clear()

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
        self.working_memory.clear()
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
        query_key = f"{query}:{top_k}:{namespace}:{include_inactive}"
        cached = self.working_memory.get(query_key)
        if cached is not None:
            return cached

        query_vec = self.embedder.encode(query)

        with ReadContext(self._lock):
            if self.kv.size == 0:
                return []
            scores, indices = self.vector_index.search(query_vec, top_k=min(top_k * 5, self.kv.size))
            id_snap = list(self._id_order)

        now = time.time()
        query_tokens = set(re.findall(r"\w+", query.lower()))
        q_len = max(len(query_tokens), 1)

        mems_to_score = []
        vec_scores = []
        importances = []
        timestamps = []
        access_counts = []
        keyword_scores = []

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

            mems_to_score.append(mem)
            vec_scores.append(float(vec_score))
            importances.append(mem.importance)
            timestamps.append(mem.timestamp)
            access_counts.append(mem.access_count)
            
            overlap = len(query_tokens & mem.tokens)
            keyword_scores.append(min(overlap / q_len, 1.0))

        if not mems_to_score:
            return []

        # Vectorized scoring (Numba JIT)
        v_vec = np.array(vec_scores, dtype=np.float32)
        v_imp = np.array(importances, dtype=np.float32)
        v_ts = np.array(timestamps, dtype=np.float32)
        v_ac = np.array(access_counts, dtype=np.float32)
        v_kw = np.array(keyword_scores, dtype=np.float32)

        combined_scores = fast_hybrid_score(
            v_vec, v_imp, v_ts, v_ac, v_kw, now,
            _W_VECTOR, _W_KEYWORD, _W_IMPORTANCE, _W_RECENCY, _W_FREQUENCY,
            _RECENCY_HALF_LIFE, _FREQUENCY_LOG_BASE, _MAX_FREQUENCY_BONUS
        )


        results: List[Memory] = []
        for i, mem in enumerate(mems_to_score):
            scored = Memory(
                id=mem.id, type=mem.type, content=mem.content,
                vector=mem.vector, timestamp=mem.timestamp, score=float(combined_scores[i]),
                metadata=mem.metadata, importance=mem.importance,
                access_count=mem.access_count, last_accessed=mem.last_accessed,
                namespace=mem.namespace, source=mem.source,
                superseded_by=mem.superseded_by, active=mem.active,
                level=mem.level, tokens=mem.tokens,
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

        self.working_memory.put(query_key, top_results)
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

        with ReadContext(self._lock):
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
            with WriteContext(self._lock):
                self._id_order.append(cm.id)
                self.kv.set(cm.id, cm)
                self.working_memory.clear()
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
            with WriteContext(self._lock):
                self._id_order.append(ref.id)
                self.kv.set(ref.id, ref)
                self.working_memory.clear()
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
            with WriteContext(self._lock):
                self._id_order.append(ref.id)
                self.kv.set(ref.id, ref)
                self.working_memory.clear()
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
            with WriteContext(self._lock):
                self.kv.clear()
                self.vector_index.reset()
                self.graph.clear()
                self._id_order.clear()
                self.working_memory.clear()

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
