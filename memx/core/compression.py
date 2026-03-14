"""Memory compression engine.

Reduces memory bloat by:
1. Clustering semantically similar memories
2. Merging clusters into single compressed memories
3. Deactivating redundant originals

Works without an LLM via sentence-dedup + clustering.
When an LLM summarizer is provided, uses it for higher-quality compression.
"""

import time
import hashlib
import logging
from typing import List, Optional, Callable, Dict, Tuple

import numpy as np

from ..types import Memory, MemoryType
from .updater import merge_memories

logger = logging.getLogger(__name__)

# Default cosine similarity threshold for grouping
_CLUSTER_THRESHOLD = 0.75


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cluster_memories(
    memories: List[Memory],
    threshold: float = _CLUSTER_THRESHOLD,
) -> List[List[Memory]]:
    """Group memories into clusters by cosine similarity (greedy single-link).

    Returns a list of clusters, where each cluster is a list of Memory objects.
    """
    if not memories:
        return []

    used = [False] * len(memories)
    clusters: List[List[Memory]] = []

    for i, mem_i in enumerate(memories):
        if used[i] or not mem_i.active:
            continue
        cluster = [mem_i]
        used[i] = True

        for j in range(i + 1, len(memories)):
            if used[j] or not memories[j].active:
                continue
            sim = _cosine_similarity(mem_i.vector, memories[j].vector)
            if sim >= threshold:
                cluster.append(memories[j])
                used[j] = True

        if len(cluster) >= 2:
            clusters.append(cluster)

    return clusters


def compress_cluster(
    cluster: List[Memory],
    summarizer: Optional[Callable[[List[str]], str]] = None,
) -> Memory:
    """Compress a cluster of similar memories into one.

    Args:
        cluster: Memories to compress.
        summarizer: Optional LLM function ``f(texts) -> summary``.
                    If None, uses sentence-dedup merge.
    """
    if len(cluster) == 1:
        return cluster[0]

    contents = [m.content for m in cluster]

    if summarizer:
        compressed_text = summarizer(contents)
    else:
        compressed_text = merge_memories(cluster)

    # Average the vectors
    avg_vector = np.mean([m.vector for m in cluster], axis=0).astype(np.float32)
    norm = np.linalg.norm(avg_vector)
    if norm > 0:
        avg_vector = avg_vector / norm

    # Take max importance, latest timestamp
    max_importance = max(m.importance for m in cluster)
    latest_ts = max(m.timestamp for m in cluster)
    total_access = sum(m.access_count for m in cluster)
    namespace = cluster[0].namespace

    new_id = hashlib.md5(compressed_text.encode("utf-8")).hexdigest()[:12]

    return Memory(
        id=new_id,
        type=MemoryType.SEMANTIC,
        content=compressed_text,
        vector=avg_vector,
        timestamp=latest_ts,
        importance=min(max_importance + 0.1, 1.0),  # compression boosts importance
        access_count=total_access,
        namespace=namespace,
        source="compression",
        metadata={
            "compressed_from": [m.id for m in cluster],
            "original_count": len(cluster),
        },
    )


def run_compression(
    memories: List[Memory],
    threshold: float = _CLUSTER_THRESHOLD,
    min_cluster_size: int = 2,
    summarizer: Optional[Callable[[List[str]], str]] = None,
) -> Tuple[List[Memory], List[str]]:
    """Run full compression pipeline.

    Returns:
        - List of new compressed memories
        - List of IDs that were deactivated (originals in clusters)
    """
    active = [m for m in memories if m.active]
    clusters = cluster_memories(active, threshold)

    # Filter to clusters meeting min size
    clusters = [c for c in clusters if len(c) >= min_cluster_size]

    compressed: List[Memory] = []
    deactivated_ids: List[str] = []

    for cluster in clusters:
        new_mem = compress_cluster(cluster, summarizer)
        compressed.append(new_mem)

        # Mark originals as superseded
        for mem in cluster:
            mem.active = False
            mem.superseded_by = new_mem.id
            deactivated_ids.append(mem.id)

    logger.info(
        "Compression: %d clusters → %d compressed memories, %d deactivated",
        len(clusters), len(compressed), len(deactivated_ids),
    )

    return compressed, deactivated_ids
