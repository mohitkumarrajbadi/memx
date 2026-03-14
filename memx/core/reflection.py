"""Automatic reflection engine.

After a batch of memories is accumulated, the reflection engine
generates higher-order insights by summarizing clusters of related
memories into structured reflection memories.

Works without an LLM via template-based summarization.
When an LLM function is provided, produces much richer reflections.
"""

import time
import hashlib
import logging
from typing import List, Optional, Callable, Dict

import numpy as np

from ..types import Memory, MemoryType
from .compression import cluster_memories

logger = logging.getLogger(__name__)

# Minimum memories needed to trigger reflection
_MIN_MEMORIES_FOR_REFLECTION = 3
_REFLECTION_CLUSTER_THRESHOLD = 0.65  # looser than compression


def _template_reflect(memories: List[Memory]) -> str:
    """Generate a reflection from memories using templates (no LLM needed)."""
    if not memories:
        return ""

    # Group by type
    by_type: Dict[str, List[str]] = {}
    for m in memories:
        by_type.setdefault(m.type.name, []).append(m.content)

    parts = []

    # Summarize each type group
    if "DECISION" in by_type:
        decisions = by_type["DECISION"]
        parts.append(f"Key decisions: {'; '.join(d[:80] for d in decisions[:3])}")

    if "EPISODIC" in by_type:
        events = by_type["EPISODIC"]
        parts.append(f"Recent events: {'; '.join(e[:80] for e in events[:3])}")

    if "SEMANTIC" in by_type:
        facts = by_type["SEMANTIC"]
        parts.append(f"Known facts: {'; '.join(f[:80] for f in facts[:3])}")

    if "CAUSAL" in by_type:
        causes = by_type["CAUSAL"]
        parts.append(f"Cause-effect patterns: {'; '.join(c[:80] for c in causes[:3])}")

    if "PROCEDURAL" in by_type:
        procs = by_type["PROCEDURAL"]
        parts.append(f"Known procedures: {'; '.join(p[:80] for p in procs[:3])}")

    if "WORKING" in by_type or "ACTIVE" in by_type:
        current = by_type.get("WORKING", []) + by_type.get("ACTIVE", [])
        parts.append(f"Current focus: {'; '.join(c[:80] for c in current[:3])}")

    if not parts:
        contents = [m.content[:80] for m in memories[:5]]
        parts.append(f"Summary of {len(memories)} related memories: {'; '.join(contents)}")

    return ". ".join(parts)


def reflect_on_memories(
    memories: List[Memory],
    embedder=None,
    summarizer: Optional[Callable[[List[str]], str]] = None,
    threshold: float = _REFLECTION_CLUSTER_THRESHOLD,
) -> List[Memory]:
    """Generate reflection memories from a set of memories.

    Args:
        memories: Source memories to reflect on.
        embedder: Embedder instance for creating vectors.
        summarizer: Optional LLM function ``f(texts) -> summary``.
        threshold: Clustering threshold.

    Returns:
        List of new REFLECTION-type memories.
    """
    active = [m for m in memories if m.active]
    if len(active) < _MIN_MEMORIES_FOR_REFLECTION:
        return []

    # Cluster related memories
    clusters = cluster_memories(active, threshold)

    reflections: List[Memory] = []

    for cluster in clusters:
        if len(cluster) < _MIN_MEMORIES_FOR_REFLECTION:
            continue

        contents = [m.content for m in cluster]

        if summarizer:
            reflection_text = summarizer(contents)
        else:
            reflection_text = _template_reflect(cluster)

        if not reflection_text.strip():
            continue

        # Create embedding for reflection
        if embedder:
            vector = embedder.encode(reflection_text)
        else:
            # Average cluster vectors
            avg = np.mean([m.vector for m in cluster], axis=0).astype(np.float32)
            norm = np.linalg.norm(avg)
            vector = avg / norm if norm > 0 else avg

        ref_id = hashlib.md5(reflection_text.encode("utf-8")).hexdigest()[:12]

        reflection = Memory(
            id=ref_id,
            type=MemoryType.REFLECTION,
            content=reflection_text,
            vector=vector,
            timestamp=time.time(),
            importance=0.8,  # reflections are high-value
            namespace=cluster[0].namespace,
            source="reflection",
            metadata={
                "reflected_from": [m.id for m in cluster],
                "source_count": len(cluster),
            },
        )
        reflections.append(reflection)

    logger.info("Reflection: generated %d insights from %d memories", len(reflections), len(active))
    return reflections


def reflect_on_conversation(
    messages: List[str],
    embedder=None,
    summarizer: Optional[Callable[[List[str]], str]] = None,
) -> Optional[Memory]:
    """Reflect on a conversation (list of message strings) to produce a single insight.

    This is the key feature for production agents: after a conversation ends,
    call this to create a structured memory of what was discussed.
    """
    if not messages or len(messages) < 2:
        return None

    if summarizer:
        reflection_text = summarizer(messages)
    else:
        # Template-based: extract topics
        all_text = " ".join(messages)
        topics = set()
        for msg in messages:
            words = msg.lower().split()
            # Extract multi-word phrases (crude noun extraction)
            for i in range(len(words) - 1):
                if len(words[i]) > 3 and len(words[i + 1]) > 3:
                    topics.add(f"{words[i]} {words[i + 1]}")

        topic_list = list(topics)[:5]
        reflection_text = f"Conversation covered {len(messages)} messages about: {', '.join(topic_list) if topic_list else 'various topics'}"

    if embedder:
        vector = embedder.encode(reflection_text)
    else:
        vector = np.zeros(384, dtype=np.float32)

    ref_id = hashlib.md5(reflection_text.encode("utf-8")).hexdigest()[:12]

    return Memory(
        id=ref_id,
        type=MemoryType.REFLECTION,
        content=reflection_text,
        vector=vector,
        timestamp=time.time(),
        importance=0.75,
        source="reflection",
        metadata={"message_count": len(messages)},
    )
