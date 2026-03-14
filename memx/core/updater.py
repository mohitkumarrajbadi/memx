"""Memory update, merge, and conflict resolution engine.

Handles the lifecycle of memories beyond simple insert:
- Update: modify content of an existing memory
- Merge: combine semantically similar memories
- Conflict resolution: detect contradictions and resolve them
- Supersede: mark old memory as replaced by new one
"""

import time
import hashlib
import re
import logging
from typing import List, Optional, Tuple, Dict

import numpy as np

from ..types import Memory, MemoryType

logger = logging.getLogger(__name__)

# Similarity threshold for merge candidates
_MERGE_THRESHOLD = 0.85
_CONFLICT_KEYWORDS = {
    "not", "no longer", "stopped", "quit", "cancel", "changed",
    "switched", "instead", "unlike", "but", "however",
}


def detect_contradiction(old: Memory, new_content: str) -> bool:
    """Heuristic contradiction detection between old memory and new content.

    Returns True if new_content likely contradicts old memory.
    """
    new_lower = new_content.lower()
    old_lower = old.content.lower()

    # Check for negation-based contradictions
    for keyword in _CONFLICT_KEYWORDS:
        if keyword in new_lower:
            # Check if they share a common topic (overlapping nouns)
            old_words = set(re.findall(r"\b[a-z]{3,}\b", old_lower))
            new_words = set(re.findall(r"\b[a-z]{3,}\b", new_lower))
            overlap = old_words & new_words - _CONFLICT_KEYWORDS
            if len(overlap) >= 2:
                return True

    return False


def find_merge_candidates(
    target_vector: np.ndarray,
    memories: List[Memory],
    threshold: float = _MERGE_THRESHOLD,
    exclude_id: Optional[str] = None,
) -> List[Tuple[Memory, float]]:
    """Find memories semantically similar enough to merge with."""
    candidates = []
    target_norm = np.linalg.norm(target_vector)
    if target_norm == 0:
        return candidates

    for mem in memories:
        if not mem.active:
            continue
        if exclude_id and mem.id == exclude_id:
            continue
        mem_norm = np.linalg.norm(mem.vector)
        if mem_norm == 0:
            continue

        similarity = float(np.dot(target_vector, mem.vector) / (target_norm * mem_norm))
        if similarity >= threshold:
            candidates.append((mem, similarity))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def merge_memories(memories: List[Memory]) -> str:
    """Merge multiple memories into a single combined content string.

    Uses a simple deduplication + concatenation strategy.
    For LLM-powered summarization, use the reflection engine.
    """
    if not memories:
        return ""
    if len(memories) == 1:
        return memories[0].content

    # Deduplicate near-identical sentences
    seen_sentences = set()
    unique_parts = []

    for mem in memories:
        sentences = re.split(r"[.!?]+", mem.content)
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            # Normalize for comparison
            normalized = re.sub(r"\s+", " ", s.lower())
            if normalized not in seen_sentences:
                seen_sentences.add(normalized)
                unique_parts.append(s)

    return ". ".join(unique_parts)


def create_updated_memory(
    old: Memory,
    new_content: str,
    new_vector: np.ndarray,
    merge: bool = False,
) -> Memory:
    """Create a new Memory that supersedes an old one.

    If merge=True, combines old and new content.
    Otherwise, replaces with new content entirely.
    """
    if merge:
        combined = merge_memories([old, Memory(
            id="tmp", type=old.type, content=new_content,
            vector=new_vector, timestamp=time.time()
        )])
    else:
        combined = new_content

    new_id = hashlib.md5(combined.encode("utf-8")).hexdigest()[:12]

    return Memory(
        id=new_id,
        type=old.type,
        content=combined,
        vector=new_vector,
        timestamp=time.time(),
        importance=max(old.importance, 0.5),  # updates are at least medium importance
        access_count=old.access_count,
        last_accessed=old.last_accessed,
        namespace=old.namespace,
        source="update",
        metadata={**old.metadata, "supersedes": old.id},
        level=old.level,
        tokens=set(re.findall(r"\w+", combined.lower())),
    )
