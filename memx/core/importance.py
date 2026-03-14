"""Importance scoring and memory decay engine.

Computes a composite memory importance score from:
- Base importance (user-assigned or auto-estimated)
- Recency decay (exponential half-life)
- Access frequency (how often retrieved)

Also handles automatic decay: memories below a threshold
are marked inactive and excluded from retrieval.
"""

import time
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Dict

from ..types import Memory, MemoryType

# ── Scoring weights ──
_W_IMPORTANCE = 0.50
_W_RECENCY = 0.30
_W_FREQUENCY = 0.20

# ── Decay config ──
_RECENCY_HALF_LIFE = 3600.0 * 24  # 24 hours
_DECAY_THRESHOLD = 0.05           # memories below this are deactivated
_MAX_FREQUENCY_BONUS = 1.0        # cap on frequency contribution
_FREQUENCY_LOG_BASE = 10          # log base for diminishing returns


# ── Importance auto-estimation heuristics ──
_HIGH_IMPORTANCE_SIGNALS = [
    (0.9, [r"\bname\s+is\b", r"\bi\s+am\b", r"\bmy\s+name\b"]),          # identity
    (0.85, [r"\bprefer", r"\bfavorite\b", r"\balways\b", r"\bnever\b"]),   # preferences
    (0.8, [r"\bpassword\b", r"\bapi.?key\b", r"\bsecret\b", r"\btoken\b"]), # secrets
    (0.8, [r"\bborn\b", r"\bbirthday\b", r"\baddress\b", r"\bphone\b"]),   # PII
    (0.75, [r"\bdecided\b", r"\bchose\b", r"\bcommit"]),                   # decisions
    (0.7, [r"\bgoal\b", r"\bobjective\b", r"\bmission\b", r"\btarget\b"]), # goals
]

_LOW_IMPORTANCE_SIGNALS = [
    (0.2, [r"\bokay\b", r"\bsure\b", r"\bgot\s+it\b", r"\bthanks?\b"]),  # filler
    (0.25, [r"\btest\b.*\btest\b", r"\basdf\b", r"\bfoo\b.*\bbar\b"]),    # test data
    (0.3, [r"\bhi\b$", r"\bhello\b$", r"\bhey\b$"]),                       # greetings
]


def estimate_importance(content: str) -> float:
    """Auto-estimate importance of content from 0.0 to 1.0."""
    text = content.lower().strip()

    # Check high-importance signals (return on first match)
    for score, patterns in _HIGH_IMPORTANCE_SIGNALS:
        for p in patterns:
            if re.search(p, text):
                return score

    # Check low-importance signals
    for score, patterns in _LOW_IMPORTANCE_SIGNALS:
        for p in patterns:
            if re.search(p, text):
                return score

    # Length-based heuristic: longer = usually more informative
    word_count = len(text.split())
    if word_count < 3:
        return 0.3
    elif word_count < 10:
        return 0.5
    elif word_count < 30:
        return 0.6
    else:
        return 0.7


def compute_recency_score(timestamp: float, now: Optional[float] = None, half_life: float = _RECENCY_HALF_LIFE) -> float:
    """Exponential recency decay: 1.0 when fresh, halves every half_life seconds."""
    now = now or time.time()
    age = max(now - timestamp, 0.0)
    return 2.0 ** (-age / half_life)


def compute_frequency_score(access_count: int) -> float:
    """Logarithmic frequency bonus with diminishing returns."""
    if access_count <= 0:
        return 0.0
    return min(math.log(1 + access_count, _FREQUENCY_LOG_BASE), _MAX_FREQUENCY_BONUS)


def compute_composite_score(
    memory: Memory,
    vector_score: float = 0.0,
    keyword_score: float = 0.0,
    now: Optional[float] = None,
) -> float:
    """Compute the full composite retrieval score."""
    recency = compute_recency_score(memory.timestamp, now)
    frequency = compute_frequency_score(memory.access_count)

    # Base retrieval signal (vector + keyword)
    retrieval_signal = 0.55 * vector_score + 0.20 * keyword_score

    # Memory quality signal (importance + recency + frequency)
    quality_signal = (
        _W_IMPORTANCE * memory.importance
        + _W_RECENCY * recency
        + _W_FREQUENCY * frequency
    )

    return 0.55 * retrieval_signal + 0.45 * quality_signal


def should_decay(memory: Memory, now: Optional[float] = None) -> bool:
    """Return True if memory's recency score has dropped below threshold."""
    recency = compute_recency_score(memory.timestamp, now)
    effective = recency * memory.importance
    return effective < _DECAY_THRESHOLD


def run_decay_sweep(memories: List[Memory], now: Optional[float] = None) -> List[str]:
    """Mark memories as inactive if they've decayed. Returns list of deactivated IDs."""
    deactivated = []
    for mem in memories:
        if mem.active and should_decay(mem, now):
            mem.active = False
            deactivated.append(mem.id)
    return deactivated
