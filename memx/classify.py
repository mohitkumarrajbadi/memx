"""Auto-classification of text into MemoryType using keyword heuristics."""

import re
from .types import MemoryType

# Signal-word patterns → MemoryType mapping (checked in priority order)
_PATTERNS = [
    (MemoryType.PROCEDURAL, [
        r"\bstep\s*\d", r"\bhow\s+to\b", r"\bprocedure\b", r"\brecipe\b",
        r"\binstructions?\b", r"\bfirst\b.*\bthen\b", r"\bworkflow\b",
    ]),
    (MemoryType.CAUSAL, [
        r"\bbecause\b", r"\bcaused?\b", r"\bresulted?\s+in\b", r"\bdue\s+to\b",
        r"\btherefore\b", r"\bconsequen", r"\beffect\b", r"\breason\b",
    ]),
    (MemoryType.DECISION, [
        r"\bdecided?\b", r"\bchose\b", r"\bchoos", r"\bselect",
        r"\bprefer", r"\boption\b", r"\balternative\b", r"\btrade-?off\b",
    ]),
    (MemoryType.EPISODIC, [
        r"\byesterday\b", r"\blast\s+(week|month|year)\b", r"\bremember\b",
        r"\bwent\s+to\b", r"\bhappened\b", r"\bexperienced?\b", r"\bvisited?\b",
    ]),
    (MemoryType.WORKING, [
        r"\bright\s+now\b", r"\bcurrently\b", r"\bat\s+the\s+moment\b",
        r"\bin\s+progress\b", r"\bactive(ly)?\b", r"\bongoing\b",
    ]),
    (MemoryType.ACTIVE, [
        r"\burgent\b", r"\bimportant\b", r"\bpriority\b", r"\basap\b",
        r"\bcritical\b", r"\bimmediate\b",
    ]),
    # Default fallback
    (MemoryType.SEMANTIC, []),
]


def auto_classify(content: str) -> MemoryType:
    """Classify *content* into one of the 7 MemoryTypes using keyword heuristics.

    Falls back to ``MemoryType.SEMANTIC`` when no signal words match.
    """
    text = content.lower()
    for mem_type, patterns in _PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text):
                return mem_type
    return MemoryType.SEMANTIC
