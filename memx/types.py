"""Memory type definitions and core data structures."""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np


class MemoryType(Enum):
    """Seven fundamental memory types inspired by cognitive neuroscience."""

    WORKING = 0       # Short-term, active manipulation
    EPISODIC = 1      # Personal events / experiences
    SEMANTIC = 2      # Facts, general knowledge
    CAUSAL = 3        # Cause-effect relationships
    DECISION = 4      # Choices and their outcomes
    PROCEDURAL = 5    # Step-by-step procedures / how-to
    ACTIVE = 6        # Currently relevant, high-priority
    REFLECTION = 7    # Auto-generated insights from reflection engine


@dataclass
class Memory:
    """A single memory record with importance, decay, and namespace support."""

    id: str
    type: MemoryType
    content: str
    vector: np.ndarray
    timestamp: float
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Memory OS fields ──
    importance: float = 0.5         # 0.0 = trivial, 1.0 = critical
    access_count: int = 0           # times retrieved via rag()
    last_accessed: float = 0.0      # last rag() hit timestamp
    namespace: str = "default"      # multi-agent namespace
    source: str = ""                # origin: "user", "agent", "reflection", "compression"
    superseded_by: Optional[str] = None   # ID of memory that replaced this one
    active: bool = True             # False = soft-deleted / decayed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.name,
            "content": self.content,
            "timestamp": self.timestamp,
            "score": self.score,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "namespace": self.namespace,
            "source": self.source,
            "active": self.active,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        preview = self.content[:60] + "..." if len(self.content) > 60 else self.content
        return f"Memory({self.type.name}, score={self.score:.3f}, imp={self.importance:.2f}, '{preview}')"


@dataclass
class RetrievalExplanation:
    """Breakdown of why a memory was retrieved — for observability."""
    memory_id: str
    final_score: float
    vector_score: float
    keyword_score: float
    recency_score: float
    importance_score: float
    frequency_bonus: float
    query: str
    matched_keywords: List[str] = field(default_factory=list)

    def explain(self) -> str:
        lines = [
            f"Memory {self.memory_id} — score {self.final_score:.4f}",
            f"  vector similarity:  {self.vector_score:.4f}",
            f"  keyword match:      {self.keyword_score:.4f}  {self.matched_keywords}",
            f"  recency:            {self.recency_score:.4f}",
            f"  importance:         {self.importance_score:.4f}",
            f"  frequency bonus:    {self.frequency_bonus:.4f}",
        ]
        return "\n".join(lines)
