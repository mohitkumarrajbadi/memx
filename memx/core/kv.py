"""In-memory key-value cache for O(1) memory lookups."""

from typing import Optional, Dict, List
from ..types import Memory


class KVCache:
    """Simple dict-backed KV store for ``Memory`` objects."""

    def __init__(self) -> None:
        self._store: Dict[str, Memory] = {}

    @property
    def size(self) -> int:
        return len(self._store)

    def get(self, key: str) -> Optional[Memory]:
        return self._store.get(key)

    def set(self, key: str, memory: Memory) -> None:
        self._store[key] = memory

    def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def all(self) -> List[Memory]:
        return list(self._store.values())

    def clear(self) -> None:
        self._store.clear()

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __len__(self) -> int:
        return self.size
