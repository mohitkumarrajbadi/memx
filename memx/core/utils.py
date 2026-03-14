"""Concurrency and caching utilities for MemX."""

import threading
from collections import OrderedDict
from typing import Any, Optional


class RWLock:
    """A standard Reader-Writer lock.
    Allows multiple concurrent readers, but mutually exclusive writers.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._read_ready = threading.Condition(self._lock)
        self._readers = 0
        self._writers = 0

    def acquire_read(self):
        self._lock.acquire()
        while self._writers > 0:
            self._read_ready.wait()
        self._readers += 1
        self._lock.release()

    def release_read(self):
        self._lock.acquire()
        self._readers -= 1
        if self._readers == 0:
            self._read_ready.notify_all()
        self._lock.release()

    def acquire_write(self):
        self._lock.acquire()
        while self._writers > 0 or self._readers > 0:
            self._read_ready.wait()
        self._writers += 1
        self._lock.release()

    def release_write(self):
        self._lock.acquire()
        self._writers -= 1
        self._read_ready.notify_all()
        self._lock.release()


class ReadContext:
    def __init__(self, lock: RWLock):
        self.lock = lock
    def __enter__(self):
        self.lock.acquire_read()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release_read()


class WriteContext:
    def __init__(self, lock: RWLock):
        self.lock = lock
    def __enter__(self):
        self.lock.acquire_write()
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lock.release_write()


class LRUCache:
    """Thread-safe LRU cache used as a Working Memory Buffer."""
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            if len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
                
    def invalidate(self, key: str) -> None:
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
