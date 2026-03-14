# MemX Developer Guide

> **Everything you need to build on, extend, and contribute to MemX.**

---

## Quick Setup

```bash
# Clone
git clone https://github.com/mohitbadi/memx-ai
cd memx-ai

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify
pytest tests/ -v
memx demo
```

After `pip install -e .`, the `memx` package is importable from **anywhere** on your machine:

```python
from memx import MemX
m = MemX()
```

---

## Architecture

```
memx/                          # Core library
├── __init__.py                # Public exports: MemX, MemoryType, Memory, RetrievalExplanation
├── types.py                   # Memory dataclass + MemoryType enum + RetrievalExplanation
├── classify.py                # Keyword-heuristic auto-classifier
├── api.py                     # MemX — the 1-class public API
├── cli.py                     # CLI: memx demo | benchmark | dashboard | inspect | stats
│
├── core/                      # Engine layer
│   ├── embeddings.py          # Embedding abstraction (sentence-transformers / hash fallback)
│   ├── vector.py              # Thread-safe FAISS IndexFlatIP wrapper
│   ├── kv.py                  # O(1) in-memory key-value cache
│   ├── graph.py               # Directed causal graph engine
│   ├── braintrace.py          # BrainTrace — the Memory OS engine (orchestrates everything)
│   ├── importance.py          # Importance scoring + recency decay + frequency bonus
│   ├── updater.py             # Memory update / merge / conflict resolution
│   ├── compression.py         # Semantic clustering + dedup merge (optional LLM summarizer)
│   ├── reflection.py          # Auto-reflection: memory-batch + conversation insights
│   ├── inspector.py           # Retrieval explainer (score breakdowns)
│   └── distributed.py         # ShardRouter + capacity planner (future distributed tier)
│
├── backends/                  # Storage backends
│   ├── base.py                # Abstract interface
│   └── sqlite_backend.py      # SQLite (WAL mode, :memory: or file)
│
├── integrations/              # Framework drop-ins
│   ├── langchain.py           # MemXChatMemory + MemXRetriever
│   └── crewai.py              # MemXSharedMemory (multi-agent namespaces)
│
└── viz/                       # Visualization
    └── server.py              # Zero-dep web dashboard (http.server)

tests/                         # 73 tests
├── test_types.py              # MemoryType, Memory dataclass
├── test_backends.py           # SQLite backend CRUD
├── test_api.py                # MemX public API
├── test_cli.py                # CLI commands
└── test_memory_os.py          # Memory OS features: importance, compression, reflection, etc.

benchmarks/                    # Performance suite
├── latency.py                 # p50/p95/p99 profiler
├── scalability.py             # Dataset size scaling
├── concurrency.py             # Thread safety + throughput
├── memory_profile.py          # RSS + per-object memory
├── integrity.py               # Data loss / dedup / recall
├── competitor.py              # vs ChromaDB / Mem0 / LanceDB
├── scale.py                   # FAISS 10K→1M + E2E latency breakdown
└── suite.py                   # Unified runner with JSON export

examples/
├── quickstart.py
├── bharatsearch.py            # Farmer agent demo
├── langchain_memx.py          # LangChain integration example
└── memory_assistant.py        # 8-scene killer demo
```

---

## How Things Wire Together

```
User Code
    │
    ▼
MemX (api.py)                    ← public interface, delegates everything
    │
    ▼
BrainTrace (braintrace.py)       ← the orchestrator / Memory OS engine
    │
    ├── Embedder (embeddings.py) ← text → 384-dim vector
    ├── VectorIndex (vector.py)  ← FAISS IndexFlatIP (thread-safe)
    ├── KVCache (kv.py)          ← mem_id → Memory object
    ├── CausalGraph (graph.py)   ← directed edges between memories
    │
    ├── importance.py            ← auto-scores importance, computes decay
    ├── updater.py               ← update / merge / contradiction detection
    ├── compression.py           ← cluster + merge similar memories
    ├── reflection.py            ← generate insights from memory clusters
    └── inspector.py             ← explain retrieval scores
```

**Data flow for `m.add("I prefer Python")`:**

1. `api.py` calls `brain.add(content)`
2. `braintrace.py`:
   - `classify.py` auto-classifies → `DECISION`
   - `importance.py` auto-scores → `0.85` (preference signal)
   - `embeddings.py` encodes → 384-dim vector
   - `vector.py` indexes vector in FAISS
   - `kv.py` stores `Memory` object
   - `sqlite_backend.py` persists to disk

**Data flow for `m.rag("programming language")`:**

1. `api.py` calls `brain.rag(query)`
2. `braintrace.py`:
   - `embeddings.py` encodes query → vector
   - `vector.py` searches FAISS → top candidate indices
   - For each candidate:
     - Keyword overlap score
     - `importance.py` recency decay + frequency bonus
     - Composite: `0.35*vector + 0.15*keyword + 0.25*importance + 0.15*recency + 0.10*frequency`
   - Sort by composite score
   - Update `access_count` and `last_accessed` on retrieved memories
   - Return top-k

---

## Key Concepts

### Memory Lifecycle

```
add() → active memory
  │
  ├── rag() hits → access_count++, last_accessed updated
  │
  ├── update() → old deactivated, new memory created
  │     old.active = False, old.superseded_by = new.id
  │
  ├── compress() → cluster merged, originals deactivated
  │
  ├── decay() → low recency × importance → deactivated
  │
  └── delete() → soft-delete (active = False)
```

### Importance Scoring

| Signal | Weight | Description |
|--------|--------|-------------|
| Base importance | 0.50 | Auto-estimated or user-provided (0.0–1.0) |
| Recency | 0.30 | Exponential decay, 24h half-life |
| Frequency | 0.20 | Log(1 + access_count), capped at 1.0 |

Auto-estimation heuristics:
- **0.9**: Name, identity (`"My name is..."`)
- **0.85**: Preferences, favorites (`"I prefer..."`)
- **0.8**: PII, secrets (`"password"`, `"api key"`)
- **0.3**: Filler (`"okay"`, `"sure"`, `"thanks"`)
- **0.5**: Default for medium-length content

### Namespaces

Memories are isolated by namespace string. Multi-agent systems use `workspace/agent` namespacing:

```python
m.add("finding", namespace="project/researcher")
m.add("strategy", namespace="project/planner")

# Agent-specific retrieval
m.rag("query", namespace="project/researcher")

# Cross-namespace retrieval (omit namespace)
m.rag("query")  # searches all
```

### Compression Pipeline

```
active memories  →  cosine similarity clustering (threshold=0.75)
                          ↓
              clusters ≥ 2 members
                          ↓
              sentence-level dedup merge (or LLM summarizer)
                          ↓
              new compressed memory (source="compression")
              originals: active=False, superseded_by=new_id
```

### Reflection Pipeline

```
active memories  →  looser clustering (threshold=0.65)
                          ↓
              clusters ≥ 3 members
                          ↓
              template-based summarization (or LLM)
                          ↓
              new REFLECTION memory (importance=0.8)
```

---

## Extending MemX

### Adding a New Backend

1. Create `memx/backends/your_backend.py`
2. Implement the `Backend` interface from `base.py`:

```python
from memx.backends.base import Backend

class YourBackend(Backend):
    def save(self, memory): ...
    def load(self, memory_id): ...
    def delete(self, memory_id): ...
    def search(self, query, limit=10): ...
    def all(self): ...
    def count(self): ...
    def clear(self): ...
```

3. Register in `api.py`'s `__init__`:

```python
elif backend == "yours":
    self._backend = YourBackend(...)
```

### Adding a New Memory Type

1. Add to the `MemoryType` enum in `types.py`:
```python
class MemoryType(Enum):
    ...
    YOUR_TYPE = 8
```

2. Add classification rules in `classify.py`:
```python
if re.search(r"your_pattern", text):
    return MemoryType.YOUR_TYPE
```

3. Update test assertions in `test_types.py`.

### Adding a New Integration

1. Create `memx/integrations/your_framework.py`
2. Import and wrap `MemX`:

```python
from memx import MemX

class YourFrameworkMemory:
    def __init__(self):
        self.memx = MemX()
    # Implement framework-specific interface
```

### Custom LLM-Powered Compression/Reflection

Pass a summarizer function to `compress()` or `reflect()`:

```python
def my_llm_summarizer(texts: list[str]) -> str:
    prompt = "Summarize these memories: " + "\n".join(texts)
    return call_my_llm(prompt)

m.compress(summarizer=my_llm_summarizer)
m.reflect(summarizer=my_llm_summarizer)
```

---

## Running Tests

```bash
# All tests (73)
pytest tests/ -v

# Specific module
pytest tests/test_memory_os.py -v

# With coverage
pytest tests/ --cov=memx --cov-report=term-missing
```

## Running Benchmarks

```bash
# Quick benchmark
memx benchmark --n 10000

# Full production suite
python3 -m benchmarks.suite --all --output report.json

# Large-scale FAISS (10K → 1M)
python3 -m benchmarks.scale --mode both --max-n 1000000

# Individual modules
python3 -m benchmarks.latency
python3 -m benchmarks.concurrency
python3 -m benchmarks.integrity
```

## Using MemX in Other Projects

After `pip install -e .` in the memx directory, use it anywhere:

```python
# In any Python project on your machine
from memx import MemX, MemoryType

m = MemX(db_path="./my_project_memory.db")  # persistent

# Store memories
m.add("User prefers dark mode", importance=0.8)
m.add("API key is stored in env vars", importance=0.95)

# Retrieve
results = m.rag("user preferences")

# Inspect why
for exp in m.inspect("user preferences"):
    print(exp.explain())

# Multi-agent
from memx.integrations.crewai import MemXSharedMemory
shared = MemXSharedMemory(workspace="my-app")
shared.store("agent-1", "Discovered important pattern")

# Web dashboard
# memx dashboard
```

---

## Thread Safety

`VectorIndex` and `BrainTrace` use `threading.Lock` to guard shared state. Safe for:
- Multi-threaded web servers
- Concurrent agent workloads
- Background compression/reflection tasks

**Not safe for**: Multi-process (use separate instances or a shared backend like PostgreSQL).

---

## Performance Characteristics

| Scale | Add | RAG p50 | RAG p99 | FAISS Index | RAM |
|------:|----:|--------:|--------:|------------:|----:|
| 1K | 0.04ms | 0.03ms | 0.05ms | 1.5 MB | ~0 |
| 10K | 0.04ms | 0.28ms | 0.32ms | 15 MB | ~50 MB |
| 100K | 0.04ms | 2.90ms | 3.55ms | 147 MB | ~430 MB |
| 1M | 0.04ms | 29.5ms | 79ms | 1.4 GB | ~2.4 GB |

Embedding cost with hash fallback: ~0.03ms (constant).
With sentence-transformers: ~5–20ms per query.

---

## Versioning

- **v0.1.0**: Core library (7 memory types, hybrid RAG, SQLite backend)
- **v0.2.0**: Memory OS upgrade (importance, compression, reflection, namespaces, integrations, dashboard)
