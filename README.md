<p align="center">
  <h1 align="center">🧠 MemX</h1>
  <p align="center"><strong>The AI Memory Operating System</strong></p>
  <p align="center">
    8 memory types · Importance scoring · Compression · Reflection · Multi-agent · One API
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/memx-ai/"><img src="https://img.shields.io/pypi/v/memx-ai?color=blue" alt="PyPI"></a>
  <a href="https://github.com/mohitbadi/memx-ai/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python"></a>
</p>

---

> **MemX is not a vector DB. It's the operating system for AI memory.**

```python
from memx import MemX

m = MemX()
m.add("User prefers Python", importance=0.9)
m.add("User prefers Python a lot")
m.add("Python is user's favorite language")

m.compress()  # Merges 3 memories → 1: "User strongly prefers Python"
m.rag("programming language")  # importance-weighted hybrid retrieval
m.inspect("programming")  # explains why results scored
```

---

## ✨ What Makes MemX Different

| Feature | MemX | Mem0 | ChromaDB | Pinecone |
|---------|:----:|:----:|:--------:|:--------:|
| Memory Types | 8 cognitive | 1 | 0 | 0 |
| Importance Scoring | ✅ Auto | ❌ | ❌ | ❌ |
| Memory Compression | ✅ Built-in | ❌ | ❌ | ❌ |
| Auto Reflection | ✅ Built-in | ❌ | ❌ | ❌ |
| Memory Updates | ✅ Merge/Replace | ❌ | ❌ | Overwrite |
| Multi-Agent Namespaces | ✅ | ❌ | Collections | ❌ |
| Retrieval Explainer | ✅ `inspect()` | ❌ | ❌ | ❌ |
| Decay & Expiry | ✅ | ❌ | ❌ | TTL |
| Zero Config | ✅ `MemX()` | ❌ API key | ❌ | ❌ Cloud |
| Web Dashboard | ✅ `memx dashboard` | ❌ | ❌ | Console |

---

## 🚀 Quick Start

### Install

**Option 1: From GitHub (Latest Features)**
```bash
git clone https://github.com/mohitbadi/memx-ai
cd memx-ai
pip install -e .                       # Core installation
# pip install -e ".[all]"              # To include LLM embeddings & integrations
```

**Option 2: From PyPI**
```bash
pip install memx-ai                    # Core (FAISS + NumPy)
pip install "memx-ai[embeddings]"      # + sentence-transformers
pip install "memx-ai[all]"             # Everything
```

### 5-Line Memory OS

```python
from memx import MemX

m = MemX()
m.add("User is a software engineer from Bangalore")  # auto-classifies + auto-importance
m.add("User prefers Python for backend work")

results = m.rag("What does the user do?")
for r in results:
    print(f"[{r.type.name}] imp={r.importance:.2f} │ {r.content}")
```

### CLI

```bash
memx demo               # Interactive demo
memx benchmark --n 1000 # Performance benchmark
memx dashboard           # Launch web UI
memx inspect "query"     # Explain retrieval scoring
memx stats               # Memory system stats
```

---

## 🧬 The Memory OS API

```python
from memx import MemX

m = MemX()

# ── STORE ──
m.add("I like Python", importance=0.9, namespace="agent-1")
m.add("Temporary note")
m.update(old_id, "I switched to Rust")    # replaces, deactivates old
m.update(old_id, "Also like Go", merge=True)  # merges old + new
m.delete(mid)                              # soft-delete

# ── RETRIEVE ──
m.rag("programming", top_k=5, namespace="agent-1")  # importance-weighted RAG
m.get(mid)                                            # by ID
m.all(namespace="agent-1")                            # list all

# ── INTELLIGENCE ──
m.compress()                            # merge similar memories
m.reflect()                             # generate insights
m.reflect_conversation(messages)        # summarize a conversation
m.decay()                               # expire unimportant memories

# ── OBSERVABILITY ──
for exp in m.inspect("Python"):
    print(exp.explain())
    # vector_similarity: 0.87
    # keyword_match:     0.14  ['python']
    # importance:        0.90
    # recency:           1.00
    # frequency:         0.30

# ── MULTI-AGENT ──
from memx.integrations.crewai import MemXSharedMemory
shared = MemXSharedMemory(workspace="project")
shared.store("researcher", "Market growing 40% YoY")
shared.recall("market", agent="researcher")

# ── LANGCHAIN ──
from memx.integrations.langchain import MemXChatMemory
memory = MemXChatMemory(namespace="chat-1")
```

---

## 🏗️ Architecture

```
Agent / Your Code
    │
    ▼
MemX API (api.py)                ← 1-class interface
    │
    ▼
BrainTrace (braintrace.py)       ← Memory OS engine
    │
    ├── Importance Engine        ← auto-scoring + decay
    ├── Compression Engine       ← cluster + merge
    ├── Reflection Engine        ← auto-insights
    ├── Update Engine            ← merge + conflicts
    ├── Inspector                ← explain retrieval
    │
    ├── Embedder                 ← sentence-transformers / hash fallback
    ├── VectorIndex (FAISS)      ← thread-safe similarity search
    ├── KV Cache                 ← O(1) memory lookup
    └── Causal Graph             ← directed edges
                │
         Backend Layer
    SQLite │ PostgreSQL │ Redis
```

---

## 📊 Benchmarks (Verified)

| Scale | RAG p50 | RAG p99 | ops/s | FAISS Index |
|------:|--------:|--------:|------:|------------:|
| 10K | 0.28ms | 0.33ms | 3,421 | 15 MB |
| 100K | 2.90ms | 3.55ms | 340 | 147 MB |
| 500K | 15.1ms | 28.7ms | 62 | 732 MB |
| **1M** | **29.5ms** | **79ms** | **31** | **1.4 GB** |

**Scaling: O(n^1.05) — near-perfect linear** ✅

```bash
# Run yourself
python3 -m benchmarks.scale --mode both --max-n 1000000
```

---

## 📁 Project Structure

```
memx-ai/
├── memx/                          # Core library
│   ├── api.py                     # MemX public API
│   ├── types.py                   # Memory + MemoryType + RetrievalExplanation
│   ├── classify.py                # Auto-classification
│   ├── cli.py                     # CLI tools
│   ├── core/                      # Engine layer
│   │   ├── braintrace.py          # Memory OS engine
│   │   ├── importance.py          # Importance + decay
│   │   ├── compression.py         # Memory compression
│   │   ├── reflection.py          # Auto-reflection
│   │   ├── updater.py             # Update / merge
│   │   ├── inspector.py           # Retrieval explainer
│   │   ├── vector.py              # FAISS wrapper
│   │   ├── embeddings.py          # Embedder
│   │   ├── kv.py                  # KV cache
│   │   ├── graph.py               # Causal graph
│   │   └── distributed.py         # Sharding (future)
│   ├── backends/                  # Storage
│   │   └── sqlite_backend.py
│   ├── integrations/              # Framework drop-ins
│   │   ├── langchain.py
│   │   └── crewai.py
│   └── viz/                       # Web dashboard
│       └── server.py
├── tests/                         # 73 tests
├── benchmarks/                    # 7 benchmark modules
├── examples/                      # Demos
├── DEVELOPER.md                   # Developer guide
├── pyproject.toml                 # Package config
└── README.md
```

---

## 🗺️ Roadmap

- [x] Core BrainTrace engine
- [x] 8 memory types + auto-classification
- [x] Importance scoring + decay
- [x] Memory compression engine
- [x] Automatic reflection
- [x] Memory update / merge / conflicts
- [x] Multi-agent namespaces
- [x] Retrieval inspector
- [x] LangChain + CrewAI integrations
- [x] Web dashboard
- [x] FAISS 1M-scale benchmarks
- [ ] Redis backend
- [ ] PostgreSQL backend
- [ ] Distributed sharding
- [ ] Ollama / OpenAI embeddings
- [ ] REST API server

---

## 🤝 Contributing

See [DEVELOPER.md](./DEVELOPER.md) for the full developer guide.

```bash
git clone https://github.com/mohitbadi/memx-ai
cd memx-ai
pip install -e ".[dev]"
pytest tests/ -v   # 73 tests, 0.33s
```

---

## 📄 License

MIT — see [LICENSE](./LICENSE)

---

<p align="center">
  <strong>Built with ❤️ for the AI developer community</strong><br>
  <sub>⭐ Star this repo if MemX saves you time!</sub>
</p>
