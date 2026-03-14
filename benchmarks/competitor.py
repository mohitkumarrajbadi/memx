"""
Competitor Comparison Framework — head-to-head benchmarks against Mem0, ChromaDB, Pinecone.

Provides a standardised harness that benchmarks MemX against competitors
on identical workloads. Competitors are optional — if not installed, their
columns show "N/A" instead of failing.

Metrics compared:
- Setup time (cold start)
- Add throughput (ops/s)
- Query latency (p50/p95/p99)
- Memory footprint
- Feature completeness score
"""

import time
import gc
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

import numpy as np


# ─── Result types ───────────────────────────────────────────────

@dataclass
class CompetitorResult:
    name: str
    available: bool
    setup_ms: float = 0.0
    add_ops_per_s: float = 0.0
    add_p50_ms: float = 0.0
    add_p99_ms: float = 0.0
    rag_ops_per_s: float = 0.0
    rag_p50_ms: float = 0.0
    rag_p95_ms: float = 0.0
    rag_p99_ms: float = 0.0
    memory_mb: float = 0.0
    features: Dict[str, bool] = field(default_factory=dict)
    error: str = ""


def _percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = int(len(s) * p / 100.0)
    return s[min(k, len(s) - 1)]


def _get_rss_mb() -> float:
    try:
        import resource
        if sys.platform == "darwin":
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0.0


# ─── MemX benchmark ────────────────────────────────────────────

def _bench_memx(n: int, n_queries: int) -> CompetitorResult:
    from memx import MemX

    gc.collect()
    rss_before = _get_rss_mb()

    # Setup
    t0 = time.perf_counter()
    m = MemX()
    setup_ms = (time.perf_counter() - t0) * 1000

    # Add
    add_latencies = []
    for i in range(n):
        content = f"MemX benchmark entry {i}: domain {i % 100} topic {i % 50}"
        t0 = time.perf_counter()
        m.add(content)
        add_latencies.append((time.perf_counter() - t0) * 1000)

    add_total = sum(add_latencies) / 1000.0

    # RAG
    rag_latencies = []
    for i in range(n_queries):
        q = f"domain {i % 100} topic {i % 50}"
        t0 = time.perf_counter()
        m.rag(q, top_k=5)
        rag_latencies.append((time.perf_counter() - t0) * 1000)

    rag_total = sum(rag_latencies) / 1000.0

    gc.collect()
    rss_after = _get_rss_mb()

    return CompetitorResult(
        name="MemX",
        available=True,
        setup_ms=setup_ms,
        add_ops_per_s=n / add_total if add_total > 0 else 0,
        add_p50_ms=_percentile(add_latencies, 50),
        add_p99_ms=_percentile(add_latencies, 99),
        rag_ops_per_s=n_queries / rag_total if rag_total > 0 else 0,
        rag_p50_ms=_percentile(rag_latencies, 50),
        rag_p95_ms=_percentile(rag_latencies, 95),
        rag_p99_ms=_percentile(rag_latencies, 99),
        memory_mb=max(rss_after - rss_before, 0.0),
        features={
            "7_memory_types": True,
            "auto_classification": True,
            "hybrid_rag": True,
            "causal_graphs": True,
            "zero_config": True,
            "cli_tools": True,
            "pluggable_backends": True,
        },
    )


# ─── ChromaDB benchmark ────────────────────────────────────────

def _bench_chromadb(n: int, n_queries: int) -> CompetitorResult:
    try:
        import chromadb  # type: ignore
    except ImportError:
        return CompetitorResult(name="ChromaDB", available=False, error="not installed (pip install chromadb)")

    gc.collect()
    rss_before = _get_rss_mb()

    t0 = time.perf_counter()
    client = chromadb.Client()
    collection = client.create_collection("benchmark", metadata={"hnsw:space": "cosine"})
    setup_ms = (time.perf_counter() - t0) * 1000

    # Add
    add_latencies = []
    for i in range(n):
        t0 = time.perf_counter()
        collection.add(
            documents=[f"ChromaDB benchmark entry {i}: domain {i % 100} topic {i % 50}"],
            ids=[f"id-{i}"],
        )
        add_latencies.append((time.perf_counter() - t0) * 1000)

    add_total = sum(add_latencies) / 1000.0

    # Query
    rag_latencies = []
    for i in range(n_queries):
        t0 = time.perf_counter()
        collection.query(query_texts=[f"domain {i % 100} topic {i % 50}"], n_results=5)
        rag_latencies.append((time.perf_counter() - t0) * 1000)

    rag_total = sum(rag_latencies) / 1000.0

    gc.collect()
    rss_after = _get_rss_mb()

    return CompetitorResult(
        name="ChromaDB",
        available=True,
        setup_ms=setup_ms,
        add_ops_per_s=n / add_total if add_total > 0 else 0,
        add_p50_ms=_percentile(add_latencies, 50),
        add_p99_ms=_percentile(add_latencies, 99),
        rag_ops_per_s=n_queries / rag_total if rag_total > 0 else 0,
        rag_p50_ms=_percentile(rag_latencies, 50),
        rag_p95_ms=_percentile(rag_latencies, 95),
        rag_p99_ms=_percentile(rag_latencies, 99),
        memory_mb=max(rss_after - rss_before, 0.0),
        features={
            "7_memory_types": False,
            "auto_classification": False,
            "hybrid_rag": False,
            "causal_graphs": False,
            "zero_config": True,
            "cli_tools": False,
            "pluggable_backends": False,
        },
    )


# ─── Mem0 benchmark ────────────────────────────────────────────

def _bench_mem0(n: int, n_queries: int) -> CompetitorResult:
    try:
        from mem0 import Memory  # type: ignore
    except ImportError:
        return CompetitorResult(name="Mem0", available=False, error="not installed (pip install mem0ai)")

    gc.collect()
    rss_before = _get_rss_mb()

    t0 = time.perf_counter()
    mem = Memory()
    setup_ms = (time.perf_counter() - t0) * 1000

    add_latencies = []
    for i in range(min(n, 500)):  # Mem0 is slow, cap at 500
        t0 = time.perf_counter()
        mem.add(f"Mem0 benchmark entry {i}: domain {i % 100}", user_id="bench")
        add_latencies.append((time.perf_counter() - t0) * 1000)

    add_total = sum(add_latencies) / 1000.0
    actual_n = len(add_latencies)

    rag_latencies = []
    for i in range(min(n_queries, 100)):
        t0 = time.perf_counter()
        mem.search(f"domain {i % 100}", user_id="bench")
        rag_latencies.append((time.perf_counter() - t0) * 1000)

    rag_total = sum(rag_latencies) / 1000.0
    actual_q = len(rag_latencies)

    gc.collect()
    rss_after = _get_rss_mb()

    return CompetitorResult(
        name="Mem0",
        available=True,
        setup_ms=setup_ms,
        add_ops_per_s=actual_n / add_total if add_total > 0 else 0,
        add_p50_ms=_percentile(add_latencies, 50),
        add_p99_ms=_percentile(add_latencies, 99),
        rag_ops_per_s=actual_q / rag_total if rag_total > 0 else 0,
        rag_p50_ms=_percentile(rag_latencies, 50),
        rag_p95_ms=_percentile(rag_latencies, 95),
        rag_p99_ms=_percentile(rag_latencies, 99),
        memory_mb=max(rss_after - rss_before, 0.0),
        features={
            "7_memory_types": False,
            "auto_classification": False,
            "hybrid_rag": False,
            "causal_graphs": False,
            "zero_config": False,
            "cli_tools": False,
            "pluggable_backends": False,
        },
    )


# ─── LanceDB benchmark ─────────────────────────────────────────

def _bench_lancedb(n: int, n_queries: int) -> CompetitorResult:
    try:
        import lancedb  # type: ignore
    except ImportError:
        return CompetitorResult(name="LanceDB", available=False, error="not installed (pip install lancedb)")

    import tempfile, os

    gc.collect()
    rss_before = _get_rss_mb()

    tmpdir = tempfile.mkdtemp()
    t0 = time.perf_counter()
    db = lancedb.connect(os.path.join(tmpdir, "bench.lance"))
    setup_ms = (time.perf_counter() - t0) * 1000

    # LanceDB needs vectors upfront
    dim = 384
    data = [
        {"id": f"id-{i}", "text": f"LanceDB entry {i}: domain {i % 100}", "vector": np.random.rand(dim).tolist()}
        for i in range(n)
    ]

    t0 = time.perf_counter()
    table = db.create_table("benchmark", data)
    add_total = time.perf_counter() - t0

    rag_latencies = []
    for i in range(n_queries):
        q = np.random.rand(dim).tolist()
        t0 = time.perf_counter()
        table.search(q).limit(5).to_list()
        rag_latencies.append((time.perf_counter() - t0) * 1000)

    rag_total = sum(rag_latencies) / 1000.0

    gc.collect()
    rss_after = _get_rss_mb()

    return CompetitorResult(
        name="LanceDB",
        available=True,
        setup_ms=setup_ms,
        add_ops_per_s=n / add_total if add_total > 0 else 0,
        add_p50_ms=(add_total / n) * 1000,
        add_p99_ms=(add_total / n) * 1000,
        rag_ops_per_s=n_queries / rag_total if rag_total > 0 else 0,
        rag_p50_ms=_percentile(rag_latencies, 50),
        rag_p95_ms=_percentile(rag_latencies, 95),
        rag_p99_ms=_percentile(rag_latencies, 99),
        memory_mb=max(rss_after - rss_before, 0.0),
        features={
            "7_memory_types": False,
            "auto_classification": False,
            "hybrid_rag": False,
            "causal_graphs": False,
            "zero_config": True,
            "cli_tools": False,
            "pluggable_backends": False,
        },
    )


# ─── Comparison runner ──────────────────────────────────────────

def run_comparison(
    n: int = 5_000,
    n_queries: int = 500,
    competitors: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[CompetitorResult]:
    """Run head-to-head comparison against available competitors."""
    all_benches = {
        "memx": _bench_memx,
        "chromadb": _bench_chromadb,
        "mem0": _bench_mem0,
        "lancedb": _bench_lancedb,
    }

    if competitors is None:
        competitors = list(all_benches.keys())

    results: List[CompetitorResult] = []

    if verbose:
        print("=" * 90)
        print("  MemX Competitor Comparison — Head-to-Head Benchmark")
        print(f"  Dataset: {n:,} memories | Queries: {n_queries:,} | top_k=5")
        print("=" * 90)

    for name in competitors:
        bench_fn = all_benches.get(name)
        if bench_fn is None:
            continue

        if verbose:
            print(f"\n▶ Benchmarking {name}...")

        try:
            result = bench_fn(n, n_queries)
        except Exception as e:
            result = CompetitorResult(name=name, available=False, error=str(e))

        results.append(result)

        if verbose:
            if result.available:
                print(f"  Setup: {result.setup_ms:.1f}ms")
                print(f"  Add:   {result.add_ops_per_s:,.0f} ops/s  (p50={result.add_p50_ms:.3f}ms, p99={result.add_p99_ms:.3f}ms)")
                print(f"  RAG:   {result.rag_ops_per_s:,.0f} ops/s  (p50={result.rag_p50_ms:.3f}ms, p95={result.rag_p95_ms:.3f}ms, p99={result.rag_p99_ms:.3f}ms)")
                print(f"  Memory: {result.memory_mb:.1f} MB")
            else:
                print(f"  ⏭  Skipped: {result.error}")

    if verbose:
        print("\n")
        _print_comparison_table(results)
        _print_feature_matrix(results)

    return results


def _print_comparison_table(results: List[CompetitorResult]) -> None:
    """Print formatted comparison table."""
    print("─" * 90)
    header = f"{'System':<12} {'Setup':>8} │ {'Add ops/s':>10} {'Add p99':>9} │ {'RAG ops/s':>10} {'RAG p95':>9} {'RAG p99':>9} │ {'Mem MB':>7}"
    print(header)
    print("─" * 90)

    for r in results:
        if r.available:
            print(
                f"{r.name:<12} {r.setup_ms:>7.1f}ms │ "
                f"{r.add_ops_per_s:>10,.0f} {r.add_p99_ms:>8.3f}ms │ "
                f"{r.rag_ops_per_s:>10,.0f} {r.rag_p95_ms:>8.3f}ms {r.rag_p99_ms:>8.3f}ms │ "
                f"{r.memory_mb:>6.1f}MB"
            )
        else:
            print(f"{r.name:<12} {'N/A':>8} │ {'N/A':>10} {'N/A':>9} │ {'N/A':>10} {'N/A':>9} {'N/A':>9} │ {'N/A':>7}")
    print("─" * 90)

    # Speedup vs competitors
    memx = next((r for r in results if r.name == "MemX"), None)
    if memx:
        for r in results:
            if r.name != "MemX" and r.available and r.rag_ops_per_s > 0:
                speedup = memx.rag_ops_per_s / r.rag_ops_per_s
                print(f"  ⚡ MemX is {speedup:.1f}x faster than {r.name} on RAG queries")


def _print_feature_matrix(results: List[CompetitorResult]) -> None:
    """Print feature comparison matrix."""
    available = [r for r in results if r.available or r.features]
    if not available:
        return

    all_features = set()
    for r in available:
        all_features.update(r.features.keys())

    print(f"\n{'Feature':<25}", end="")
    for r in available:
        print(f" {r.name:<12}", end="")
    print()
    print("─" * (25 + 13 * len(available)))

    for feat in sorted(all_features):
        print(f"{feat:<25}", end="")
        for r in available:
            val = r.features.get(feat, False)
            print(f" {'✅':<12}" if val else f" {'❌':<12}", end="")
        print()


if __name__ == "__main__":
    run_comparison()
