"""MemX CLI — the AI Memory Operating System command-line interface."""

import time
import click

from .api import MemX
from .types import MemoryType


@click.group()
@click.version_option(package_name="memx-ai")
def cli():
    """🧠 MemX — AI Memory Operating System."""
    pass


@cli.command()
def init():
    """Initialize a MemX project in the current directory."""
    import os
    os.makedirs("memx_data", exist_ok=True)
    click.echo("✅ MemX project initialized.")
    click.echo("   Data dir: ./memx_data")
    click.echo("   Usage:    from memx import MemX; m = MemX()")


@cli.command()
def demo():
    """Run an interactive demo of MemX Memory OS."""
    m = MemX()

    click.echo("🧠 MemX Memory OS Demo")
    click.echo("=" * 50)

    samples = [
        ("My name is Mohit and I'm building MemX", None),
        ("Decided to use FAISS for vector search", None),
        ("Step 1: install memx, Step 2: import MemX", None),
        ("Yesterday deployed v0.2.0 to production", None),
        ("Rain caused server outage in Mumbai region", None),
        ("Currently optimizing the hybrid RAG pipeline", None),
        ("Urgent: security vulnerability in auth module", None),
        ("Python is the most popular programming language", None),
    ]

    click.echo("\n📝 Adding memories...")
    for content, _ in samples:
        mid = m.add(content)
        mem = m.get(mid)
        click.echo(f"  [{mem.type.name:11s}] imp={mem.importance:.2f} │ {content[:60]}")

    click.echo(f"\n📊 Stats: {m.stats()['total']} memories, types: {m.stats()['types']}")

    click.echo("\n🔍 RAG Queries:")
    for q in ["Who am I?", "deployment production", "security urgent"]:
        results = m.rag(q, top_k=2)
        click.echo(f"\n  Q: \"{q}\"")
        for r in results:
            click.echo(f"    [{r.score:.3f}] {r.content[:65]}")

    click.echo("\n🗜️  Compressing memories...")
    result = m.compress()
    click.echo(f"  Compressed: {result['compressed']} groups, deactivated: {result['deactivated']}")

    click.echo("\n💭 Generating reflections...")
    refs = m.reflect()
    for ref in refs:
        click.echo(f"  💭 {ref.content[:80]}")

    click.echo("\n✅ Demo complete! Try: from memx import MemX")


@cli.command()
@click.option("--n", default=10_000, help="Number of memories to benchmark")
def benchmark(n):
    """Run a quick performance benchmark."""
    m = MemX()

    click.echo(f"🏎️  Benchmarking with {n:,} memories...")

    t0 = time.perf_counter()
    for i in range(n):
        m.add(f"Benchmark entry {i}: domain {i % 100} topic {i % 50}", importance=0.5)
    add_time = time.perf_counter() - t0

    click.echo(f"  Add: {add_time:.3f}s total, {(add_time / n) * 1000:.3f}ms/op, {n / add_time:,.0f} ops/s")

    queries = 1000
    t0 = time.perf_counter()
    for i in range(queries):
        m.rag(f"domain {i % 100} topic {i % 50}", top_k=5)
    rag_time = time.perf_counter() - t0

    click.echo(f"  RAG: {rag_time:.3f}s total, {(rag_time / queries) * 1000:.3f}ms/query, {queries / rag_time:,.0f} ops/s")

    t0 = time.perf_counter()
    m.compress()
    comp_time = time.perf_counter() - t0
    click.echo(f"  Compress: {comp_time:.3f}s")

    click.echo(f"\n  📊 Final: {m.stats()['total']} active, {m.stats()['inactive']} inactive")


@cli.command()
def stats():
    """Show memory system stats."""
    m = MemX()
    s = m.stats()
    click.echo("📊 MemX Memory OS Stats")
    click.echo("=" * 40)
    click.echo(f"  Active:      {s['total']}")
    click.echo(f"  Inactive:    {s['inactive']}")
    click.echo(f"  Avg import:  {s['avg_importance']:.2f}")
    click.echo(f"  Namespaces:  {s['namespaces']}")
    click.echo(f"  Graph edges: {s['graph_edges']}")
    click.echo(f"  Types:       {s['types']}")


@cli.command()
@click.option("--port", default=7900, help="Dashboard port")
def dashboard(port):
    """Launch the MemX Memory Dashboard."""
    from .viz.server import serve
    m = MemX()

    # Seed with demo data for visual appeal
    samples = [
        "My name is Mohit and I build AI systems",
        "Prefer Python for all backend development",
        "Working on MemX — open source AI memory",
        "Decided to use FAISS for vector indexing",
        "Step 1: design API, Step 2: implement core",
        "Yesterday released MemX v0.2.0",
        "Server latency caused by N+1 queries",
        "Currently optimizing compression engine",
    ]
    for s in samples:
        m.add(s)

    serve(memx=m, port=port)


@cli.command()
@click.argument("query")
@click.option("--top-k", default=5, help="Number of results")
def inspect(query, top_k):
    """Inspect retrieval scoring for a query."""
    m = MemX()
    exps = m.inspect(query, top_k=top_k)
    if not exps:
        click.echo("No memories to inspect. Add some first!")
        return
    for exp in exps:
        click.echo(exp.explain())
        click.echo()


def main():
    cli()


if __name__ == "__main__":
    main()
