"""MemX Quickstart — 10 lines to AI memory."""

from memx import MemX, MemoryType

# 1-line setup
memx = MemX()

# Add memories (auto-classified)
memx.add("The capital of France is Paris")
memx.add("User visited Mumbai last week for a conference")
memx.add("To deploy: step 1 build Docker image, step 2 push to registry")
memx.add("Slow response times caused the team to add caching")
memx.add("User decided to migrate from MySQL to PostgreSQL")

print(f"Stored {memx.stats()['total']} memories\n")

# Query
for query in ["France capital", "deploy steps", "database decision"]:
    print(f"🔍 rag('{query}'):")
    for r in memx.rag(query, top_k=3):
        print(f"   [{r.type.name:<12}] score={r.score:.3f}  {r.content}")
    print()

print(f"📊 Stats: {memx.stats()}")
