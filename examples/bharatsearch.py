"""BharatSearch — Farmer agent demo using MemX memory."""

from memx import MemX, MemoryType

print("🌾 BharatSearch: Farmer Agent Memory Demo")
print("=" * 50)

memx = MemX()

# ----- Biographical / Semantic memories -----
memx.add("User is Ramesh, a sugarcane farmer from Kolhapur, Maharashtra")
memx.add("Ramesh owns 5 acres of irrigated farmland")
memx.add("Primary crop is sugarcane; secondary crops are soybean and turmeric")
memx.add("Ramesh's Aadhaar is linked to PM-Kisan account")

# ----- Episodic memories -----
memx.add("Yesterday Ramesh checked PNR status for train 12127 Pune-Mumbai")
memx.add("Last week Ramesh visited the Krishi Vigyan Kendra for soil testing")
memx.add("Ramesh experienced crop loss due to unseasonal rain in October 2024")

# ----- Decision memories -----
memx.add("Ramesh decided to switch from chemical urea to organic compost")
memx.add("Ramesh chose drip irrigation over flood irrigation to save water")

# ----- Procedural memories -----
memx.add("To apply for PM-Kisan: step 1 visit pmkisan.gov.in, step 2 register Aadhaar, step 3 verify bank details")
memx.add("How to check soil health card: visit soilhealth.dac.gov.in and enter survey number")

# ----- Causal memories -----
memx.add("Crop failure in kharif 2024 caused Ramesh to take a ₹2 lakh bank loan")
memx.add("Switching to organic fertilizer resulted in 15% yield improvement")

# ----- Working / Active -----
memx.add("Ramesh is currently preparing land for rabi season sowing")
memx.add("Urgent: Ramesh needs to renew crop insurance before December 31")

# Causal links
ids = memx.all()
if len(ids) >= 13:
    memx.link(ids[6].id, ids[11].id, label="crop_loss→loan")
    memx.link(ids[7].id, ids[12].id, label="organic_switch→yield_up")

print(f"\n📝 Loaded {memx.stats()['total']} memories")
print(f"📊 Types: {memx.stats()['types']}\n")

# ----- RAG queries a farmer agent would make -----
queries = [
    "What crops does the farmer grow?",
    "train PNR status",
    "How to apply for PM-Kisan?",
    "Why did the farmer take a loan?",
    "farmer irrigation decision",
    "urgent tasks",
]

for q in queries:
    print(f"🔍 \"{q}\"")
    results = memx.rag(q, top_k=3)
    for r in results:
        print(f"   [{r.type.name:<12}] {r.score:.3f}  {r.content[:70]}")
    print()
