"""Retrieval hit@k: is the page that holds the answer among the top-k retrieved chunks?

Usage (index your PDF first):  python -m evaluation.run_eval
"""
import json

from app import config, embeddings
from app.vector_store import VectorStore

store = VectorStore(config.DATA_DIR, dim=config.EMBED_DIM)
with open("evaluation/questions.json", encoding="utf-8") as f:
    cases = json.load(f)

hits = 0
for case in cases:
    results = store.search(embeddings.embed([case["question"]]), config.TOP_K)
    found = any(
        r["page"] == case["expected_page"] and r["filename"] == case["filename"]
        for r in results
    )
    hits += found
    print("PASS" if found else "FAIL", "-", case["question"])

print(f"\nhit@{config.TOP_K}: {hits}/{len(cases)} = {hits / len(cases):.0%}")