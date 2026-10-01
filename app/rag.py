import time

from app import embeddings, llm
from app.vector_store import VectorStore

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions using ONLY the numbered context "
    "excerpts provided. Cite the excerpts you rely on inline, like [1] or [2][3]. "
    "If the excerpts do not contain the answer, reply exactly: "
    "\"I couldn't find that in the uploaded documents.\" Never use outside knowledge."
)


def build_prompt(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i}] ({h['filename']}, page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits, start=1)
    )
    return f"Context excerpts:\n\n{context}\n\nQuestion: {question}"


def answer_question(store: VectorStore, question: str, top_k: int) -> dict:
    start = time.perf_counter()
    hits = store.search(embeddings.embed([question]), top_k)
    if not hits:
        answer, sources = "No documents have been uploaded yet.", []
    else:
        answer = llm.complete(SYSTEM_PROMPT, build_prompt(question, hits))
        sources = [
            {
                "id": i,
                "filename": h["filename"],
                "page": h["page"],
                "score": round(h["score"], 3),
                "snippet": h["text"][:300],
            }
            for i, h in enumerate(hits, start=1)
        ]
    latency_ms = int((time.perf_counter() - start) * 1000)
    return {"answer": answer, "sources": sources, "latency_ms": latency_ms}