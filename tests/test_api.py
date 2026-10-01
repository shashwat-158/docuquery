import io
import re
import zlib

import numpy as np
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app import config, embeddings, llm, main
from app.vector_store import VectorStore


def fake_embed(texts, dim=384):
    """Deterministic bag-of-words vectors: tests retrieval without downloading a model."""
    vecs = np.zeros((len(texts), dim), dtype="float32")
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z]+", text.lower()):
            vecs[row, zlib.crc32(word.encode()) % dim] += 1
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-12, None)


def make_pdf(pages: list[str]) -> bytes:
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf)
    for text in pages:
        pdf.drawString(72, 750, text)
        pdf.showPage()
    pdf.save()
    return buf.getvalue()


def test_health():
    assert TestClient(main.app).get("/health").status_code == 200


def test_rejects_non_pdf():
    files = {"file": ("notes.txt", b"hello", "text/plain")}
    assert TestClient(main.app).post("/documents", files=files).status_code == 400


def test_upload_then_ask(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(main, "store", VectorStore(str(tmp_path), dim=384))
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    monkeypatch.setattr(llm, "complete", lambda system, user: "Chloroplasts do this. [1]")
    client = TestClient(main.app)

    pdf = make_pdf([
        "Paris is the capital of France.",
        "Chloroplasts convert light into chemical energy.",
        "Python was created by Guido van Rossum.",
    ])
    upload = client.post("/documents", files={"file": ("facts.pdf", pdf, "application/pdf")})
    assert upload.status_code == 200
    assert upload.json()["chunks"] == 3

    reply = client.post(
        "/ask", json={"question": "Which part of a cell converts light into energy?", "top_k": 2}
    )
    body = reply.json()
    assert body["sources"][0]["page"] == 2
    assert "[1]" in body["answer"]