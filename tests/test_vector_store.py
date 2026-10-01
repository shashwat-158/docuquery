import numpy as np

from app.vector_store import VectorStore


def test_add_search_and_persist(tmp_path):
    store = VectorStore(str(tmp_path), dim=3)
    vectors = np.array([[1, 0, 0], [0, 1, 0]], dtype="float32")
    store.add(vectors, [
        {"filename": "a.pdf", "page": 1, "text": "x"},
        {"filename": "b.pdf", "page": 2, "text": "y"},
    ])

    hits = store.search(np.array([[0, 1, 0]], dtype="float32"), k=1)
    assert hits[0]["filename"] == "b.pdf"

    reloaded = VectorStore(str(tmp_path), dim=3)  # survives a restart
    assert reloaded.index.ntotal == 2