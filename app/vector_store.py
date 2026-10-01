import json
import os
import threading

import faiss
import numpy as np


class VectorStore:
    """FAISS index (cosine similarity via inner product on normalised vectors)
    plus a parallel metadata list. Both are persisted to disk."""

    def __init__(self, data_dir: str, dim: int):
        os.makedirs(data_dir, exist_ok=True)
        self.index_path = os.path.join(data_dir, "index.faiss")
        self.meta_path = os.path.join(data_dir, "meta.json")
        self._lock = threading.Lock()
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(dim)  # exact search
            self.meta = []

    def add(self, vectors: np.ndarray, metadatas: list[dict]) -> None:
        with self._lock:
            self.index.add(vectors)
            self.meta.extend(metadatas)
            faiss.write_index(self.index, self.index_path)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.meta, f, ensure_ascii=False)

    def search(self, query_vec: np.ndarray, k: int) -> list[dict]:
        with self._lock:
            if self.index.ntotal == 0:
                return []
            scores, ids = self.index.search(query_vec, min(k, self.index.ntotal))
        return [
            {**self.meta[i], "score": float(s)}
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]

    def documents(self) -> list[dict]:
        counts: dict[str, int] = {}
        for m in self.meta:
            counts[m["filename"]] = counts.get(m["filename"], 0) + 1
        return [{"filename": name, "chunks": n} for name, n in counts.items()]