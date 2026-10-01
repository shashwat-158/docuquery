import numpy as np

from app import config

_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding  # imported lazily: heavy import, only needed at runtime

        _model = TextEmbedding(model_name=config.EMBED_MODEL, cache_dir=config.EMBED_CACHE)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised float32 vectors, shape (len(texts), dim).
    After normalising, inner product == cosine similarity."""
    vectors = np.array(list(_get_model().embed(texts)), dtype="float32")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)