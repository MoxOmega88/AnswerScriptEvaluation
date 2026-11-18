"""Sentence embeddings with MiniLM. Loads the model once and caches every sentence it has seen."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

_model = None
_cache: dict[str, np.ndarray] = {}
_override = None  # tests can plug in a fake embedder


def set_embedder(fn) -> None:
    """Replace the real model with any function: list[str] -> (n, d) array of unit vectors."""
    global _override
    _override = fn
    _cache.clear()


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Return one L2-normalised vector per text, so cosine similarity = dot product."""
    if _override is not None:
        return np.asarray(_override(texts), dtype=np.float32)
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    new = [t for t in dict.fromkeys(texts) if t not in _cache]
    if new:
        vecs = _get_model().encode(
            new, batch_size=64, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=len(new) > 500,
        )
        _cache.update(zip(new, vecs))
    return np.vstack([_cache[t] for t in texts])


def similarity(a: str, b: str) -> float:
    va, vb = embed([a, b])
    return float(va @ vb)
