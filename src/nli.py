"""Natural language inference (NLI) contradiction check.

Semantic similarity cannot tell "current is directly proportional to V" from "current is
inversely proportional to V": the words are almost the same. An NLI cross-encoder reads the
student's evidence sentence (premise) and the marking point (hypothesis) together and says
whether the premise entails, is neutral to, or contradicts the point.

Loads the model once (lazily) and caches it, like src/embedder.py.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

_model = None
_contra_idx: int | None = None
_override = None  # tests can plug in a fake NLI function


def set_nli(fn) -> None:
    """Replace the real model with any function: list[(premise, hypothesis)] -> list[float]."""
    global _override
    _override = fn


def _get_model():
    """Load the cross-encoder once and find which output is the 'contradiction' label."""
    global _model, _contra_idx
    if _model is None:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(config.NLI_MODEL)
        id2label = model.config.id2label
        matches = [int(i) for i, lab in id2label.items() if lab.lower().startswith("contradiction")]
        if not matches:
            raise ValueError(f"{config.NLI_MODEL} has no 'contradiction' label: {id2label}")
        _model, _contra_idx = model, matches[0]
    return _model


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def contradiction_probs(pairs: list[tuple[str, str]]) -> list[float]:
    """For each (premise=student evidence, hypothesis=marking point) return P(contradiction)."""
    if _override is not None:
        return [float(p) for p in _override(pairs)]
    if not pairs:
        return []
    model = _get_model()
    logits = np.asarray(model.predict(list(pairs), apply_softmax=False, show_progress_bar=False))
    return [float(p) for p in _softmax(logits.reshape(len(pairs), -1))[:, _contra_idx]]
