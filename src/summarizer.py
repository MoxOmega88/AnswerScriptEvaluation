"""Answer summarisation.

* Abstractive (DistilBART): writes a short new summary. Shown to the teacher as a quick gist.
* Extractive: keeps the student's own most central sentences (cannot invent content).
Both are compared against "no summary" in scripts/ablation_summary.py.

transformers v5 removed the "summarization" pipeline, so we load the model and call
generate() ourselves. Summaries are cached on disk, so re-runs are fast.
"""
import hashlib
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.embedder import embed  # noqa: E402
from src.text_utils import split_sentences  # noqa: E402

CACHE_PATH = config.PROCESSED_DIR / "summary_cache.json"
_cache: dict[str, str] | None = None


# ---------------------------------------------------------------- cache
def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _get_cache() -> dict[str, str]:
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    return _cache


def _save_cache() -> None:
    CACHE_PATH.write_text(json.dumps(_get_cache()), encoding="utf-8")


# ---------------------------------------------------------------- abstractive
@lru_cache(maxsize=1)
def _load():
    """Load tokenizer + model once and reuse them (loading is slow)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.SUMMARIZER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.SUMMARIZER_MODEL)
    model.eval()
    return tokenizer, model


def _lengths(n_tokens: int) -> tuple[int, int]:
    """Summary length relative to the input: about 25-60% of it, never longer than the answer."""
    max_len = min(120, max(20, int(n_tokens * 0.6)))
    min_len = min(max_len - 5, max(10, int(n_tokens * 0.25)))
    return max_len, min_len


def summarize_many(texts: list[str], batch_size: int = 8, show_progress: bool = False) -> list[str]:
    """Abstractive summaries for many texts (batched and cached)."""
    cache = _get_cache()
    todo = [t for t in dict.fromkeys(texts) if _key(t) not in cache]

    if todo:
        import torch

        tokenizer, model = _load()
        todo.sort(key=lambda t: len(t.split()))          # similar lengths in one batch
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=1024, padding=True)
            max_len, min_len = _lengths(int(enc["attention_mask"].sum(dim=1).min()))
            with torch.no_grad():
                ids = model.generate(**enc, max_length=max_len, min_length=min_len, num_beams=4,
                                     no_repeat_ngram_size=3, early_stopping=True)
            for text, summary in zip(batch, tokenizer.batch_decode(ids, skip_special_tokens=True)):
                cache[_key(text)] = summary.strip()
            if show_progress:
                print(f"  summarised {min(start + batch_size, len(todo))}/{len(todo)}", end="\r")
        if show_progress:
            print()
        _save_cache()

    return [cache[_key(t)] for t in texts]


def summarize(text: str, min_words: int | None = None) -> str:
    """Abstractive summary, or the original text if it is short enough already."""
    limit = config.SUMMARIZE_MIN_WORDS if min_words is None else min_words
    if len(text.split()) <= limit:
        return text
    return summarize_many([text])[0]


# ---------------------------------------------------------------- extractive
def extractive_summary(text: str, ratio: float = 0.5) -> str:
    """Keep the most central sentences (closest to the answer's average meaning), in original order."""
    sents = split_sentences(text)
    if len(sents) <= 2:
        return text
    vecs = embed(sents)
    centrality = vecs @ vecs.mean(axis=0)
    k = max(1, math.ceil(len(sents) * ratio))
    keep = sorted(np.argsort(-centrality)[:k])
    return " ".join(sents[i] for i in keep)