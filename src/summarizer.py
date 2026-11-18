"""Abstractive summariser (DistilBART) used to give the teacher a short gist of long answers.

transformers v5 removed the "summarization" pipeline, so we load the model and call
generate() ourselves. This works on both transformers v4 and v5.
"""
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


@lru_cache(maxsize=1)
def _load():
    """Load tokenizer + model once and reuse them (loading is slow)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.SUMMARIZER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.SUMMARIZER_MODEL)
    model.eval()
    return tokenizer, model


def summarize(text: str, max_length: int = 60, min_length: int = 15) -> str:
    """Return a short summary, or the original text if it is short enough already."""
    if len(text.split()) <= config.SUMMARIZE_MIN_WORDS:
        return text

    import torch

    tokenizer, model = _load()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True).strip()