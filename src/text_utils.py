"""Small text helpers: sentence splitting, answer spans and keyword coverage."""
import re
from functools import lru_cache

_FALLBACK_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Small built-in stop-word list (no download needed)
STOPWORDS = set("""
a an the and or but if of to in on at by for with from as is are was were be been being
it its this that these those which who whom what when where why how such so than then
there their they them he she his her we our you your i me my do does did done has have
had having can could will would should may might must shall also very more most some
any each every all both other into over under about after before up down out not no
""".split())

_token_re = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=1)
def _stemmer():
    from nltk.stem import PorterStemmer  # no download needed

    return PorterStemmer()


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    try:
        from nltk.tokenize import sent_tokenize

        sents = sent_tokenize(text)
    except LookupError:  # punkt not downloaded
        sents = _FALLBACK_SPLIT.split(text)
    return [s.strip() for s in sents if s.strip()]


def candidate_spans(text: str) -> list[str]:
    """Single sentences plus pairs of neighbouring sentences.

    A marking point is sometimes split across two sentences
    ("...converted into glucose. Oxygen is released."), so pairs are checked too.
    """
    sents = split_sentences(text)
    pairs = [f"{sents[i]} {sents[i + 1]}" for i in range(len(sents) - 1)]
    return sents + pairs


@lru_cache(maxsize=50_000)
def keywords(text: str) -> frozenset[str]:
    stem = _stemmer().stem
    return frozenset(stem(t) for t in _token_re.findall(text.lower()) if t not in STOPWORDS)


def keyword_coverage(reference: str, candidate: str) -> float:
    """Fraction of the reference's keywords (stemmed) that appear in the candidate."""
    ref = keywords(reference)
    if not ref:
        return 0.0
    return len(ref & keywords(candidate)) / len(ref)


def round_half(x: float) -> float:
    """Round to the nearest 0.5 (CBSE allows half marks). 1.25 -> 1.5, 1.2 -> 1.0"""
    return int(x * 2 + 0.5) / 2
