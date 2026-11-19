"""Numeric features describing how well an answer matches the reference.
Used to evaluate on Mohler and as DNN inputs (Phase 4). None of them depend on
the full/half thresholds, so the DNN does not need retraining when thresholds change."""
from src.embedder import embed
from src.grader import point_scores
from src.schema_parser import MarkingPoint
from src.text_utils import keyword_coverage, split_sentences

FEATURE_NAMES = [
    "sim_model",         # cosine(answer, model answer)
    "mean_point_score",  # average best-match score over marking points
    "min_point_score",   # weakest point
    "max_point_score",   # strongest point
    "keyword_overlap",   # share of model-answer keywords present in the answer
    "length_ratio",      # answer words / model-answer words (capped at 3)
]


def points_from_model_answer(model_answer: str, max_marks: float) -> list[MarkingPoint]:
    """When there is no marking scheme (e.g. Mohler), each model-answer sentence is one point."""
    sents = split_sentences(model_answer) or [model_answer]
    return [MarkingPoint(s, max_marks / len(sents)) for s in sents]


def assemble_features(answer: str, model_answer: str, scores: list[float], sim_model: float) -> dict:
    """Build the feature dict from values the grader has already computed."""
    scores = scores or [0.0]
    return {
        "sim_model": float(sim_model),
        "mean_point_score": sum(scores) / len(scores),
        "min_point_score": min(scores),
        "max_point_score": max(scores),
        "keyword_overlap": keyword_coverage(model_answer, answer),
        "length_ratio": min(len(answer.split()) / max(len(model_answer.split()), 1), 3.0),
    }


def compute_features(answer: str, model_answer: str, points: list[MarkingPoint]) -> dict:
    if not answer.strip():
        return {name: 0.0 for name in FEATURE_NAMES}
    va, vm = embed([answer, model_answer])
    scores = [s[0] for s in point_scores(points, answer)]
    return assemble_features(answer, model_answer, scores, float(va @ vm))