"""Tests for Phase 7 feedback (no API key or internet needed).  Run:  python -m pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.feedback import AnswerFeedback, template_feedback  # noqa: E402

PTS = [{"text": "Defines photosynthesis", "status": "full"},
       {"text": "Mentions chlorophyll", "status": "half"},
       {"text": "Writes the equation", "status": "missed"}]


def test_template_feedback_lists_missing_and_partial_points():
    text = template_feedback("pointwise", PTS, 1.5, 3)
    assert "Missing: Writes the equation." in text
    assert "Explain more clearly: Mentions chlorophyll." in text
    assert "Defines photosynthesis" not in text


def test_template_feedback_other_cases():
    full = [{"text": "A", "status": "full"}]
    assert template_feedback("pointwise", full, 1, 1).startswith("Good answer")
    assert template_feedback("missing", [], 0, 3) == "This question was not answered."
    assert template_feedback("mcq", [], 0, 1) == "The option chosen is not correct."
    assert template_feedback("mcq", [], 1, 1) == ""
    assert template_feedback("manual", [], None, 3) == ""


def test_llm_output_formatting():
    fbk = AnswerFeedback(strength="You defined photosynthesis correctly.",
                         missing=["the role of chlorophyll", "the chemical equation."],
                         advice="Learn the equation by heart.")
    assert fbk.as_text() == ("You defined photosynthesis correctly. "
                             "To improve: the role of chlorophyll; the chemical equation. "
                             "Learn the equation by heart.")
    assert AnswerFeedback(strength="", missing=[], advice="Revise this topic.").as_text() == "Revise this topic."