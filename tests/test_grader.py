"""Tests for Phase 3 grading logic. A fake bag-of-words embedder is used so the tests run
fast and without downloading MiniLM.  Run:  python -m pytest -q"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import embedder  # noqa: E402
from src.grader import grade_mcq, grade_question, grade_script  # noqa: E402
from src.schema_parser import MarkingPoint, Question, Schema  # noqa: E402
from src.script_parser import StudentScript  # noqa: E402
from src.text_utils import keyword_coverage, keywords, round_half  # noqa: E402


def fake_embed(texts):
    """Hash stemmed keywords into a vector: same words -> similarity 1, no shared words -> 0."""
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in keywords(t):
            out[i, zlib.crc32(w.encode()) % 256] += 1
        n = np.linalg.norm(out[i])
        if n:
            out[i] /= n
    return out


@pytest.fixture(autouse=True)
def use_fake_embedder():
    embedder.set_embedder(fake_embed)
    yield
    embedder.set_embedder(None)


def test_round_half():
    assert [round_half(x) for x in (0.2, 0.25, 0.7, 1.24, 1.25, 2.0)] == [0.0, 0.5, 0.5, 1.0, 1.5, 2.0]


def test_keyword_coverage_uses_stems():
    assert keyword_coverage("SI unit of resistance is ohm", "Resistance is measured in ohms") == 0.5


# ---------- MCQ ----------
MCQ = Question("1a", 1, "mcq", text="Gas released? (a) Carbon dioxide (b) Oxygen (c) Nitrogen (d) Hydrogen",
               answer_key="(b)")


@pytest.mark.parametrize("answer, marks", [
    ("(b) Oxygen", 1), ("b", 1), ("B.", 1), ("Option b", 1), ("Oxygen", 1),
    ("c", 0), ("(a)", 0), ("Nitrogen", 0),
])
def test_mcq(answer, marks):
    assert grade_mcq(MCQ, answer).suggested == marks


def test_mcq_unreadable_is_flagged():
    r = grade_mcq(MCQ, "I think it is the gas we breathe")
    assert r.suggested == 0 and r.flags


# ---------- descriptive ----------
Q2 = Question("2", 3, "descriptive", text="State Ohm's law ...",
              model_answer="Current is proportional to potential difference. V = IR. Unit is ohm.",
              points=[MarkingPoint("Current is directly proportional to potential difference", 1),
                      MarkingPoint("Relation V = IR", 1),
                      MarkingPoint("SI unit of resistance is ohm", 1)])


def test_full_answer_gets_full_marks():
    ans = ("The current is directly proportional to the potential difference. "
           "The relation is V = IR. The SI unit of resistance is the ohm.")
    r = grade_question(Q2, ans)
    assert r.suggested == 3
    assert all(p.status == "full" for p in r.points)
    assert all(p.evidence for p in r.points)


def test_partial_answer():
    r = grade_question(Q2, "V = IR. Unit is volt.")
    assert 0 < r.suggested < 3
    assert r.points[0].status == "missed"


def test_empty_answer():
    r = grade_question(Q2, "   ")
    assert r.suggested == 0 and r.method == "empty"


def test_manual_question_has_no_suggestion():
    q = Question("5", 3, "manual", text="Solve the numerical")
    assert grade_question(q, "R = 5 ohm").suggested is None


# ---------- whole script ----------
def make_schema():
    alt = Question("4-OR", 2, "descriptive", model_answer="Decomposers break down dead matter.",
                   points=[MarkingPoint("Decomposers break down dead matter", 2)])
    main = Question("4", 2, "descriptive", model_answer="A food chain transfers energy.",
                    points=[MarkingPoint("A food chain transfers energy", 2)])
    return Schema("t", {"1a": MCQ, "2": Q2, "4": main, "4-OR": alt})


def test_script_totals_missing_and_or():
    script = StudentScript("s.docx", answers={"1a": "(b)", "4-OR": "Decomposers break down dead matter."})
    r = grade_script(make_schema(), script)
    by_label = {q.label: q for q in r.questions}
    assert by_label["2"].method == "missing"
    assert "4" not in by_label            # only the answered OR half is graded
    assert r.total_suggested == 1 + 2
    assert r.total_max == 1 + 3 + 2       # OR pair counted once


def test_both_or_answered_counts_best():
    script = StudentScript("s.docx", answers={"4": "Something unrelated.",
                                              "4-OR": "Decomposers break down dead matter."})
    r = grade_script(make_schema(), script)
    counted = [q for q in r.questions if q.label.startswith("4") and q.counted]
    assert len(counted) == 1 and counted[0].label == "4-OR"
