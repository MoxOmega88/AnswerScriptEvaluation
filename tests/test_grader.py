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
def use_fake_embedder(monkeypatch):
    import config
    
    monkeypatch.setattr(config, "USE_DNN", False)   # tests must not depend on the trained model
    monkeypatch.setattr(config, "USE_NLI", False)   # ... or download the NLI model
    monkeypatch.setattr(config, "SHOW_SUMMARIES", False)
    embedder.set_embedder(fake_embed)
    yield
    embedder.set_embedder(None)


def test_round_half():
    assert [round_half(x) for x in (0.2, 0.25, 0.7, 1.24, 1.25, 2.0)] == [0.0, 0.5, 0.5, 1.0, 1.5, 2.0]


def test_keyword_coverage_uses_stems():
    assert keyword_coverage("SI unit of resistance is ohm", "Resistance is measured in ohms") == 0.5

def test_formula_normalisation():
    from src.text_utils import normalise_formulas
    assert keyword_coverage("Relation V = IR", "So V = I x R.") == keyword_coverage("Relation V = IR", "V = IR")
    assert keyword_coverage("Relation V = IR", "V = I × R") > 0.6
    assert normalise_formulas("e.g. a") == "e.g. a"   # normal text is left alone
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



# ---------- DNN second opinion ----------
def test_dnn_disagreement_is_flagged(monkeypatch):
    from src import dnn
    monkeypatch.setattr(dnn, "is_available", lambda: True)
    monkeypatch.setattr(dnn, "predict_norm", lambda rows: np.array([0.0]))  # DNN says 0 marks
    ans = ("The current is directly proportional to the potential difference. "
           "The relation is V = IR. The SI unit of resistance is the ohm.")
    r = grade_question(Q2, ans)
    assert r.suggested == 3 and r.dnn_marks == 0
    assert any("Second opinion differs" in f for f in r.flags)


def test_dnn_agreement_is_not_flagged(monkeypatch):
    from src import dnn
    monkeypatch.setattr(dnn, "is_available", lambda: True)
    monkeypatch.setattr(dnn, "predict_norm", lambda rows: np.array([0.95]))
    ans = ("The current is directly proportional to the potential difference. "
           "The relation is V = IR. The SI unit of resistance is the ohm.")
    r = grade_question(Q2, ans)
    assert r.dnn_marks == 3 and not any("Second opinion" in f for f in r.flags)



# ---------- summaries ----------
def test_long_answer_gets_summary_but_marks_use_full_text(monkeypatch):
    import config
    from src import summarizer
    monkeypatch.setattr(config, "SHOW_SUMMARIES", True)
    monkeypatch.setattr(summarizer, "summarize", lambda text: "SHORT SUMMARY")
    filler = " Some extra words about electricity." * 20          # makes the answer > 80 words
    ans = ("The current is directly proportional to the potential difference. "
           "The relation is V = IR. The SI unit of resistance is the ohm." + filler)
    r = grade_question(Q2, ans)
    assert r.summary == "SHORT SUMMARY"
    assert r.suggested == 3            # marks still come from the full answer


def test_short_answer_has_no_summary(monkeypatch):
    import config
    monkeypatch.setattr(config, "SHOW_SUMMARIES", True)
    r = grade_question(Q2, "V = IR.")
    assert r.summary == ""


def test_extractive_summary_keeps_student_sentences():
    from src.summarizer import extractive_summary
    text = "Ohm's law relates current and voltage. V = IR. Unit is ohm. I like cricket."
    out = extractive_summary(text, ratio=0.5)
    assert all(s in text for s in out.split(". ") if s)
    assert len(out) < len(text)



# ---------- OR answer written under the wrong number ----------
def test_or_answer_with_wrong_label_is_regraded():
    script = StudentScript("s.docx", answers={"4": "Decomposers break down dead matter."})
    r = grade_script(make_schema(), script)
    by_label = {q.label: q for q in r.questions}
    assert "4-OR" in by_label and "4" not in by_label
    assert by_label["4-OR"].suggested == 2
    assert any("matches Q4-OR" in f for f in by_label["4-OR"].flags)


def test_correct_label_is_not_switched():
    script = StudentScript("s.docx", answers={"4": "A food chain transfers energy."})
    r = grade_script(make_schema(), script)
    assert "4" in {q.label for q in r.questions}


# ---------- NLI contradiction check ----------
FULL_ANS = ("The current is directly proportional to the potential difference. "
            "The relation is V = IR. The SI unit of resistance is the ohm.")


def fake_nli_for(word):
    """Fake NLI: any evidence sentence containing `word` contradicts its point."""
    return lambda pairs: [0.95 if word in premise.lower() else 0.05 for premise, _ in pairs]


@pytest.fixture
def nli_on(monkeypatch):
    import config
    from src import nli
    monkeypatch.setattr(config, "USE_NLI", True)
    yield nli
    nli.set_nli(None)


def test_contradicted_point_loses_marks_and_is_flagged(nli_on):
    nli_on.set_nli(fake_nli_for("ohm"))
    r = grade_question(Q2, FULL_ANS)
    p = r.points[2]
    assert p.status == "missed" and p.awarded == 0 and p.contradiction == 0.95
    assert "ohm" in p.evidence                      # evidence kept for the teacher
    assert r.suggested == 2
    assert any(f.startswith("Point 3 may be contradicted") for f in r.flags)


def test_not_contradicted_points_keep_marks(nli_on):
    nli_on.set_nli(fake_nli_for("ohm"))
    r = grade_question(Q2, FULL_ANS)
    assert [p.status for p in r.points[:2]] == ["full", "full"]
    assert r.points[0].contradiction == 0.05
    assert sum(1 for f in r.flags if "contradicted" in f) == 1


def test_nli_off_changes_nothing(monkeypatch):
    from src import nli
    called = []
    nli.set_nli(lambda pairs: called.append(pairs) or [1.0] * len(pairs))
    try:
        r = grade_question(Q2, FULL_ANS)                # USE_NLI is False in the fixture
    finally:
        nli.set_nli(None)
    assert not called
    assert r.suggested == 3 and all(p.contradiction == 0.0 for p in r.points)
    assert not any("contradicted" in f for f in r.flags)
