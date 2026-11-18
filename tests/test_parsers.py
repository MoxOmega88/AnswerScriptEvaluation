"""Tests for Phase 2 parsers. Run from the project root:  python -m pytest -q"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.labels import parse_label  # noqa: E402
from src.schema_parser import parse_schema  # noqa: E402
from src.script_parser import parse_script  # noqa: E402


# ---------- label detection ----------
@pytest.mark.parametrize("line, expected", [
    ("Q1 (a) (b) Oxygen", "1a"),
    ("Q1(a) text", "1a"),
    ("1.a) c", "1a"),
    ("1a. text", "1a"),
    ("1 (b) text", "1b"),
    ("Q2. Ohm", "2"),
    ("2) Ohm", "2"),
    ("Ans 2: V=IR", "2"),
    ("Question 3: text", "3"),
    ("4 OR. Decomposers", "4-OR"),
    ("Q4-OR text", "4-OR"),
    ("Q4 (or) text", "4-OR"),
    ("Q5(ii) text", "5ii"),
    ("3 (A) text", "3a"),
])
def test_labels_detected(line, expected):
    assert parse_label(line)[0] == expected


@pytest.mark.parametrize("line", [
    "10 percent of energy is passed on",
    "2 molecules of water",
    "4 or 5 trophic levels",
    "1.5 V is the voltage",
    "2:1 ratio",
    "Quick answer here",
    "Answer: (b)",
])
def test_not_labels(line):
    assert parse_label(line) is None


# ---------- helpers ----------
SCHEMA_TXT = """Test Paper
Total marks: 8

Q1 [1 mark] [type: mcq]
Question: Pick one.
Answer: (b)

Q2 [2 marks] [type: descriptive]
Question: Explain X.
Model answer: X is Y. It does Z.
Points:
- X is Y (1)
- It does Z (1)

Q3 [5 marks]
Question: Explain A.
Model answer: A is B.
Points:
- A is B (5)

Q3-OR [5 marks]
Question: Explain C.
Model answer: C is D.
Points:
- C is D (5)
"""


@pytest.fixture
def schema(tmp_path):
    f = tmp_path / "schema.txt"
    f.write_text(SCHEMA_TXT, encoding="utf-8")
    return parse_schema(f)


def script_from(tmp_path, text, schema):
    f = tmp_path / "student.txt"
    f.write_text(text, encoding="utf-8")
    return parse_script(f, schema)


# ---------- schema ----------
def test_schema_basic(schema):
    assert list(schema.questions) == ["1", "2", "3", "3-OR"]
    assert schema.total_marks == 8          # OR pair counted once
    assert schema.questions["1"].answer_key == "(b)"
    assert [p.marks for p in schema.questions["2"].points] == [1, 1]
    assert schema.questions["3"].qtype == "descriptive"  # default type
    assert schema.warnings == []


def test_schema_warns_on_bad_points(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("Q1 [3 marks]\nQuestion: Q\nModel answer: M\nPoints:\n- one (1)\n", encoding="utf-8")
    s = parse_schema(f)
    assert any("add up to 1" in w for w in s.warnings)


# ---------- scripts ----------
def test_script_mapping_and_missing(tmp_path, schema):
    s = script_from(tmp_path, "Name: Asha\nRoll No: 7\n\n1) b\n\n3 OR. C is D.", schema)
    assert s.name == "Asha" and s.roll_no == "7"
    assert s.answers == {"1": "b", "3-OR": "C is D."}
    assert s.missing == ["2"]               # Q3 is covered by Q3-OR


def test_multiline_and_numbered_list_inside_answer(tmp_path, schema):
    text = "Q2.\nX is Y.\nReasons:\n1. first reason\n2. second reason\nQ1) (b)"
    s = script_from(tmp_path, text, schema)
    # "1." / "2." follow "Reasons:", so they are list items inside Q2, not answers to Q1/Q2
    assert s.answers["2"] == "X is Y. Reasons: 1. first reason 2. second reason"
    assert s.answers["1"] == "(b)"


def test_label_reused_is_kept_as_text(tmp_path, schema):
    text = "Q1) b\nQ2. X is Y.\nSteps:\n1. do this\n2. do that"
    s = script_from(tmp_path, text, schema)
    assert s.answers["1"] == "b"
    assert "1. do this" in s.answers["2"] and "2. do that" in s.answers["2"]


def test_both_or_answered_warns(tmp_path, schema):
    s = script_from(tmp_path, "Q3. A is B.\nQ3 OR. C is D.", schema)
    assert any("both Q3 and Q3-OR" in w for w in s.warnings)


def test_no_answers_warns(tmp_path, schema):
    s = script_from(tmp_path, "I did not label anything.", schema)
    assert s.answers == {} and s.warnings
