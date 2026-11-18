"""Parse the teacher's answer schema (.docx / .pdf / .txt) into Python objects.

Expected format for each question (see data/samples/sample_answer_schema.docx):

    Q1b [2 marks] [type: descriptive]
    Question: What is photosynthesis?
    Model answer: Photosynthesis is ...
    Points:
    - Process by which green plants make their own food (1)
    - Uses sunlight, carbon dioxide and water to produce glucose and oxygen (1)

    Q1a [1 mark] [type: mcq]
    Question: ...
    Answer: (b)

Types: descriptive (default), mcq, manual.  Internal choice: Q4 and Q4-OR.
Anything before the first question (title, "Total marks: 14") is the header.
"""
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.doc_reader import read_lines
from src.labels import base_label, parse_label

VALID_TYPES = {"descriptive", "mcq", "manual"}

_MARKS_RE = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*marks?\s*\]", re.IGNORECASE)
_TYPE_RE = re.compile(r"\[\s*type\s*:\s*(\w+)\s*\]", re.IGNORECASE)
_FIELD_RE = re.compile(r"^(question|model answer|answer|points)\s*:\s*(.*)$", re.IGNORECASE)
_POINT_RE = re.compile(r"^[-•*]\s*(.+?)\s*(?:\(\s*(\d+(?:\.\d+)?)\s*(?:marks?)?\s*\))?\s*$", re.IGNORECASE)
_TOTAL_RE = re.compile(r"total\s*marks?\s*[:\-]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass
class MarkingPoint:
    text: str
    marks: float


@dataclass
class Question:
    label: str                 # normalised, e.g. "1b", "4-OR"
    max_marks: float
    qtype: str = "descriptive"
    text: str = ""
    model_answer: str = ""
    answer_key: str = ""       # for mcq
    points: list[MarkingPoint] = field(default_factory=list)

    @property
    def is_alternative(self) -> bool:
        return self.label.endswith("-OR")


@dataclass
class Schema:
    title: str
    questions: dict[str, Question]           # label -> Question, in document order
    stated_total: float | None = None        # "Total marks: 14" if written in the header
    warnings: list[str] = field(default_factory=list)

    @property
    def total_marks(self) -> float:
        """Sum of max marks, counting each OR pair only once."""
        return sum(q.max_marks for q in self.questions.values() if not q.is_alternative)

    def alternatives(self, label: str) -> list[str]:
        """Labels that are interchangeable with this one (e.g. '4' <-> '4-OR')."""
        base = base_label(label)
        return [lab for lab in self.questions if base_label(lab) == base and lab != label]

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "total_marks": self.total_marks,
            "stated_total": self.stated_total,
            "warnings": self.warnings,
            "questions": [asdict(q) for q in self.questions.values()],
        }


def _is_question_header(line: str) -> tuple[str, str] | None:
    """A header is a label followed by [N marks]."""
    parsed = parse_label(line)
    if parsed and _MARKS_RE.search(parsed[1]):
        return parsed
    return None


def parse_schema(path: str | Path) -> Schema:
    lines = read_lines(path)
    header_lines: list[str] = []
    questions: dict[str, Question] = {}
    warnings: list[str] = []

    current: Question | None = None
    current_field: str | None = None

    for line in lines:
        header = _is_question_header(line)
        if header:
            label, rest = header
            marks = float(_MARKS_RE.search(rest).group(1))
            type_match = _TYPE_RE.search(rest)
            qtype = type_match.group(1).lower() if type_match else "descriptive"
            if qtype not in VALID_TYPES:
                warnings.append(f"Q{label}: unknown type '{qtype}', treated as 'manual'.")
                qtype = "manual"
            if label in questions:
                warnings.append(f"Q{label} appears twice; the second one replaces the first.")
            current = Question(label=label, max_marks=marks, qtype=qtype)
            questions[label] = current
            current_field = None
            continue

        if current is None:  # still in the header (title, total marks, instructions)
            header_lines.append(line)
            continue

        field_match = _FIELD_RE.match(line)
        if field_match:
            name, value = field_match.group(1).lower(), field_match.group(2).strip()
            current_field = name
            if name == "question":
                current.text = value
            elif name == "model answer":
                current.model_answer = value
            elif name == "answer":
                current.answer_key = value
            continue

        if current_field == "points":
            point_match = _POINT_RE.match(line)
            if point_match:
                text, marks = point_match.group(1), point_match.group(2)
                if marks is None:
                    warnings.append(f"Q{current.label}: point '{text[:40]}...' has no marks; assumed 1.")
                current.points.append(MarkingPoint(text=text.strip(), marks=float(marks or 1)))
            elif current.points:  # wrapped point text continues on the next line
                current.points[-1].text += " " + line
            continue

        # Continuation of a multi-line field
        if current_field == "question":
            current.text += " " + line
        elif current_field == "model answer":
            current.model_answer += " " + line
        elif current_field == "answer":
            current.answer_key += " " + line

    title = header_lines[0] if header_lines else Path(path).stem
    stated_total = None
    for line in header_lines:
        m = _TOTAL_RE.search(line)
        if m:
            stated_total = float(m.group(1))

    schema = Schema(title=title, questions=questions, stated_total=stated_total, warnings=warnings)
    _validate(schema)
    return schema


def _validate(schema: Schema) -> None:
    w = schema.warnings
    if not schema.questions:
        w.append("No questions found. Check that each question starts like 'Q1a [2 marks]'.")
        return

    for q in schema.questions.values():
        if not q.text:
            w.append(f"Q{q.label}: missing 'Question:' line.")
        if q.qtype == "mcq" and not q.answer_key:
            w.append(f"Q{q.label}: MCQ has no 'Answer:' line.")
        if q.qtype == "descriptive":
            if not q.model_answer:
                w.append(f"Q{q.label}: missing 'Model answer:'.")
            if not q.points:
                w.append(f"Q{q.label}: no marking points; the model answer will be used as one point.")
            else:
                pts = sum(p.marks for p in q.points)
                if abs(pts - q.max_marks) > 1e-6:
                    w.append(f"Q{q.label}: points add up to {pts:g} but the question is worth {q.max_marks:g}.")
        if q.is_alternative and base_label(q.label) not in schema.questions:
            w.append(f"Q{q.label}: OR question without a matching Q{base_label(q.label)}.")
        if q.is_alternative:
            main = schema.questions.get(base_label(q.label))
            if main and main.max_marks != q.max_marks:
                w.append(f"Q{q.label}: worth {q.max_marks:g} but Q{main.label} is worth {main.max_marks:g}.")

    if schema.stated_total is not None and abs(schema.stated_total - schema.total_marks) > 1e-6:
        w.append(f"Header says total {schema.stated_total:g} but questions add up to {schema.total_marks:g}.")
