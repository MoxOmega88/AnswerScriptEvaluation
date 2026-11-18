"""Core grading (Phase 3).

For each answer:
  * mcq          -> exact match on the option letter (or the option text)
  * descriptive  -> point-wise marking: every marking point is compared with every
                    sentence (and pair of sentences) of the answer; the best match decides
                    full / half / zero marks for that point
  * manual       -> no suggestion, teacher marks it
Also computed for comparison: a simple baseline = cosine(answer, model answer) x max marks.

The teacher always sees the evidence sentence for each point and has the final say.
"""
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.embedder import embed  # noqa: E402
from src.schema_parser import MarkingPoint, Question, Schema  # noqa: E402
from src.script_parser import StudentScript  # noqa: E402
from src.text_utils import candidate_spans, keyword_coverage, round_half  # noqa: E402


# ---------------------------------------------------------------- results
@dataclass
class PointResult:
    text: str
    max_marks: float
    score: float          # best match score, 0-1
    semantic: float       # semantic part of the score
    keyword: float        # keyword-coverage part of the score
    status: str           # "full" | "half" | "missed"
    awarded: float
    evidence: str         # the student's sentence(s) that matched best


@dataclass
class QuestionResult:
    label: str
    qtype: str
    max_marks: float
    answer: str
    suggested: float | None              # None -> teacher must mark (manual)
    method: str                          # pointwise | mcq | manual | missing | empty
    baseline: float | None = None        # cosine-to-model-answer baseline
    similarity_to_model: float | None = None
    points: list[PointResult] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    counted: bool = True                 # False for the unused half of an OR pair


@dataclass
class ScriptResult:
    file_name: str
    name: str
    roll_no: str
    questions: list[QuestionResult]
    total_suggested: float
    total_max: float
    manual_marks: float                  # marks the teacher still has to award by hand
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- point scores
def point_scores(points: list[MarkingPoint], answer: str) -> list[tuple[float, float, float, str]]:
    """For each point return (score, semantic, keyword, evidence). Independent of thresholds,
    so threshold tuning can reuse these numbers."""
    spans = candidate_spans(answer)
    if not spans or not points:
        return [(0.0, 0.0, 0.0, "") for _ in points]

    sem = embed([p.text for p in points]) @ embed(spans).T          # (points, spans)
    out = []
    for i, p in enumerate(points):
        kw = np.array([keyword_coverage(p.text, s) for s in spans]) if config.USE_KEYWORD_BACKUP \
            else np.zeros(len(spans))
        combined = np.maximum(sem[i], kw)
        j = int(np.argmax(combined))
        out.append((float(combined[j]), float(sem[i, j]), float(kw[j]), spans[j]))
    return out


def apply_thresholds(points, scores, full_th, half_th) -> list[PointResult]:
    results = []
    for p, (score, s, k, ev) in zip(points, scores):
        if score >= full_th:
            status, awarded = "full", p.marks
        elif score >= half_th:
            status, awarded = "half", p.marks / 2
        else:
            status, awarded = "missed", 0.0
        results.append(PointResult(p.text, p.marks, round(score, 3), round(s, 3), round(k, 3),
                                   status, awarded, ev if status != "missed" else ""))
    return results


# ---------------------------------------------------------------- per question type
def grade_descriptive(q: Question, answer: str, full_th=None, half_th=None) -> QuestionResult:
    full_th = config.POINT_FULL_THRESHOLD if full_th is None else full_th
    half_th = config.POINT_HALF_THRESHOLD if half_th is None else half_th
    res = QuestionResult(q.label, q.qtype, q.max_marks, answer, suggested=0.0, method="pointwise")

    if not answer.strip():
        res.method = "empty"
        res.flags.append("Empty answer.")
        return res

    points = q.points or [MarkingPoint(q.model_answer, q.max_marks)]
    res.points = apply_thresholds(points, point_scores(points, answer), full_th, half_th)
    res.suggested = min(round_half(sum(p.awarded for p in res.points)), q.max_marks)

    if q.model_answer:
        va, vm = embed([answer, q.model_answer])
        res.similarity_to_model = round(float(va @ vm), 3)
        res.baseline = min(round_half(q.max_marks * max(res.similarity_to_model, 0.0)), q.max_marks)

    if len(answer.split()) < 5 and q.max_marks >= 2:
        res.flags.append("Very short answer.")
    if res.points and max(p.score for p in res.points) < half_th:
        res.flags.append("No marking point matched - check if the answer is off-topic or worded unusually.")
    return res


_OPTION_RE = re.compile(r"^\s*(?i:option\s*)?\(?\s*([a-dA-D])\s*[).]?(?=\s|$)")
_OPTIONS_IN_Q_RE = re.compile(r"\(\s*([a-d])\s*\)\s*([^()]+?)(?=\s*\(\s*[a-d]\s*\)|$)", re.IGNORECASE)


def _option_letter(text: str) -> str | None:
    m = _OPTION_RE.match(text or "")
    return m.group(1).lower() if m else None


def grade_mcq(q: Question, answer: str) -> QuestionResult:
    res = QuestionResult(q.label, q.qtype, q.max_marks, answer, suggested=0.0, method="mcq")
    key = _option_letter(q.answer_key)
    chosen = _option_letter(answer)

    if chosen is None and answer.strip():  # student wrote the option text, e.g. "Oxygen"
        for letter, opt_text in _OPTIONS_IN_Q_RE.findall(q.text):
            if opt_text.strip().lower() in answer.lower():
                chosen = letter.lower()
                break

    if key is None:
        res.suggested = None
        res.flags.append("MCQ answer key could not be read; mark manually.")
    elif chosen is None:
        res.flags.append("Could not read which option was chosen; please check.")
    elif chosen == key:
        res.suggested = q.max_marks
    return res


def grade_question(q: Question, answer: str, full_th=None, half_th=None) -> QuestionResult:
    if q.qtype == "mcq":
        return grade_mcq(q, answer)
    if q.qtype == "manual":
        r = QuestionResult(q.label, q.qtype, q.max_marks, answer, suggested=None, method="manual")
        r.flags.append("Manual question - teacher to mark.")
        return r
    return grade_descriptive(q, answer, full_th, half_th)


# ---------------------------------------------------------------- whole script
def grade_script(schema: Schema, script: StudentScript, full_th=None, half_th=None) -> ScriptResult:
    results: list[QuestionResult] = []
    done = set()

    for label, q in schema.questions.items():
        if label in done:
            continue
        group = [label] + schema.alternatives(label)        # e.g. ["4", "4-OR"]
        done.update(group)
        answered = [lab for lab in group if lab in script.answers]

        if not answered:
            r = QuestionResult(label, q.qtype, q.max_marks, "", suggested=0.0, method="missing")
            r.flags.append("Not answered.")
            results.append(r)
            continue

        graded = [grade_question(schema.questions[lab], script.answers[lab], full_th, half_th)
                  for lab in answered]
        if len(graded) > 1:  # both halves of an OR pair answered: count the better one
            best = max(graded, key=lambda r: r.suggested or 0)
            for r in graded:
                r.counted = r is best
                r.flags.append("Both OR options answered; the higher-scoring one is counted.")
        results.extend(graded)

    counted = [r for r in results if r.counted]
    total = sum(r.suggested for r in counted if r.suggested is not None)
    manual = sum(r.max_marks for r in counted if r.suggested is None)

    flags = list(script.warnings)
    if manual:
        flags.append(f"{manual:g} marks need manual marking.")
    return ScriptResult(script.file_name, script.name, script.roll_no, results,
                        total_suggested=total, total_max=schema.total_marks,
                        manual_marks=manual, flags=flags)
