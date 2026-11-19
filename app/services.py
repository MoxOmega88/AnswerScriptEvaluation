"""Business logic used by the web app: exams, uploads, background grading, teacher review."""
import csv
import io
import shutil
import sys
import threading
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app import db as dbmod  # noqa: E402
from app.models import AnswerEval, Exam, Submission  # noqa: E402
from src.doc_reader import SUPPORTED  # noqa: E402
from src.grader import grade_script  # noqa: E402
from src.schema_parser import MarkingPoint, Question, Schema, parse_schema  # noqa: E402
from src.script_parser import parse_script  # noqa: E402

_grading_lock = threading.Lock()  # grade one script at a time (models are shared)


class ServiceError(Exception):
    """A problem the teacher can fix (bad file, missing marks, ...). Shown as a 400 error."""


class NotFound(Exception):
    pass


def get_or_404(db: Session, model, obj_id: int):
    obj = db.get(model, obj_id)
    if obj is None:
        raise NotFound(f"{model.__name__} {obj_id} not found")
    return obj


# ---------------------------------------------------------------- files
def save_upload(fileobj, original_name: str, folder: Path) -> Path:
    name = Path(original_name or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise ServiceError(f"'{name}': unsupported file type. Use {', '.join(sorted(SUPPORTED))}.")
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{uuid.uuid4().hex[:8]}_{name}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(fileobj, out)
    if dest.stat().st_size > config.MAX_UPLOAD_MB * 1024 * 1024:
        dest.unlink()
        raise ServiceError(f"'{name}' is larger than {config.MAX_UPLOAD_MB} MB.")
    return dest


# ---------------------------------------------------------------- exams
def create_exam(db: Session, schema_path: Path, original_name: str, title: str | None = None) -> Exam:
    try:
        schema = parse_schema(schema_path)
    except Exception as err:
        raise ServiceError(f"Could not read the answer schema: {err}") from err
    if not schema.questions:
        raise ServiceError("No questions found in the schema. " + " ".join(schema.warnings))
    exam = Exam(title=(title or "").strip() or schema.title, schema_file=original_name,
                total_marks=schema.total_marks, schema_json=schema.to_dict(), warnings=schema.warnings)
    db.add(exam)
    db.commit()
    db.refresh(exam)
    return exam


def load_schema(exam: Exam) -> Schema:
    """Rebuild the Schema object from the JSON stored in the database."""
    d = exam.schema_json
    questions = {}
    for qd in d["questions"]:
        questions[qd["label"]] = Question(
            label=qd["label"], max_marks=qd["max_marks"], qtype=qd["qtype"], text=qd["text"],
            model_answer=qd["model_answer"], answer_key=qd["answer_key"],
            points=[MarkingPoint(**p) for p in qd.get("points", [])])
    return Schema(title=d["title"], questions=questions, stated_total=d.get("stated_total"),
                  warnings=list(d.get("warnings", [])))


def exam_summary(exam: Exam) -> dict:
    subs = exam.submissions
    return {
        "id": exam.id, "title": exam.title, "schema_file": exam.schema_file,
        "total_marks": exam.total_marks, "warnings": exam.warnings,
        "created_at": exam.created_at.isoformat(timespec="seconds"),
        "n_submissions": len(subs),
        "n_evaluated": sum(s.status in ("evaluated", "finalized") for s in subs),
        "n_finalized": sum(s.status == "finalized" for s in subs),
        "n_flagged_scripts": sum(s.n_flagged > 0 and s.status == "evaluated" for s in subs),
    }


def _review_order(s: Submission):
    """Dashboard order: errors, then scripts with flags (most first), then other evaluated,
    then still processing, finalized last."""
    rank = {"error": 0, "evaluated": 1, "processing": 3, "pending": 3, "finalized": 4}[s.status]
    if s.status == "evaluated" and s.n_flagged == 0:
        rank = 2
    return rank, -s.n_flagged, s.roll_no or "~", s.id


def exam_detail(exam: Exam) -> dict:
    return {
        **exam_summary(exam),
        "questions": exam.schema_json["questions"],
        "submissions": [submission_summary(s) for s in sorted(exam.submissions, key=_review_order)],
    }


# ---------------------------------------------------------------- submissions
def add_submissions(db: Session, exam: Exam, saved: list[tuple[Path, str]]) -> list[Submission]:
    subs = [Submission(exam_id=exam.id, file_name=name, file_path=str(path)) for path, name in saved]
    db.add_all(subs)
    db.commit()
    return subs


def evaluate_submissions(ids: list[int]) -> None:
    """Background task: grade each script and store the results."""
    for sub_id in ids:
        evaluate_one(sub_id)


def evaluate_one(sub_id: int) -> None:
    db = dbmod.SessionLocal()
    try:
        sub = db.get(Submission, sub_id)
        if sub is None:
            return
        sub.status, sub.error = "processing", ""
        db.commit()
        try:
            schema = load_schema(sub.exam)
            with _grading_lock:
                script = parse_script(sub.file_path, schema)
                result = grade_script(schema, script)

            warnings = list(result.flags)
            if script.unmatched:
                labels = ", ".join(lab for lab, _ in script.unmatched)
                warnings.append(f"Question numbers not in the schema: {labels}.")
            if script.roll_no:
                dup = db.scalars(select(Submission).where(
                    Submission.exam_id == sub.exam_id, Submission.roll_no == script.roll_no,
                    Submission.id != sub.id)).first()
                if dup:
                    warnings.append(f"Roll number {script.roll_no} also appears in '{dup.file_name}'.")

            sub.student_name, sub.roll_no = script.name, script.roll_no
            sub.answers = [
                AnswerEval(position=i, label=q.label, qtype=q.qtype, max_marks=q.max_marks,
                           answer_text=q.answer, summary=q.summary, method=q.method,
                           suggested=q.suggested, baseline=q.baseline, dnn_marks=q.dnn_marks,
                           similarity=q.similarity_to_model, counted=q.counted,
                           points=[asdict(p) for p in q.points], flags=list(q.flags))
                for i, q in enumerate(result.questions)
            ]
            sub.warnings = warnings
            sub.total_suggested = result.total_suggested
            sub.total_max = result.total_max
            sub.manual_marks = result.manual_marks
            sub.n_flagged = sum(1 for q in result.questions
                                if q.counted and q.flags and q.method not in ("missing", "manual"))
            sub.total_final, sub.finalized_at = None, None
            sub.status, sub.evaluated_at = "evaluated", datetime.now()
        except Exception as err:  # keep the app running; show the error on the dashboard
            db.rollback()
            sub = db.get(Submission, sub_id)
            sub.status, sub.error = "error", f"{type(err).__name__}: {err}"
        db.commit()
    finally:
        db.close()


def submission_summary(s: Submission) -> dict:
    return {
        "id": s.id, "exam_id": s.exam_id, "file_name": s.file_name,
        "student_name": s.student_name, "roll_no": s.roll_no, "status": s.status, "error": s.error,
        "total_suggested": s.total_suggested, "total_final": s.total_final, "total_max": s.total_max,
        "manual_marks": s.manual_marks, "n_flagged": s.n_flagged, "warnings": s.warnings,
        "uploaded_at": s.uploaded_at.isoformat(timespec="seconds"),
        "finalized_at": s.finalized_at.isoformat(timespec="seconds") if s.finalized_at else None,
    }


def submission_detail(s: Submission) -> dict:
    questions = {q["label"]: q for q in s.exam.schema_json["questions"]}
    answers = []
    for a in s.answers:
        q = questions.get(a.label, {})
        answers.append({
            "id": a.id, "label": a.label, "qtype": a.qtype, "max_marks": a.max_marks,
            "question_text": q.get("text", ""), "model_answer": q.get("model_answer", ""),
            "answer_text": a.answer_text, "summary": a.summary, "method": a.method,
            "suggested": a.suggested, "baseline": a.baseline, "dnn_marks": a.dnn_marks,
            "similarity": a.similarity, "counted": a.counted, "points": a.points, "flags": a.flags,
            "final_marks": a.final_marks, "teacher_comment": a.teacher_comment,
        })
    return {**submission_summary(s), "exam_title": s.exam.title, "answers": answers}


# ---------------------------------------------------------------- teacher review
def set_final_marks(db: Session, answer: AnswerEval, *, update_marks: bool, final_marks: float | None,
                    comment: str | None) -> AnswerEval:
    if answer.submission.status == "finalized":
        raise ServiceError("This script is finalised. Reopen it to change marks.")
    if update_marks:
        if final_marks is not None:
            if not 0 <= final_marks <= answer.max_marks:
                raise ServiceError(f"Marks for Q{answer.label} must be between 0 and {answer.max_marks:g}.")
            if not float(final_marks * 2).is_integer():
                raise ServiceError("Marks must be in steps of 0.5.")
        answer.final_marks = final_marks
    if comment is not None:
        answer.teacher_comment = comment.strip()
    db.commit()
    return answer


def accept_all_suggestions(db: Session, sub: Submission) -> int:
    """Copy every AI suggestion into final marks where the teacher hasn't set one yet."""
    if sub.status != "evaluated":
        raise ServiceError(f"Script is '{sub.status}', not ready for review.")
    n = 0
    for a in sub.answers:
        if a.counted and a.final_marks is None and a.suggested is not None:
            a.final_marks = a.suggested
            n += 1
    db.commit()
    return n


def finalize(db: Session, sub: Submission) -> Submission:
    if sub.status != "evaluated":
        raise ServiceError(f"Script is '{sub.status}', it can't be finalised.")
    missing = [a.label for a in sub.answers if a.counted and a.final_marks is None]
    if missing:
        raise ServiceError("Give final marks for: " + ", ".join(f"Q{lab}" for lab in missing))
    sub.total_final = sum(a.final_marks for a in sub.answers if a.counted)
    sub.status, sub.finalized_at = "finalized", datetime.now()
    db.commit()
    return sub


def reopen(db: Session, sub: Submission) -> Submission:
    if sub.status != "finalized":
        raise ServiceError("Only finalised scripts can be reopened.")
    sub.status, sub.finalized_at, sub.total_final = "evaluated", None, None
    db.commit()
    return sub


def delete_submission(db: Session, sub: Submission) -> None:
    Path(sub.file_path).unlink(missing_ok=True)
    db.delete(sub)
    db.commit()


def delete_exam(db: Session, exam: Exam) -> None:
    for s in exam.submissions:
        Path(s.file_path).unlink(missing_ok=True)
    db.delete(exam)
    db.commit()


# ---------------------------------------------------------------- export
def export_csv(exam: Exam) -> str:
    """One row per student. Final marks if the teacher set them, otherwise the AI suggestion."""
    labels = []
    for q in exam.schema_json["questions"]:
        base = q["label"].removesuffix("-OR")
        if base not in labels:
            labels.append(base)

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["roll_no", "name", "file", "status", *[f"Q{lab}" for lab in labels],
                "total", "total_max", "marks_are"])
    for s in sorted(exam.submissions, key=lambda s: (s.roll_no or "~", s.id)):
        per_q = {}
        for a in s.answers:
            if a.counted:
                per_q[a.label.removesuffix("-OR")] = a.final_marks if a.final_marks is not None else a.suggested
        final = s.status == "finalized"
        total = s.total_final if final else s.total_suggested
        w.writerow([s.roll_no, s.student_name, s.file_name, s.status,
                    *["" if per_q.get(lab) is None else f"{per_q[lab]:g}" for lab in labels],
                    "" if total is None else f"{total:g}",
                    "" if s.total_max is None else f"{s.total_max:g}",
                    "final" if final else "AI suggestion (not reviewed)"])
    return buf.getvalue()


# ---------------------------------------------------------------- start-up
def warm_up() -> None:
    """Load the models once so the first grading request isn't slow."""
    try:
        from src.embedder import embed
        embed(["warm up"])
        from src import dnn
        if dnn.is_available():
            dnn._load()
        print("[warm-up] models loaded")
    except Exception as err:  # never stop the server because of warm-up
        print(f"[warm-up] skipped: {err}")