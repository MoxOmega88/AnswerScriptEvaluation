"""HTML pages for teachers (server-rendered with Jinja2). The pages call the /api endpoints
from app.js for actions like uploading, saving marks and finalising."""
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import services  # noqa: E402
from app.db import get_db  # noqa: E402
from app.models import Exam, Submission  # noqa: E402
from src.text_utils import split_sentences  # noqa: E402

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
router = APIRouter(include_in_schema=False)

STATUS_TEXT = {"pending": "Waiting", "processing": "Grading", "evaluated": "Ready for review",
               "finalized": "Finalised", "error": "Error"}


def fmt_marks(value) -> str:
    """2.0 -> '2', 1.5 -> '1.5', None -> '–'"""
    return "–" if value is None else f"{value:g}"


templates.env.filters["marks"] = fmt_marks
templates.env.globals["STATUS_TEXT"] = STATUS_TEXT


def _get(db: Session, model, obj_id: int):
    obj = db.get(model, obj_id)
    if obj is None:
        raise HTTPException(404, f"{model.__name__} {obj_id} not found")
    return obj


def highlight_segments(answer: str, points: list[dict]) -> list[dict]:
    """Split the answer into sentences and mark which ones were used as evidence for which
    marking points, so the page can highlight them."""
    segments = []
    for sent in split_sentences(answer):
        status, nums = None, []
        for i, p in enumerate(points, start=1):
            if p["status"] != "missed" and p["evidence"] and sent in p["evidence"]:
                nums.append(i)
                status = "full" if p["status"] == "full" or status == "full" else "half"
        segments.append({"text": sent, "status": status, "points": nums})
    return segments


def exam_stats(exam: dict) -> dict:
    """Numbers for the summary strip at the top of the exam page."""
    subs = exam["submissions"]
    graded = [s for s in subs if s["status"] in ("evaluated", "finalized")]
    suggested = [s["total_suggested"] for s in graded if s["total_suggested"] is not None]
    return {
        "uploaded": len(subs),
        "graded": len(graded),
        "busy": sum(s["status"] in ("pending", "processing") for s in subs),
        "attention": sum(s["status"] == "error" or (s["status"] == "evaluated" and s["n_flagged"] > 0)
                         for s in subs),
        "finalised": sum(s["status"] == "finalized" for s in subs),
        "average": round(sum(suggested) / len(suggested), 1) if suggested else None,
    }


def answer_state(a: dict) -> dict:
    """What the question navigator and the answer card need to know about one answer."""
    return {
        "saved": a["final_marks"] is not None,
        "flagged": bool(a["flags"]) and a["method"] != "missing",
        "missing": a["method"] == "missing",
        "counted": a["counted"],
    }


def needs_feedback(a: dict) -> bool:
    """Answers the AI feedback button writes for (same rule as services.generate_feedback)."""
    return a["counted"] and a["method"] == "pointwise" and any(p["status"] != "full" for p in a["points"])


@router.get("/")
def exams_page(request: Request, db: Session = Depends(get_db)):
    exams = [services.exam_summary(e) for e in db.scalars(select(Exam).order_by(Exam.id.desc())).all()]
    return templates.TemplateResponse(request, "exams.html", {"exams": exams})


@router.get("/exams/{exam_id}")
def exam_page(exam_id: int, request: Request, db: Session = Depends(get_db)):
    exam = services.exam_detail(_get(db, Exam, exam_id))
    stats = exam_stats(exam)
    return templates.TemplateResponse(request, "exam.html",
                                      {"exam": exam, "stats": stats, "busy": stats["busy"] > 0})


@router.get("/submissions/{sub_id}")
def submission_page(sub_id: int, request: Request, db: Session = Depends(get_db)):
    sub = _get(db, Submission, sub_id)
    detail = services.submission_detail(sub)
    for a in detail["answers"]:
        a["segments"] = highlight_segments(a["answer_text"], a["points"])
        a["state"] = answer_state(a)

    counted = [a for a in detail["answers"] if a["counted"]]
    reviewed = sum(a["final_marks"] is not None for a in counted)
    final_so_far = sum(a["final_marks"] for a in counted if a["final_marks"] is not None)

    order = [s["id"] for s in services.exam_detail(sub.exam)["submissions"]]
    pos = order.index(sub.id)
    prev_id = order[pos - 1] if pos > 0 else None
    next_id = order[pos + 1] if pos < len(order) - 1 else None

    return templates.TemplateResponse(request, "submission.html", {
        "sub": detail, "reviewed": reviewed, "n_counted": len(counted),
        "final_so_far": final_so_far, "prev_id": prev_id, "next_id": next_id,
        "position": pos + 1, "n_scripts": len(order),
        "n_feedback": sum(needs_feedback(a) for a in detail["answers"]),
    })