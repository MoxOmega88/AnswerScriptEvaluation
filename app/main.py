"""FastAPI app.  Run from the project root:

    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs to try every endpoint in the browser.
(The HTML pages for teachers come in Phase 6 part 2.)
"""
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app import services  # noqa: E402
from app.db import get_db, init_db  # noqa: E402
from app.models import AnswerEval, Exam, Submission  # noqa: E402


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    if config.WARMUP_MODELS:
        threading.Thread(target=services.warm_up, daemon=True).start()
    yield


app = FastAPI(title="AI Answer Script Evaluation", version="1.0", lifespan=lifespan)


def _get(db: Session, model, obj_id: int):
    try:
        return services.get_or_404(db, model, obj_id)
    except services.NotFound as err:
        raise HTTPException(404, str(err)) from err


def _run(fn, *args, **kwargs):
    """Call a service function and turn ServiceError into a 400 response."""
    try:
        return fn(*args, **kwargs)
    except services.ServiceError as err:
        raise HTTPException(400, str(err)) from err


class AnswerUpdate(BaseModel):
    final_marks: float | None = None
    teacher_comment: str | None = None


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")


# ---------------------------------------------------------------- exams
@app.post("/api/exams", tags=["exams"])
def create_exam(file: UploadFile = File(...), title: str | None = Form(None), db: Session = Depends(get_db)):
    path = _run(services.save_upload, file.file, file.filename, config.UPLOAD_DIR / "schemas")
    try:
        exam = _run(services.create_exam, db, path, file.filename, title)
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    return services.exam_detail(exam)


@app.get("/api/exams", tags=["exams"])
def list_exams(db: Session = Depends(get_db)):
    exams = db.scalars(select(Exam).order_by(Exam.id.desc())).all()
    return [services.exam_summary(e) for e in exams]


@app.get("/api/exams/{exam_id}", tags=["exams"])
def get_exam(exam_id: int, db: Session = Depends(get_db)):
    return services.exam_detail(_get(db, Exam, exam_id))


@app.delete("/api/exams/{exam_id}", tags=["exams"])
def delete_exam(exam_id: int, db: Session = Depends(get_db)):
    services.delete_exam(db, _get(db, Exam, exam_id))
    return {"deleted": exam_id}


@app.get("/api/exams/{exam_id}/export.csv", tags=["exams"])
def export_exam(exam_id: int, db: Session = Depends(get_db)):
    exam = _get(db, Exam, exam_id)
    return Response(services.export_csv(exam), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="exam_{exam_id}_marks.csv"'})


# ---------------------------------------------------------------- submissions
@app.post("/api/exams/{exam_id}/submissions", tags=["submissions"])
def upload_scripts(exam_id: int, background: BackgroundTasks, files: list[UploadFile] = File(...),
                   db: Session = Depends(get_db)):
    exam = _get(db, Exam, exam_id)
    saved, rejected = [], []
    for f in files:
        try:
            saved.append((services.save_upload(f.file, f.filename, config.UPLOAD_DIR / f"exam_{exam_id}"),
                          f.filename))
        except services.ServiceError as err:
            rejected.append(str(err))
    if not saved:
        raise HTTPException(400, "No valid files. " + " ".join(rejected))
    subs = services.add_submissions(db, exam, saved)
    background.add_task(services.evaluate_submissions, [s.id for s in subs])
    return {"queued": [services.submission_summary(s) for s in subs], "rejected": rejected}


@app.get("/api/submissions/{sub_id}", tags=["submissions"])
def get_submission(sub_id: int, db: Session = Depends(get_db)):
    return services.submission_detail(_get(db, Submission, sub_id))


@app.post("/api/submissions/{sub_id}/reevaluate", tags=["submissions"])
def reevaluate(sub_id: int, background: BackgroundTasks, db: Session = Depends(get_db)):
    sub = _get(db, Submission, sub_id)
    if sub.status == "finalized":
        raise HTTPException(400, "Reopen the script before re-evaluating it.")
    sub.status = "pending"
    db.commit()
    background.add_task(services.evaluate_submissions, [sub.id])
    return services.submission_summary(sub)


@app.post("/api/submissions/{sub_id}/accept-all", tags=["review"])
def accept_all(sub_id: int, db: Session = Depends(get_db)):
    sub = _get(db, Submission, sub_id)
    n = _run(services.accept_all_suggestions, db, sub)
    return {"accepted": n, **services.submission_detail(sub)}


@app.post("/api/submissions/{sub_id}/finalize", tags=["review"])
def finalize(sub_id: int, db: Session = Depends(get_db)):
    sub = _get(db, Submission, sub_id)
    return services.submission_summary(_run(services.finalize, db, sub))


@app.post("/api/submissions/{sub_id}/reopen", tags=["review"])
def reopen(sub_id: int, db: Session = Depends(get_db)):
    sub = _get(db, Submission, sub_id)
    return services.submission_summary(_run(services.reopen, db, sub))


@app.delete("/api/submissions/{sub_id}", tags=["submissions"])
def delete_submission(sub_id: int, db: Session = Depends(get_db)):
    services.delete_submission(db, _get(db, Submission, sub_id))
    return {"deleted": sub_id}


@app.patch("/api/answers/{answer_id}", tags=["review"])
def update_answer(answer_id: int, body: AnswerUpdate, db: Session = Depends(get_db)):
    """Teacher sets/changes the final mark (send null to clear it) and/or a comment."""
    answer = _get(db, AnswerEval, answer_id)
    _run(services.set_final_marks, db, answer,
         update_marks="final_marks" in body.model_fields_set,
         final_marks=body.final_marks, comment=body.teacher_comment)
    return {"id": answer.id, "label": answer.label, "final_marks": answer.final_marks,
            "teacher_comment": answer.teacher_comment}