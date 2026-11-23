"""Tests for the FastAPI backend (Phase 6). Uses a temporary database and upload folder,
the fake embedder, and no DNN / summaries, so it runs in a few seconds.
Run:  python -m pytest -q"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402

import config  # noqa: E402
from scripts.make_samples import SCHEMA, STUDENT_GOOD, STUDENT_WEAK  # noqa: E402
from src import embedder  # noqa: E402
from test_grader import fake_embed  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "USE_DNN", False)
    monkeypatch.setattr(config, "USE_NLI", False)   # no NLI model download in tests
    monkeypatch.setattr(config, "SHOW_SUMMARIES", False)
    monkeypatch.setattr(config, "WARMUP_MODELS", False)
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    embedder.set_embedder(fake_embed)

    from app import db
    monkeypatch.setattr(db, "engine", None)
    db.init_db(f"sqlite:///{tmp_path / 'test.db'}")
    from app.main import app

    with TestClient(app) as c:
        yield c
    embedder.set_embedder(None)


def txt(name, content):
    return (name, content.encode("utf-8"), "text/plain")


def make_exam(client):
    r = client.post("/api/exams", files={"file": txt("schema.txt", SCHEMA)}, data={"title": "Unit Test 1"})
    assert r.status_code == 200, r.text
    return r.json()


def upload(client, exam_id, *files):
    r = client.post(f"/api/exams/{exam_id}/submissions", files=[("files", f) for f in files])
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------- exams
def test_create_exam(client):
    exam = make_exam(client)
    assert exam["title"] == "Unit Test 1"
    assert exam["total_marks"] == 14
    assert [q["label"] for q in exam["questions"]] == ["1a", "1b", "2", "3", "4", "4-OR"]
    assert client.get("/api/exams").json()[0]["id"] == exam["id"]


def test_bad_schema_rejected(client):
    r = client.post("/api/exams", files={"file": txt("schema.txt", "nothing useful here")})
    assert r.status_code == 400
    r = client.post("/api/exams", files={"file": ("schema.png", b"x", "image/png")})
    assert r.status_code == 400


# ---------------------------------------------------------------- upload + grading
def test_upload_and_grade(client):
    exam = make_exam(client)
    res = upload(client, exam["id"], txt("ananya.txt", STUDENT_GOOD), txt("rahul.txt", STUDENT_WEAK),
                 ("photo.jpg", b"x", "image/jpeg"))
    assert len(res["queued"]) == 2 and len(res["rejected"]) == 1

    # background grading has finished by the time TestClient returns
    detail = client.get(f"/api/exams/{exam['id']}").json()
    assert detail["n_evaluated"] == 2
    subs = {s["roll_no"]: s for s in detail["submissions"]}
    assert subs["01"]["student_name"] == "Ananya R"
    assert subs["02"]["status"] == "evaluated"

    rahul = client.get(f"/api/submissions/{subs['02']['id']}").json()
    by_label = {a["label"]: a for a in rahul["answers"]}
    assert by_label["3"]["method"] == "missing"
    assert by_label["1a"]["suggested"] == 0          # wrong MCQ option
    assert "4-OR" in by_label
    assert rahul["total_max"] == 14


def test_duplicate_roll_number_warning(client):
    exam = make_exam(client)
    upload(client, exam["id"], txt("a.txt", STUDENT_GOOD))
    upload(client, exam["id"], txt("a_again.txt", STUDENT_GOOD))
    subs = client.get(f"/api/exams/{exam['id']}").json()["submissions"]
    assert any("also appears" in w for s in subs for w in s["warnings"])


# ---------------------------------------------------------------- teacher review
def test_review_flow(client):
    exam = make_exam(client)
    sub_id = upload(client, exam["id"], txt("rahul.txt", STUDENT_WEAK))["queued"][0]["id"]
    answers = client.get(f"/api/submissions/{sub_id}").json()["answers"]
    q2 = next(a for a in answers if a["label"] == "2")

    # validation
    assert client.patch(f"/api/answers/{q2['id']}", json={"final_marks": 0.3}).status_code == 400
    assert client.patch(f"/api/answers/{q2['id']}", json={"final_marks": 9}).status_code == 400

    # teacher overrides Q2 and adds a comment
    r = client.patch(f"/api/answers/{q2['id']}", json={"final_marks": 1, "teacher_comment": "Wrong unit"})
    assert r.status_code == 200 and r.json()["final_marks"] == 1

    # cannot finalise while other answers have no final mark
    assert client.post(f"/api/submissions/{sub_id}/finalize").status_code == 400

    # accept the rest, then finalise
    r = client.post(f"/api/submissions/{sub_id}/accept-all")
    assert r.status_code == 200 and r.json()["accepted"] >= 1
    r = client.post(f"/api/submissions/{sub_id}/finalize")
    assert r.status_code == 200 and r.json()["status"] == "finalized"

    detail = client.get(f"/api/submissions/{sub_id}").json()
    counted = [a for a in detail["answers"] if a["counted"]]
    assert detail["total_final"] == sum(a["final_marks"] for a in counted)
    assert next(a for a in detail["answers"] if a["label"] == "2")["final_marks"] == 1  # override kept

    # locked after finalising, editable again after reopening
    assert client.patch(f"/api/answers/{q2['id']}", json={"final_marks": 2}).status_code == 400
    assert client.post(f"/api/submissions/{sub_id}/reopen").status_code == 200
    assert client.patch(f"/api/answers/{q2['id']}", json={"final_marks": 2}).status_code == 200


def test_export_csv(client):
    exam = make_exam(client)
    upload(client, exam["id"], txt("ananya.txt", STUDENT_GOOD), txt("rahul.txt", STUDENT_WEAK))
    r = client.get(f"/api/exams/{exam['id']}/export.csv")
    assert r.status_code == 200 and "text/csv" in r.headers["content-type"]
    lines = r.text.strip().splitlines()
    assert lines[0].startswith("roll_no,name,file,status,Q1a,Q1b,Q2,Q3,Q4,total")
    assert len(lines) == 3 and lines[1].startswith("01,Ananya R")


def test_delete(client):
    exam = make_exam(client)
    sub_id = upload(client, exam["id"], txt("ananya.txt", STUDENT_GOOD))["queued"][0]["id"]
    assert client.delete(f"/api/submissions/{sub_id}").status_code == 200
    assert client.get(f"/api/submissions/{sub_id}").status_code == 404
    assert client.delete(f"/api/exams/{exam['id']}").status_code == 200
    assert client.get(f"/api/exams/{exam['id']}").status_code == 404



# ---------------------------------------------------------------- HTML pages
def test_pages_render(client):
    assert "No exams yet" in client.get("/").text
    exam = make_exam(client)
    sub_id = upload(client, exam["id"], txt("rahul.txt", STUDENT_WEAK))["queued"][0]["id"]

    home = client.get("/")
    assert home.status_code == 200 and "Unit Test 1" in home.text

    page = client.get(f"/exams/{exam['id']}")
    assert page.status_code == 200 and "Rahul K" in page.text and "Upload and grade" in page.text

    review = client.get(f"/submissions/{sub_id}")
    assert review.status_code == 200
    assert "Rahul K" in review.text and "Q4-OR" in review.text and "Save mark" in review.text

    assert client.get("/static/style.css").status_code == 200
    assert client.get("/submissions/9999").status_code == 404


def test_highlight_segments():
    from app.pages import highlight_segments
    points = [{"status": "full", "evidence": "V = IR."}, {"status": "missed", "evidence": ""},
              {"status": "half", "evidence": "V = IR. Unit is volt."}]
    segs = highlight_segments("Ohm's law. V = IR. Unit is volt.", points)
    assert [s["status"] for s in segs] == [None, "full", "half"]
    assert segs[1]["points"] == [1, 3]



# ---------------------------------------------------------------- feedback (Phase 7)
def test_template_feedback_filled_when_graded(client):
    exam = make_exam(client)
    sub_id = upload(client, exam["id"], txt("rahul.txt", STUDENT_WEAK))["queued"][0]["id"]
    answers = {a["label"]: a for a in client.get(f"/api/submissions/{sub_id}").json()["answers"]}
    assert answers["3"]["teacher_comment"] == "This question was not answered."
    assert answers["1a"]["teacher_comment"] == "The option chosen is not correct."


def test_ai_feedback(client, monkeypatch):
    from src import feedback
    exam = make_exam(client)
    sub_id = upload(client, exam["id"], txt("rahul.txt", STUDENT_WEAK))["queued"][0]["id"]

    monkeypatch.setattr(feedback, "llm_available", lambda: False)
    r = client.post(f"/api/submissions/{sub_id}/feedback")
    assert r.status_code == 400 and "COHERE_API_KEY" in r.json()["detail"]

    # teacher writes their own feedback on Q2 - it must be kept
    answers = {a["label"]: a for a in client.get(f"/api/submissions/{sub_id}").json()["answers"]}
    client.patch(f"/api/answers/{answers['2']['id']}", json={"teacher_comment": "See me after class."})

    monkeypatch.setattr(feedback, "llm_available", lambda: True)
    monkeypatch.setattr(feedback, "llm_feedback", lambda question, answer, points: "AI FEEDBACK")
    r = client.post(f"/api/submissions/{sub_id}/feedback")
    assert r.status_code == 200, r.text

    answers = {a["label"]: a for a in client.get(f"/api/submissions/{sub_id}").json()["answers"]}
    assert answers["2"]["teacher_comment"] == "See me after class."
    written = [a for a in answers.values() if a["teacher_comment"] == "AI FEEDBACK"]
    assert len(written) == r.json()["written"]

    csv_text = client.get(f"/api/exams/{exam['id']}/feedback.csv").text
    assert csv_text.startswith("roll_no,name,question,marks,max_marks,feedback,status")
    assert "See me after class." in csv_text