"""Phase 5: does summarising the answer BEFORE marking help or hurt?

Grades the same answers three ways and compares with human marks:
  1. full answer            (what the app does)
  2. extractive summary     (student's own most central sentences)
  3. abstractive summary    (DistilBART rewrites it)
Also reports how much shorter the summaries are, and how many marking points are
lost (or gained) because of summarising.

Usage:
    python scripts/ablation_summary.py                       # Mohler, answers > 40 words
    python scripts/ablation_summary.py --min-words 30        # more Mohler answers (slower)
    python scripts/ablation_summary.py --schema data/cbse/schema/x.docx \
        --scripts data/cbse/scripts --marks data/cbse/human_marks.csv --min-words 40
The first run downloads nothing new but takes a few minutes (DistilBART on CPU).
Summaries are cached, so re-runs are quick.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features import points_from_model_answer  # noqa: E402
from src.grader import apply_thresholds, point_scores  # noqa: E402
from src.metrics import evaluate, format_row  # noqa: E402
from src.schema_parser import MarkingPoint, parse_schema  # noqa: E402
from src.script_parser import parse_script  # noqa: E402
from src.summarizer import extractive_summary, summarize_many  # noqa: E402
from src.text_utils import round_half  # noqa: E402


def mohler_items(min_words: int, limit: int | None) -> list[dict]:
    df = pd.read_csv(config.MOHLER_CSV)
    df = df[df["student_answer"].str.split().str.len() > min_words]
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=config.RANDOM_SEED)
    return [{"answer": r.student_answer, "human": float(r.score_avg), "max": 5.0,
             "points": points_from_model_answer(r.instructor_answer, 5.0), "id": r.id}
            for r in df.itertuples()]


def cbse_items(schema_path, scripts_dir, marks_csv, min_words: int) -> list[dict]:
    schema = parse_schema(schema_path)
    marks = pd.read_csv(marks_csv, dtype={"question": str})
    marks["question"] = marks["question"].str.strip()
    items = []
    for f in sorted(Path(scripts_dir).iterdir()):
        sub = marks[marks["file"] == f.name]
        if sub.empty:
            continue
        script = parse_script(f, schema)
        for row in sub.itertuples():
            q = schema.questions.get(row.question)
            ans = script.answers.get(row.question, "")
            if q is None or q.qtype != "descriptive" or len(ans.split()) <= min_words:
                continue
            items.append({"answer": ans, "human": float(row.marks), "max": q.max_marks,
                          "points": q.points or [MarkingPoint(q.model_answer, q.max_marks)],
                          "id": f"{f.name}:{row.question}"})
    return items


def mark(points, text, max_marks, full_th, half_th):
    res = apply_thresholds(points, point_scores(points, text), full_th, half_th)
    return min(round_half(sum(p.awarded for p in res)), max_marks), [p.status for p in res]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema")
    ap.add_argument("--scripts")
    ap.add_argument("--marks")
    ap.add_argument("--min-words", type=int, default=40, help="only answers longer than this")
    ap.add_argument("--limit", type=int, default=None, help="Mohler only: max answers to use")
    args = ap.parse_args()

    if args.marks:
        mode = "cbse"
        items = cbse_items(args.schema, args.scripts, args.marks, args.min_words)
        full_th, half_th = config.POINT_FULL_THRESHOLD, config.POINT_HALF_THRESHOLD
    else:
        mode = "mohler"
        items = mohler_items(args.min_words, args.limit)
        full_th, half_th = config.MOHLER_FULL_THRESHOLD, config.MOHLER_HALF_THRESHOLD
    if not items:
        sys.exit("No answers longer than --min-words. Try a smaller value.")
    print(f"Mode: {mode} | answers longer than {args.min_words} words: {len(items)}")
    print(f"Thresholds: full >= {full_th}, half >= {half_th}")

    answers = [it["answer"] for it in items]
    print("Abstractive summaries (DistilBART) ...")
    abstractive = summarize_many(answers, show_progress=True)
    print("Extractive summaries ...")
    extractive = [extractive_summary(a) for a in answers]

    conditions = {"Full answer (app)": answers, "Extractive summary": extractive,
                  "Abstractive summary": abstractive}
    human = np.array([it["human"] for it in items])
    full_status = None
    rows = []
    print(f"\n=== Point-wise marking with and without summarisation ({len(items)} answers) ===")
    for name, texts in conditions.items():
        marks, statuses = zip(*(mark(it["points"], t, it["max"], full_th, half_th)
                                for it, t in zip(items, texts)))
        m = evaluate(human, marks)
        ratio = np.mean([len(t.split()) / len(a.split()) for t, a in zip(texts, answers)])
        if full_status is None:
            full_status, lost, gained = statuses, 0.0, 0.0
        else:  # compare point by point with the full-answer result
            pairs = [(a, b) for fa, sa in zip(full_status, statuses) for a, b in zip(fa, sa)]
            lost = np.mean([a != "missed" and b == "missed" for a, b in pairs])
            gained = np.mean([a == "missed" and b != "missed" for a, b in pairs])
        print(format_row(name, m) + f"  length {ratio:.0%}  points lost {lost:.1%}  gained {gained:.1%}")
        rows.append({"condition": name, **m, "length_ratio": ratio,
                     "points_lost": lost, "points_gained": gained})

    out_dir = config.ROOT / "reports"
    pd.DataFrame(rows).to_csv(out_dir / f"summary_ablation_{mode}.csv", index=False)

    # A few examples for the paper / viva
    with open(out_dir / f"summary_examples_{mode}.txt", "w", encoding="utf-8") as fh:
        for it, ext, abs_ in list(zip(items, extractive, abstractive))[:5]:
            fh.write(f"ID: {it['id']}   human mark: {it['human']:g}/{it['max']:g}\n")
            fh.write(f"ORIGINAL ({len(it['answer'].split())} words):\n{it['answer']}\n\n")
            fh.write(f"EXTRACTIVE ({len(ext.split())} words):\n{ext}\n\n")
            fh.write(f"ABSTRACTIVE ({len(abs_.split())} words):\n{abs_}\n")
            fh.write("-" * 70 + "\n")
    print(f"\nSaved reports/summary_ablation_{mode}.csv and reports/summary_examples_{mode}.txt")
    print("points lost = marking points found in the full answer but missed after summarising")
    print("points gained = points missed in the full answer but 'found' in the summary")