"""Find the full/half similarity thresholds that best match human marks.

Two modes:
  * CBSE (preferred, once you have human marks):
        python scripts/tune_thresholds.py --schema data/cbse/schema/my_schema.docx \
            --scripts data/cbse/scripts --marks data/cbse/human_marks.csv
  * Mohler train split (works right now, gives a starting point):
        python scripts/tune_thresholds.py

Human marks CSV format (one row per student per question):
    file,question,marks
    student_01_ananya.docx,1b,2
    student_01_ananya.docx,4,5
Only descriptive questions are used for tuning.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features import points_from_model_answer  # noqa: E402
from src.grader import point_scores  # noqa: E402
from src.schema_parser import MarkingPoint, parse_schema  # noqa: E402
from src.script_parser import parse_script  # noqa: E402
from src.text_utils import round_half  # noqa: E402


def items_from_cbse(schema_path, scripts_dir, marks_csv):
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
            if q is None or q.qtype != "descriptive" or row.question not in script.answers:
                continue
            points = q.points or [MarkingPoint(q.model_answer, q.max_marks)]
            items.append((points, point_scores(points, script.answers[row.question]),
                          q.max_marks, float(row.marks)))
    return items


def items_from_mohler():
    feats = config.PROCESSED_DIR / "mohler_features.csv"
    df = pd.read_csv(config.MOHLER_CSV)
    if feats.exists():  # use the same train split as eval_mohler.py
        split = pd.read_csv(feats, usecols=["id", "split"])
        df = df.merge(split, on="id")
        df = df[df["split"] == "train"]
        print(f"Using Mohler TRAIN split ({len(df)} answers)")
    else:
        print("Run scripts/eval_mohler.py first to use the same train/test split. Using all rows.")
    items = []
    for r in df.itertuples():
        points = points_from_model_answer(r.instructor_answer, 5.0)
        items.append((points, point_scores(points, r.student_answer), 5.0, float(r.score_avg)))
    return items


def mae_for(items, full_th, half_th) -> float:
    errs = []
    for points, scores, max_marks, human in items:
        total = 0.0
        for p, (score, *_rest) in zip(points, scores):
            total += p.marks if score >= full_th else p.marks / 2 if score >= half_th else 0
        errs.append(abs(min(round_half(total), max_marks) - human))
    return float(np.mean(errs))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema")
    ap.add_argument("--scripts")
    ap.add_argument("--marks")
    args = ap.parse_args()

    if args.marks:
        items = items_from_cbse(args.schema, args.scripts, args.marks)
        print(f"CBSE items (descriptive answers with human marks): {len(items)}")
    else:
        items = items_from_mohler()
    if not items:
        sys.exit("No items to tune on. Check file names in the marks CSV match the script files.")

    results = []
    for full_th in np.arange(0.40, 0.91, 0.05):
        for half_th in np.arange(0.20, full_th - 0.001, 0.05):
            results.append((mae_for(items, full_th, half_th), round(full_th, 2), round(half_th, 2)))
    results.sort()

    prefix = "POINT" if args.marks else "MOHLER"
    cur_full = getattr(config, f"{prefix}_FULL_THRESHOLD")
    cur_half = getattr(config, f"{prefix}_HALF_THRESHOLD")
    cur = mae_for(items, cur_full, cur_half)
    print(f"\nCurrent config  full={cur_full}  half={cur_half}  ->  MAE {cur:.3f}")
    print("Best 5:")
    for mae, f, h in results[:5]:
        print(f"  full={f:.2f}  half={h:.2f}  ->  MAE {mae:.3f}")
    best = results[0]
    print(f"\nTo use the best values, set in config.py:\n"
          f"  {prefix}_FULL_THRESHOLD = {best[1]:.2f}\n  {prefix}_HALF_THRESHOLD = {best[2]:.2f}")
    if len(items) < 40:
        print("(Few items - treat these as rough. More marked scripts give more reliable thresholds.)")
