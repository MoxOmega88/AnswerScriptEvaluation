"""Phase 3 demo: grade student scripts against a schema and print the teacher's view.

Usage (defaults to the sample files):
    python scripts/grade_demo.py
    python scripts/grade_demo.py --schema data/cbse/schema/x.docx --scripts data/cbse/scripts \
        --marks data/cbse/human_marks.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

import pandas as pd  # noqa: E402

from src.doc_reader import SUPPORTED  # noqa: E402
from src.grader import grade_script  # noqa: E402
from src.schema_parser import parse_schema  # noqa: E402
from src.script_parser import parse_script  # noqa: E402

ICON = {"full": "[+]", "half": "[~]", "missed": "[ ]"}


def show(result, human: dict) -> None:
    print("=" * 78)
    print(f"{result.file_name} | {result.name or '?'} (roll {result.roll_no or '?'})")
    print("=" * 78)
    for q in result.questions:
        sug = "manual" if q.suggested is None else f"{q.suggested:g}/{q.max_marks:g}"
        extra = f"  baseline {q.baseline:g}" if q.baseline is not None else ""
        extra += f"  dnn {q.dnn_marks:g}" if q.dnn_marks is not None else ""
        hum = f"  human {human[q.label]:g}" if q.label in human else ""
        note = "" if q.counted else "  (not counted)"
        print(f"Q{q.label:<6} suggested {sug:<8}{extra}{hum}  [{q.method}]{note}")
        for p in q.points:
            print(f"   {ICON[p.status]} {p.awarded:g}/{p.max_marks:g}  score {p.score:.2f} "
                  f"(sem {p.semantic:.2f}, kw {p.keyword:.2f})  {p.text[:55]}")
            if p.evidence:
                print(f"         evidence: \"{p.evidence[:70]}\"")
        for f in q.flags:
            print(f"   ! {f}")
    print("-" * 78)
    print(f"TOTAL suggested: {result.total_suggested:g} / {result.total_max:g}"
          + (f"   (+ {result.manual_marks:g} marks to mark manually)" if result.manual_marks else ""))
    if human:
        print(f"TOTAL human:     {sum(human.values()):g}")
    for f in result.flags:
        print(f"! {f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default=str(config.SAMPLES_DIR / "sample_answer_schema.docx"))
    ap.add_argument("--scripts", default=str(config.SAMPLES_DIR))
    ap.add_argument("--marks", default=str(config.SAMPLES_DIR / "human_marks_sample.csv"))
    args = ap.parse_args()

    marks = pd.read_csv(args.marks, dtype={"question": str}) if Path(args.marks).exists() else None
    schema = parse_schema(args.schema)
    out_dir = config.PROCESSED_DIR / "graded"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in Path(args.scripts).iterdir()
                   if p.suffix.lower() in SUPPORTED and "schema" not in p.name.lower())
    for f in files:
        result = grade_script(schema, parse_script(f, schema))
        human = {}
        if marks is not None:
            sub = marks[marks["file"] == f.name]
            human = dict(zip(sub["question"].str.strip(), sub["marks"].astype(float)))
        show(result, human)
        (out_dir / f"{f.stem}.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"\nJSON saved in {out_dir.relative_to(config.ROOT)}")
