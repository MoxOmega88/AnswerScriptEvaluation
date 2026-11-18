"""Phase 2 demo: parse an answer schema and student scripts, print a mapping preview,
and save the results as JSON in data/processed/ so you can inspect them.

Usage (defaults to the sample files):
    python scripts/parse_demo.py
    python scripts/parse_demo.py --schema data/cbse/schema/my_schema.docx --scripts data/cbse/scripts
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.doc_reader import SUPPORTED  # noqa: E402
from src.schema_parser import parse_schema  # noqa: E402
from src.script_parser import parse_script  # noqa: E402


def show_schema(schema) -> None:
    print("=" * 70)
    print(f"SCHEMA: {schema.title}")
    print(f"Questions: {len(schema.questions)} | Total marks: {schema.total_marks:g}"
          + (f" (header says {schema.stated_total:g})" if schema.stated_total is not None else ""))
    print("-" * 70)
    for q in schema.questions.values():
        extra = f"answer key {q.answer_key}" if q.qtype == "mcq" else f"{len(q.points)} points"
        print(f"  Q{q.label:<6} {q.max_marks:>4g} marks  {q.qtype:<12} {extra:<14} {q.text[:30]}...")
    for w in schema.warnings:
        print(f"  ! {w}")


def show_script(script, schema) -> None:
    print("=" * 70)
    print(f"SCRIPT: {script.file_name}  |  Name: {script.name or '?'}  |  Roll: {script.roll_no or '?'}")
    print("-" * 70)
    for label in schema.questions:
        if label in script.answers:
            text = script.answers[label]
            print(f"  [found]   Q{label:<6} ({len(text.split()):>3} words) {text[:45]}...")
        elif label in script.missing:
            print(f"  [MISSING] Q{label}")
    for label, text in script.unmatched:
        print(f"  [UNKNOWN] '{label}' not in schema: {text[:40]}...")
    for w in script.warnings:
        print(f"  ! {w}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default=str(config.SAMPLES_DIR / "sample_answer_schema.docx"))
    ap.add_argument("--scripts", default=str(config.SAMPLES_DIR),
                    help="folder of student scripts (files with 'schema' in the name are skipped)")
    args = ap.parse_args()

    schema = parse_schema(args.schema)
    show_schema(schema)

    out_dir = config.PROCESSED_DIR / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "schema.json").write_text(json.dumps(schema.to_dict(), indent=2), encoding="utf-8")

    files = sorted(p for p in Path(args.scripts).iterdir()
                   if p.suffix.lower() in SUPPORTED and "schema" not in p.name.lower())
    for f in files:
        script = parse_script(f, schema)
        show_script(script, schema)
        (out_dir / f"{f.stem}.json").write_text(json.dumps(script.to_dict(), indent=2), encoding="utf-8")

    print("=" * 70)
    print(f"JSON saved in {out_dir.relative_to(config.ROOT)}")
