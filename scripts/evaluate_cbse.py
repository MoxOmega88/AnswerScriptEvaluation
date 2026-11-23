"""Phase 8: final evaluation on the CBSE scripts with reference marks.

1. Splits the Science scripts: about 35% for tuning the full/half thresholds, the rest for testing.
2. Tunes the thresholds on the tuning scripts only.
3. Evaluates on the Science test scripts AND all Social Science scripts (a subject the
   thresholds never saw), comparing: cosine baseline, DNN, point-wise (default thresholds),
   point-wise + NLI contradiction check (default thresholds), point-wise (tuned thresholds).
   All columns except "pointwise_nli" are graded with the NLI check off, so the NLI effect is
   a clean ablation.
4. Reports per-question and per-script agreement, whether the flags point at the errors, and
   how often each kind of flag fires.

Expects:
    data/cbse/science/schema/*.docx          data/cbse/social_science/schema/*.docx
    data/cbse/science/scripts/*              data/cbse/social_science/scripts/*
    data/cbse/science/human_marks.csv        data/cbse/social_science/human_marks.csv
Usage:
    python scripts/evaluate_cbse.py
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

config.SHOW_SUMMARIES = False  # summaries don't change marks; skip them to keep this fast

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.tune_thresholds import mae_for  # noqa: E402
from src.doc_reader import SUPPORTED  # noqa: E402
from src.grader import grade_script, point_scores  # noqa: E402
from src.labels import base_label  # noqa: E402
from src.metrics import evaluate, format_row  # noqa: E402
from src.schema_parser import MarkingPoint, parse_schema  # noqa: E402
from src.script_parser import parse_script  # noqa: E402

REPORTS = config.ROOT / "reports"
SUBJECTS = ["science", "social_science"]


# ---------------------------------------------------------------- loading
def load_subject(name: str):
    root = config.CBSE_DIR / name
    schema_files = [p for p in (root / "schema").iterdir() if p.suffix.lower() in SUPPORTED]
    if not schema_files:
        sys.exit(f"No schema found in {root / 'schema'}")
    schema = parse_schema(schema_files[0])

    marks_df = pd.read_csv(root / "human_marks.csv", dtype={"question": str, "file": str})
    marks_df["question"] = marks_df["question"].str.strip()
    if marks_df["marks"].isna().any():
        empty = marks_df[marks_df["marks"].isna()]
        sys.exit(f"{name}: {len(empty)} rows in human_marks.csv have no mark, e.g. "
                 f"{empty.iloc[0]['file']} Q{empty.iloc[0]['question']}. Fill them in first.")
    marks = {(r.file, r.question): float(r.marks) for r in marks_df.itertuples()}

    scripts = {}
    files_with_marks = set(marks_df["file"])
    for f in sorted((root / "scripts").iterdir()):
        if f.suffix.lower() in SUPPORTED:
            if f.name in files_with_marks:
                scripts[f.name] = parse_script(f, schema)
            else:
                print(f"  ({name}) no marks for {f.name} - skipped")
    for missing in sorted(files_with_marks - set(scripts)):
        print(f"  ({name}) marks given for {missing} but the file is not in scripts/ - skipped")
    return schema, scripts, marks


# ---------------------------------------------------------------- tuning
def tune(schema, scripts, marks, dev_files):
    items = []
    for fname in dev_files:
        script = scripts[fname]
        for (f, label), human in marks.items():
            q = schema.questions.get(label)
            if f != fname or q is None or q.qtype != "descriptive" or label not in script.answers:
                continue
            points = q.points or [MarkingPoint(q.model_answer, q.max_marks)]
            items.append((points, point_scores(points, script.answers[label]), q.max_marks, human))
    results = []
    for full_th in np.arange(0.40, 0.91, 0.05):
        for half_th in np.arange(0.20, full_th - 0.001, 0.05):
            results.append((round(mae_for(items, full_th, half_th), 4), round(full_th, 2), round(half_th, 2)))
    results.sort()
    return results, len(items)


# ---------------------------------------------------------------- grading

def _find(results: dict, label: str):
    """The result for this question, even if it was graded under the other OR option."""
    if label in results:
        return results[label]
    same = [q for lab, q in results.items() if base_label(lab) == base_label(label) and q.counted]
    return same[0] if same else None

def _grade(schema, script, thresholds, use_nli: bool):
    """grade_script with the NLI contradiction check switched on or off for this call only."""
    old = config.USE_NLI
    config.USE_NLI = use_nli
    try:
        return grade_script(schema, script, *thresholds)
    finally:
        config.USE_NLI = old


def grade_rows(subject, schema, scripts, marks, files, tuned):
    rows, totals = [], []
    default = (config.POINT_FULL_THRESHOLD, config.POINT_HALF_THRESHOLD)
    for fname in files:
        script = scripts[fname]
        res_default_obj = _grade(schema, script, default, use_nli=False)
        res_nli_obj = _grade(schema, script, default, use_nli=True)
        res_tuned_obj = _grade(schema, script, tuned, use_nli=False)
        res_default = {q.label: q for q in res_default_obj.questions}
        res_nli = {q.label: q for q in res_nli_obj.questions}
        res_tuned = {q.label: q for q in res_tuned_obj.questions}

        for (f, label), human in marks.items():
            if f != fname:
                continue
            qt = _find(res_tuned, label)
            qd = _find(res_default, label)
            qn = _find(res_nli, label)
            if qt is None:
                continue
            flagged = bool([fl for fl in qt.flags if fl != "Not answered."])
            rows.append({
                "subject": subject, "file": fname, "question": label, "qtype": qt.qtype,
                "method": qt.method, "max_marks": qt.max_marks, "human": human,
                "pointwise_default": qd.suggested, "pointwise_nli": qn.suggested,
                "pointwise_tuned": qt.suggested,
                "baseline": qt.baseline, "dnn": qt.dnn_marks, "flagged": flagged,
                "flags_tuned": " | ".join(qt.flags), "flags_nli": " | ".join(qn.flags),
            })

        # Script totals: for an OR pair the better-marked option counts (as a teacher would)
        human_total, groups = 0.0, {}
        for (f, label), human in marks.items():
            if f == fname:
                groups.setdefault(base_label(label), []).append(human)
        human_total = sum(max(v) for v in groups.values())
        totals.append({"subject": subject, "file": fname, "human_total": human_total,
                       "system_total": res_tuned_obj.total_suggested,
                       "system_total_default": res_default_obj.total_suggested,
                       "system_total_nli": res_nli_obj.total_suggested, "max": res_tuned_obj.total_max})
    return rows, totals


def metrics_table(df: pd.DataFrame, title: str) -> list[dict]:
    print(f"\n=== {title} ({len(df)} answers) ===")
    out = []
    methods = [("baseline", "Cosine baseline"), ("dnn", "DNN"),
               ("pointwise_default", "Point-wise (default thresholds)"),
               ("pointwise_nli", "Point-wise + NLI"),
               ("pointwise_tuned", "Point-wise (tuned thresholds)")]
    for col, name in methods:
        sub = df.dropna(subset=[col])
        if len(sub) < 2:
            continue
        m = evaluate(sub["human"], sub[col])
        m["MAE_pct_of_max"] = float((abs(sub[col] - sub["human"]) / sub["max_marks"]).mean())
        print(format_row(name, m) + f"  error {m['MAE_pct_of_max']:.0%} of max")
        out.append({"set": title, "method": name, **m})
    return out


# ---------------------------------------------------------------- flag kinds
FLAG_KINDS = ["Second opinion differs", "Very short answer", "No marking point matched",
              "may be contradicted", "Labelled Q", "Both OR options answered", "Empty answer"]


def flag_kind(flag: str) -> str:
    """Group a flag message by its kind, e.g. all 'Labelled Q4 ...' flags -> 'Labelled Q'."""
    return next((k for k in FLAG_KINDS if k in flag), "Other: " + flag[:40])


def flag_kind_counts(flag_cells: pd.Series) -> pd.Series:
    """Number of answers with at least one flag of each kind."""
    kinds = flag_cells.fillna("").map(lambda cell: {flag_kind(f) for f in cell.split(" | ") if f})
    return pd.Series([k for ks in kinds for k in ks], dtype=object).value_counts()


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-share", type=float, default=0.35, help="share of Science scripts used for tuning")
    args = ap.parse_args()

    data = {}
    for subject in SUBJECTS:
        if (config.CBSE_DIR / subject).exists():
            data[subject] = load_subject(subject)
    if "science" not in data:
        sys.exit("data/cbse/science not found.")

    # 1. split Science into tuning / test scripts
    sci_schema, sci_scripts, sci_marks = data["science"]
    files = sorted(sci_scripts)
    dev = sorted(random.Random(config.RANDOM_SEED).sample(files, max(2, round(len(files) * args.dev_share))))
    test = [f for f in files if f not in dev]
    print(f"Science: {len(files)} scripts -> tuning {len(dev)}, testing {len(test)}")
    print("  tuning scripts:", ", ".join(dev))

    # 2. tune thresholds on the tuning scripts only
    results, n_items = tune(sci_schema, sci_scripts, sci_marks, dev)
    best_mae, full_th, half_th = results[0]
    print(f"\nTuned on {n_items} descriptive answers: full >= {full_th}, half >= {half_th} (MAE {best_mae:.3f})")
    print(f"Default in config.py: full >= {config.POINT_FULL_THRESHOLD}, half >= {config.POINT_HALF_THRESHOLD}")

    # 3. grade the test sets
    all_rows, all_totals = [], []
    r, t = grade_rows("science", sci_schema, sci_scripts, sci_marks, test, (full_th, half_th))
    all_rows += r
    all_totals += t
    if "social_science" in data:
        ss_schema, ss_scripts, ss_marks = data["social_science"]
        r, t = grade_rows("social_science", ss_schema, ss_scripts, ss_marks, sorted(ss_scripts), (full_th, half_th))
        all_rows += r
        all_totals += t

    df = pd.DataFrame(all_rows)
    tot = pd.DataFrame(all_totals)
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / "cbse_eval_answers.csv", index=False)
    tot.to_csv(REPORTS / "cbse_eval_totals.csv", index=False)

    # 4. per-answer metrics (descriptive answers the student actually wrote)
    summary = []
    desc = df[(df["qtype"] == "descriptive") & (df["method"] == "pointwise")]
    for subject in df["subject"].unique():
        label = "Science TEST scripts" if subject == "science" else "Social Science (unseen subject)"
        summary += metrics_table(desc[desc["subject"] == subject], f"{label}, descriptive answers")
    summary += metrics_table(desc, "ALL test answers, descriptive")

    allq = df.dropna(subset=["pointwise_tuned"])
    m = evaluate(allq["human"], allq["pointwise_tuned"])
    print(f"\nAll question types incl. MCQ and skipped ({len(allq)}): "
          f"MAE {m['MAE']:.3f}, within 0.5 mark {m['within_0.5']:.0%}")

    # 5. script totals
    print("\n=== Script totals (tuned thresholds) ===")
    print(f"{'file':28} {'human':>6} {'system':>7} {'diff':>6}   {'default':>7} {'+NLI':>6} {'change':>6}")
    for row in tot.itertuples():
        nli_change = row.system_total_nli - row.system_total_default
        print(f"{row.file:28} {row.human_total:6g} {row.system_total:7g} {row.system_total - row.human_total:+6g}"
              f"   {row.system_total_default:7g} {row.system_total_nli:6g} {nli_change:+6g}")
    diff = (tot["system_total"] - tot["human_total"]).abs()
    r_tot = np.corrcoef(tot["human_total"], tot["system_total"])[0, 1] if len(tot) > 2 else float("nan")
    print(f"Total-mark MAE {diff.mean():.2f} / 14, within 1 mark: {(diff <= 1).mean():.0%}, "
          f"within 2 marks: {(diff <= 2).mean():.0%}, Pearson r {r_tot:.3f}")
    summary.append({"set": "Script totals", "method": "Point-wise (tuned)", "MAE": diff.mean(),
                    "Pearson": r_tot, "within_1_mark": (diff <= 1).mean(), "n": len(tot)})

    # 6. do the flags point at the mistakes?
    err = (desc["pointwise_tuned"] - desc["human"]).abs()
    big = err >= 1
    f = desc["flagged"]
    print("\n=== Flags (descriptive answers) ===")
    print(f"Flagged: {f.sum()} of {len(desc)} ({f.mean():.0%})")
    if f.any() and (~f).any():
        print(f"Point-wise MAE on flagged answers: {err[f].mean():.3f} | on the rest: {err[~f].mean():.3f}")
    if big.any():
        print(f"Answers off by 1+ mark: {big.sum()}, of which flagged: {(big & f).sum()} ({(big & f).sum() / big.sum():.0%})")

    print("\n=== Flag kinds (descriptive answers; an answer counts once per kind) ===")
    kinds = pd.DataFrame({"tuned, NLI off": flag_kind_counts(desc["flags_tuned"]),
                          "default + NLI": flag_kind_counts(desc["flags_nli"])}).fillna(0).astype(int)
    kinds = kinds.sort_values("tuned, NLI off", ascending=False)
    print(f"{'kind':32} {'tuned, NLI off':>15} {'default + NLI':>14}")
    for kind, row in kinds.iterrows():
        print(f"{kind:32} {row['tuned, NLI off']:15d} {row['default + NLI']:14d}")

    pd.DataFrame(summary).to_csv(REPORTS / "cbse_eval_summary.csv", index=False)

    # 7. figure: human vs system totals
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for subject, marker in (("science", "o"), ("social_science", "s")):
        s = tot[tot["subject"] == subject]
        if len(s):
            ax.scatter(s["human_total"], s["system_total"], marker=marker, s=55,
                       label=subject.replace("_", " ").title())
    lim = [0, float(tot["max"].max())]
    ax.plot(lim, lim, "--", color="grey", label="perfect agreement")
    ax.set(xlim=lim, ylim=lim, xlabel="Reference total (human marking)", ylabel="System total (point-wise)",
           title="Script totals: system vs reference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "cbse_totals_scatter.png", dpi=150)

    print(f"\nSaved reports/cbse_eval_answers.csv, cbse_eval_totals.csv, cbse_eval_summary.csv "
          f"and reports/figures/cbse_totals_scatter.png")
    print(f"\nTo use the tuned thresholds in the app, set in config.py:\n"
          f"  POINT_FULL_THRESHOLD = {full_th:.2f}\n  POINT_HALF_THRESHOLD = {half_th:.2f}")