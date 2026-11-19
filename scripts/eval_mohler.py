"""Phase 3 evaluation on Mohler + feature extraction for the Phase 4 DNN.

Mohler has no marking scheme, so each sentence of the instructor's answer is treated
as one marking point (equal share of the 5 marks).

Compares, on a held-out test split (questions not seen in training):
  * "always predict the training average"   (trivial baseline)
  * cosine(answer, model answer) x 5         (simple baseline)
  * cosine + linear calibration on train     (fairer baseline)
  * point-wise marking (our method)

Usage:
    python scripts/eval_mohler.py
    python scripts/eval_mohler.py --limit 300     # quick run on a subset
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

from src.embedder import embed  # noqa: E402
from src.features import FEATURE_NAMES, compute_features, points_from_model_answer  # noqa: E402
from src.grader import apply_thresholds, point_scores  # noqa: E402
from src.metrics import evaluate, format_row  # noqa: E402
from src.text_utils import candidate_spans, round_half  # noqa: E402

FEATURES_CSV = config.PROCESSED_DIR / "mohler_features.csv"
RESULTS_CSV = config.ROOT / "reports" / "mohler_phase3_results.csv"


def main(csv_path: Path, limit: int | None) -> None:
    df = pd.read_csv(csv_path)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=config.RANDOM_SEED)
    df = df.reset_index(drop=True)
    print(f"Rows: {len(df)}  |  questions: {df['question'].nunique()}")

    # Split by question so test questions are never seen during training/tuning
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=config.RANDOM_SEED)
    train_idx, test_idx = next(gss.split(df, groups=df["question"]))
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    # Embed every text once up front (much faster than one row at a time)
    t = time.time()
    texts = set(df["student_answer"]) | set(df["instructor_answer"])
    for ans in df["student_answer"]:
        texts.update(candidate_spans(ans))
    for ref in df["instructor_answer"]:
        texts.update(p.text for p in points_from_model_answer(ref, 5))
    print(f"Embedding {len(texts)} unique texts ...")
    embed(sorted(texts))
    print(f"  done in {time.time() - t:.0f}s")

    rows = []
    for r in df.itertuples():
        points = points_from_model_answer(r.instructor_answer, 5.0)
        scores = point_scores(points, r.student_answer)
        pw = apply_thresholds(points, scores, config.MOHLER_FULL_THRESHOLD, config.MOHLER_HALF_THRESHOLD)
        feats = compute_features(r.student_answer, r.instructor_answer, points)
        rows.append({**feats,
                     "pointwise": min(round_half(sum(p.awarded for p in pw)), 5.0),
                     "n_points": len(points)})
    df = pd.concat([df, pd.DataFrame(rows)], axis=1)

    keep = ["id", "question", "split", "score_avg", "score_norm", "n_points", "pointwise"] + FEATURE_NAMES
    df[keep].to_csv(FEATURES_CSV, index=False)
    print(f"Features saved -> {FEATURES_CSV.relative_to(config.ROOT)}  (used in Phase 4)\n")

    train, test = df[df.split == "train"], df[df.split == "test"]
    y = test["score_avg"]

    lin = LinearRegression().fit(train[["sim_model"]], train["score_avg"])
    preds = {
        "Always predict train mean": np.full(len(test), train["score_avg"].mean()),
        "Cosine x 5 (baseline)": (5 * test["sim_model"].clip(0, 1)).map(round_half),
        "Cosine + linear calibration": np.clip(lin.predict(test[["sim_model"]]), 0, 5),
        "Point-wise (ours)": test["pointwise"],
    }

    print(f"=== Mohler test split ({len(test)} answers, {test['question'].nunique()} unseen questions) ===")
    print(f"thresholds: full >= {config.MOHLER_FULL_THRESHOLD}, half >= {config.MOHLER_HALF_THRESHOLD}, "
          f"keyword backup = {config.USE_KEYWORD_BACKUP}")
    out = []
    for name, p in preds.items():
        m = evaluate(y, p)
        print(format_row(name, m))
        out.append({"method": name, **m})
    if {"score_grader_1", "score_grader_2"} <= set(test.columns):
        m = evaluate(test["score_grader_1"], test["score_grader_2"])
        print(format_row("Human vs human (reference)", m))
        out.append({"method": "Human vs human", **m})

    pd.DataFrame(out).to_csv(RESULTS_CSV, index=False)
    print(f"\nResults saved -> {RESULTS_CSV.relative_to(config.ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.MOHLER_CSV))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(Path(args.csv), args.limit)
