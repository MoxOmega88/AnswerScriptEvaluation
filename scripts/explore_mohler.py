"""Quick exploration of the Mohler dataset. Prints stats and saves figures for the paper.

Usage:
    python scripts/explore_mohler.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

if not config.MOHLER_CSV.exists():
    sys.exit("mohler.csv not found. Run: python scripts/download_data.py")

df = pd.read_csv(config.MOHLER_CSV)
df["student_words"] = df["student_answer"].str.split().str.len()
df["ref_words"] = df["instructor_answer"].str.split().str.len()

print("=== Mohler ASAG overview ===")
print(f"Answers:            {len(df)}")
print(f"Unique questions:   {df['question'].nunique()}")
print(f"Answers / question: {len(df) / df['question'].nunique():.1f} on average")
print()
print("Score (avg of 2 graders, 0-5):")
print(df["score_avg"].describe().round(2).to_string())
print()
print("Student answer length (words):")
print(df["student_words"].describe().round(1).to_string())
long_share = (df["student_words"] > config.SUMMARIZE_MIN_WORDS).mean() * 100
print(f"Answers longer than {config.SUMMARIZE_MIN_WORDS} words (would be summarised): {long_share:.1f}%")
print()

# Human-human agreement = a realistic upper bound for any automatic grader
if {"score_grader_1", "score_grader_2"} <= set(df.columns):
    r = df["score_grader_1"].corr(df["score_grader_2"])
    mae = (df["score_grader_1"] - df["score_grader_2"]).abs().mean()
    print(f"Grader 1 vs Grader 2: Pearson r = {r:.3f}, MAE = {mae:.3f}")
    print("(Use this as the 'human agreement' reference line in your results.)")

# ---- Figures ----
fig, ax = plt.subplots(figsize=(6, 4))
df["score_avg"].plot.hist(bins=20, ax=ax, edgecolor="black")
ax.set(title="Mohler: distribution of scores", xlabel="Average score (0-5)", ylabel="Answers")
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "mohler_score_hist.png", dpi=150)

fig, ax = plt.subplots(figsize=(6, 4))
df["student_words"].clip(upper=150).plot.hist(bins=30, ax=ax, edgecolor="black")
ax.axvline(config.SUMMARIZE_MIN_WORDS, color="red", linestyle="--", label="summarise threshold")
ax.set(title="Mohler: student answer length", xlabel="Words (clipped at 150)", ylabel="Answers")
ax.legend()
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "mohler_length_hist.png", dpi=150)

print(f"\nFigures saved in {config.FIGURES_DIR}")
