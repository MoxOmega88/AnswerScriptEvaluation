"""Choose the disagreement-flag threshold: for each candidate, how many answers get flagged,
and how many of point-wise marking's big mistakes do those flags catch?

Needs models/score_dnn.keras (train_dnn.py) and data/processed/mohler_features.csv.
Usage:
    python scripts/analyse_flags.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import dnn  # noqa: E402
from src.features import FEATURE_NAMES  # noqa: E402

df = pd.read_csv(config.PROCESSED_DIR / "mohler_features.csv")
test = df[df["split"] == "test"].reset_index(drop=True)

dnn_marks = np.floor(dnn.predict_norm(test[FEATURE_NAMES].to_dict("records")) * 5 * 2 + 0.5) / 2
pw = test["pointwise"].to_numpy()
human = test["score_avg"].to_numpy()

diff = np.abs(pw - dnn_marks) / 5            # disagreement as a fraction of max marks
err = np.abs(pw - human)
big = err >= 1.5                              # point-wise is off by 1.5+ marks

print(f"Test answers: {len(test)} | point-wise big mistakes (>= 1.5 marks off): {big.sum()}")
print(f"Average point-wise error overall: {err.mean():.3f}\n")
print(f"{'flag if diff >=':<16}{'flagged':>9}{'% flagged':>11}{'MAE flagged':>13}"
      f"{'MAE others':>12}{'big mistakes caught':>21}")
for th in (0.10, 0.15, 0.20, 0.25, 0.30):
    f = diff >= th - 1e-9
    mae_f = err[f].mean() if f.any() else float("nan")
    mae_o = err[~f].mean() if (~f).any() else float("nan")
    caught = (f & big).sum() / big.sum() if big.any() else float("nan")
    print(f"{th:<16.2f}{f.sum():>9}{f.mean():>10.1%}{mae_f:>13.3f}{mae_o:>12.3f}{caught:>20.1%}")

print("\nPick the threshold where roughly 10-25% of answers are flagged, the flagged MAE is")
print("clearly higher than the others, and a good share of big mistakes is caught.")
print("Then set DISAGREEMENT_FLAG in config.py to that value.")