"""Phase 4: train the Keras DNN on Mohler features and compare it with the other methods.

Needs data/processed/mohler_features.csv (created by scripts/eval_mohler.py).
Usage:
    python scripts/train_dnn.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

from src.dnn import build_model  # noqa: E402
from src.features import FEATURE_NAMES  # noqa: E402
from src.metrics import evaluate, format_row  # noqa: E402

FEATURES_CSV = config.PROCESSED_DIR / "mohler_features.csv"
RESULTS_CSV = config.ROOT / "reports" / "mohler_phase4_results.csv"


def round_half_arr(x):
    return np.floor(np.asarray(x) * 2 + 0.5) / 2


if not FEATURES_CSV.exists():
    sys.exit("mohler_features.csv not found. Run: python scripts/eval_mohler.py")

df = pd.read_csv(FEATURES_CSV)
train_all = df[df["split"] == "train"].reset_index(drop=True)
test = df[df["split"] == "test"].reset_index(drop=True)

# Validation set = some TRAIN questions held out (used only for early stopping)
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=config.RANDOM_SEED)
tr_idx, val_idx = next(gss.split(train_all, groups=train_all["question"]))
tr, val = train_all.iloc[tr_idx], train_all.iloc[val_idx]


def xy(part):
    return part[FEATURE_NAMES].to_numpy(np.float32), part["score_norm"].to_numpy(np.float32)


x_tr, y_tr = xy(tr)
x_val, y_val = xy(val)
x_test, _ = xy(test)
print(f"Train {len(tr)} | validation {len(val)} | test {len(test)} answers")

# ---------------- train ----------------
tf.keras.utils.set_random_seed(config.RANDOM_SEED)
model = build_model(x_tr)
model.summary()
history = model.fit(
    x_tr, y_tr,
    validation_data=(x_val, y_val),
    epochs=300, batch_size=32, verbose=0,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25,
                                                restore_best_weights=True)],
)
best_epoch = int(np.argmin(history.history["val_loss"])) + 1
print(f"Stopped after {len(history.history['loss'])} epochs (best epoch {best_epoch})")
model.save(config.DNN_MODEL_PATH)
print(f"Model saved -> {config.DNN_MODEL_PATH.relative_to(config.ROOT)}")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(history.history["loss"], label="train")
ax.plot(history.history["val_loss"], label="validation")
ax.axvline(best_epoch - 1, color="grey", linestyle="--", label="best epoch")
ax.set(title="DNN training curve (MSE on normalised score)", xlabel="Epoch", ylabel="Loss")
ax.legend()
fig.tight_layout()
fig.savefig(config.FIGURES_DIR / "dnn_training_curve.png", dpi=150)

# ---------------- compare on the test split ----------------
y_test = test["score_avg"].to_numpy()
dnn_pred = np.asarray(model(x_test, training=False)).ravel() * 5
lin_cos = LinearRegression().fit(train_all[["sim_model"]], train_all["score_avg"])
lin_all = LinearRegression().fit(train_all[FEATURE_NAMES], train_all["score_avg"])

preds = {
    "Always predict train mean": np.full(len(test), train_all["score_avg"].mean()),
    "Cosine + linear calibration": np.clip(lin_cos.predict(test[["sim_model"]]), 0, 5),
    "Linear regression (6 features)": np.clip(lin_all.predict(test[FEATURE_NAMES]), 0, 5),
    "Point-wise (ours)": test["pointwise"].to_numpy(),
    "DNN (ours)": dnn_pred,
    "DNN rounded to 0.5": round_half_arr(dnn_pred),
}

print(f"\n=== Mohler test split ({len(test)} answers, {test['question'].nunique()} unseen questions) ===")
rows = []
for name, p in preds.items():
    m = evaluate(y_test, p)
    print(format_row(name, m))
    rows.append({"method": name, **m})
pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
print(f"Results saved -> {RESULTS_CSV.relative_to(config.ROOT)}")

# ---------------- does the disagreement flag find the risky answers? ----------------
pw = test["pointwise"].to_numpy()
dnn_r = round_half_arr(dnn_pred)
flagged = np.abs(pw - dnn_r) / 5 > config.DISAGREEMENT_FLAG
err_pw = np.abs(pw - y_test)
print(f"\n=== Disagreement flag (|point-wise - DNN| > {config.DISAGREEMENT_FLAG:.0%} of max marks) ===")
print(f"Flagged: {flagged.sum()} of {len(test)} answers ({flagged.mean():.1%})")
if flagged.any() and (~flagged).any():
    print(f"Point-wise MAE on flagged answers:     {err_pw[flagged].mean():.3f}")
    print(f"Point-wise MAE on non-flagged answers: {err_pw[~flagged].mean():.3f}")
    print("(If flagged MAE is clearly higher, the flag is pointing the teacher at the right answers.)")