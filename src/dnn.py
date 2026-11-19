"""Keras DNN that predicts a holistic score (0-1) from the 6 answer features.

Trained on Mohler in scripts/train_dnn.py. In the app it is a *second opinion*:
if it disagrees strongly with point-wise marking, the answer is flagged for the teacher.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src.features import FEATURE_NAMES  # noqa: E402

_model = None


def build_model(train_x: np.ndarray):
    """Small MLP. The Normalization layer learns feature means/variances from the training
    data and is saved inside the model, so no separate scaler file is needed."""
    import tensorflow as tf

    norm = tf.keras.layers.Normalization()
    norm.adapt(train_x)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(len(FEATURE_NAMES),)),
        norm,
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),   # output in 0-1 = fraction of max marks
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def is_available() -> bool:
    return getattr(config, "USE_DNN", True) and config.DNN_MODEL_PATH.exists()


def _load():
    global _model
    if _model is None:
        import tensorflow as tf

        _model = tf.keras.models.load_model(config.DNN_MODEL_PATH)
    return _model


def predict_norm(feature_rows: list[dict]) -> np.ndarray:
    """Predicted score as a fraction of max marks (0-1), one value per feature dict."""
    x = np.array([[row[name] for name in FEATURE_NAMES] for row in feature_rows], dtype=np.float32)
    return np.asarray(_load()(x, training=False)).ravel()