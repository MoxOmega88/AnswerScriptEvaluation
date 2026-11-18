"""Agreement metrics between suggested marks and human marks."""
import numpy as np
from sklearn.metrics import cohen_kappa_score


def evaluate(y_true, y_pred) -> dict:
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    err = p - t
    pearson = float(np.corrcoef(t, p)[0, 1]) if t.std() > 0 and p.std() > 0 else float("nan")
    # QWK needs categories: use half-mark steps (0, 0.5, 1, ...) -> integers 0, 1, 2, ...
    t_bins = np.rint(t * 2).astype(int)
    p_bins = np.rint(p * 2).astype(int)
    try:
        qwk = float(cohen_kappa_score(t_bins, p_bins, weights="quadratic"))
    except ValueError:
        qwk = float("nan")
    return {
        "n": len(t),
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "Pearson": pearson,
        "QWK": qwk,
        "within_0.5": float((np.abs(err) <= 0.5 + 1e-9).mean()),
    }


def format_row(name: str, m: dict) -> str:
    return (f"{name:<34} MAE {m['MAE']:.3f}  RMSE {m['RMSE']:.3f}  "
            f"r {m['Pearson']:.3f}  QWK {m['QWK']:.3f}  within0.5 {m['within_0.5'] * 100:5.1f}%")
