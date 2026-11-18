"""Checks that everything for Phase 1 is in place. Run this last.

Usage:
    python scripts/check_setup.py
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

ok = True


def check(label: str, passed: bool, hint: str = "") -> None:
    global ok
    print(f"  [{'OK' if passed else 'FAIL'}] {label}" + ("" if passed else f"  -> {hint}"))
    ok &= passed


print("Python")
check(f"version {sys.version.split()[0]}", sys.version_info[:2] >= (3, 10),
      "use Python 3.10 or newer")
# Newer Python is fine as long as TensorFlow imports - that is tested below.

print("Packages")
packages = ["numpy", "pandas", "sklearn", "matplotlib", "nltk", "torch", "transformers",
            "sentence_transformers", "datasets", "tensorflow", "docx", "pdfplumber",
            "fastapi", "uvicorn", "jinja2", "sqlalchemy"]
for name in packages:
    try:
        mod = importlib.import_module(name)
        check(f"{name} {getattr(mod, '__version__', '')}", True)
    except Exception as err:
        check(name, False, f"pip install -r requirements.txt ({err.__class__.__name__})")

print("TensorFlow sanity test")
try:
    import tensorflow as tf

    x = tf.constant([[1.0, 2.0]])
    model = tf.keras.Sequential([tf.keras.Input(shape=(2,)), tf.keras.layers.Dense(1)])
    check(f"tiny Keras model runs, output shape {tuple(model(x).shape)}", True)
except Exception as err:
    check("Keras model", False, str(err))

print("Data and folders")
check("data/raw/mohler.csv", config.MOHLER_CSV.exists(), "run scripts/download_data.py")
check("sample .docx files", any(config.SAMPLES_DIR.glob("*.docx")), "run scripts/make_samples.py")
for d in (config.CBSE_DIR / "schema", config.CBSE_DIR / "scripts", config.MODELS_DIR, config.FIGURES_DIR):
    check(str(d.relative_to(config.ROOT)), d.exists(), "folder missing")

print("\nAll good - ready for Phase 2!" if ok else "\nFix the FAIL lines above, then run again.")