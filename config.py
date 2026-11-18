"""Central settings for the project. Import from here instead of hard-coding paths."""
import os
from pathlib import Path

# Hide TensorFlow's noisy info logs (oneDNN / CPU instruction messages). Errors still show.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CBSE_DIR = DATA_DIR / "cbse"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
DB_PATH = ROOT / "answer_eval.db"

MOHLER_CSV = RAW_DIR / "mohler.csv"

# ---------- Models ----------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
DNN_MODEL_PATH = MODELS_DIR / "score_dnn.keras"

# ---------- Grading settings (tuned later on the CBSE dev split) ----------
SUMMARIZE_MIN_WORDS = 80     # only summarise answers longer than this
POINT_FULL_THRESHOLD = 0.55  # similarity >= this  -> full marks for the point
POINT_HALF_THRESHOLD = 0.20 # similarity >= this  -> half marks for the point
USE_KEYWORD_BACKUP = True  # point score = max(semantic similarity, keyword coverage)
DISAGREEMENT_FLAG = 0.30     # |point-wise - DNN| (as fraction of max marks) -> flag

RANDOM_SEED = 42

# Make sure output folders exist whenever config is imported
for _d in (RAW_DIR, PROCESSED_DIR, CBSE_DIR, SAMPLES_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
