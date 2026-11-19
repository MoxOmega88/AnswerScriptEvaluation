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

# ---------- Grading settings ----------
SUMMARIZE_MIN_WORDS = 80     # only summarise answers longer than this
SHOW_SUMMARIES = True        # add a short summary of long answers for the teacher (slower)

# Thresholds used by the app / CBSE grading. Re-tune on your marked CBSE scripts later.
POINT_FULL_THRESHOLD = 0.65  # point score >= this -> full marks for the point
POINT_HALF_THRESHOLD = 0.50  # point score >= this -> half marks for the point

# Thresholds used only for the Mohler experiments (tuned on Mohler's train split).
MOHLER_FULL_THRESHOLD = 0.55
MOHLER_HALF_THRESHOLD = 0.20

USE_KEYWORD_BACKUP = True    # point score = max(semantic similarity, keyword coverage)
USE_DNN = True               # use the trained DNN as a second opinion (if models/score_dnn.keras exists)
DISAGREEMENT_FLAG = 0.20     # |point-wise - DNN| (as fraction of max marks) -> flag


RANDOM_SEED = 42

# Make sure output folders exist whenever config is imported
for _d in (RAW_DIR, PROCESSED_DIR, CBSE_DIR, SAMPLES_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)