"""Download the Mohler ASAG dataset from Hugging Face and save it as CSV.

Usage (from the project root, with the venv active):
    python scripts/download_data.py
    python scripts/download_data.py --asag2024     # also fetch the optional ASAG2024 set
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `import config` works
import config  # noqa: E402

from datasets import load_dataset  # noqa: E402


def download_mohler() -> None:
    print("Downloading Mohler ASAG (Viswa45/ASAG) ...")
    try:
        ds = load_dataset("Viswa45/ASAG", "cleaned", split="open_ended")
        version = "cleaned"
    except Exception as err:  # fall back to the default config if 'cleaned' fails
        print(f"  'cleaned' config failed ({err}); trying 'raw' ...")
        ds = load_dataset("Viswa45/ASAG", "raw", split="open_ended")
        version = "raw"

    df = ds.to_pandas()
    df = df.dropna(subset=["student_answer", "instructor_answer", "score_avg"])
    df["student_answer"] = df["student_answer"].astype(str).str.strip()
    df = df[df["student_answer"] != ""]
    df["score_norm"] = df["score_avg"] / 5.0  # 0-5 -> 0-1, so it can scale to any max marks

    df.to_csv(config.MOHLER_CSV, index=False)
    print(f"  Saved {len(df)} rows ({version}) -> {config.MOHLER_CSV}")
    print(f"  Columns: {list(df.columns)}")


def download_asag2024() -> None:
    print("Downloading ASAG2024 (Meyerger/ASAG2024) ...")
    ds = load_dataset("Meyerger/ASAG2024")
    for split_name, split in ds.items():
        out = config.RAW_DIR / f"asag2024_{split_name}.csv"
        split.to_pandas().to_csv(out, index=False)
        print(f"  Saved {len(split)} rows -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asag2024", action="store_true", help="also download ASAG2024")
    args = parser.parse_args()

    download_mohler()
    if args.asag2024:
        download_asag2024()
    print("Done.")
