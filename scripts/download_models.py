"""Download (and cache) the pre-trained models once, and run a tiny test on each.

The first run downloads ~1.5 GB in total, so use a good connection.
Usage:
    python scripts/download_models.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def nltk_data() -> None:
    import nltk

    print("[1/4] NLTK sentence tokenizer ...")
    for pkg in ("punkt", "punkt_tab"):
        nltk.download(pkg, quiet=True)
    from nltk.tokenize import sent_tokenize

    print("      ", sent_tokenize("Plants make food. They use sunlight."))


def embedder() -> None:
    from sentence_transformers import SentenceTransformer, util

    print(f"[2/4] Embedding model: {config.EMBED_MODEL} ...")
    t = time.time()
    model = SentenceTransformer(config.EMBED_MODEL)
    a, b, c = model.encode(
        [
            "Photosynthesis is how green plants make food using sunlight.",
            "Plants prepare their food with the help of light energy.",
            "Ohm's law relates voltage and current.",
        ],
        convert_to_tensor=True,
    )
    print(f"       loaded in {time.time() - t:.1f}s")
    print(f"       similar sentences   -> {util.cos_sim(a, b).item():.3f}  (should be high)")
    print(f"       unrelated sentences -> {util.cos_sim(a, c).item():.3f}  (should be low)")


def summarizer() -> None:
    from src.summarizer import summarize

    print(f"[3/4] Summarizer: {config.SUMMARIZER_MODEL} (largest download) ...")
    t = time.time()
    text = (
        "Photosynthesis is the process by which green plants prepare their own food. "
        "It takes place in the chloroplasts, which contain the green pigment chlorophyll. "
        "Chlorophyll absorbs sunlight, and this energy is used to convert carbon dioxide "
        "and water into glucose. Oxygen is released as a by-product. The glucose is used "
        "by the plant for energy and growth, and the extra is stored as starch. "
        "Without photosynthesis there would be no food for animals and no oxygen in the "
        "air, so almost all life on earth depends on this process happening in plants."
    )  # > 80 words, so it actually gets summarised
    out = summarize(text, max_length=50, min_length=15)
    print(f"       loaded + ran in {time.time() - t:.1f}s")
    print(f"       summary: {out}")


def nli_model() -> None:
    from src.nli import contradiction_probs

    print(f"[4/4] NLI contradiction model: {config.NLI_MODEL} ...")
    t = time.time()
    prob = contradiction_probs([(
        "The current is inversely proportional to the potential difference.",
        "Current is directly proportional to potential difference",
    )])[0]
    print(f"       loaded + ran in {time.time() - t:.1f}s")
    print(f"       'inversely' vs 'directly' -> P(contradiction) {prob:.3f}  (should be high)")


if __name__ == "__main__":
    nltk_data()
    embedder()
    summarizer()
    nli_model()
    print("All models ready.")