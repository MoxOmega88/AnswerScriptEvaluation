"""Read .docx, .pdf or .txt files into a list of text lines."""
import re
from collections import Counter
from pathlib import Path

from src.labels import parse_label

SUPPORTED = {".docx", ".pdf", ".txt"}


def read_lines(path: str | Path) -> list[str]:
    """Return the document's text as a list of lines (blank lines removed, whitespace trimmed)."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".docx":
        from docx import Document

        doc = Document(str(path))
        raw = [p.text for p in doc.paragraphs]
        # Text typed inside tables (some students do this) - read row by row
        for table in doc.tables:
            for row in table.rows:
                raw.append(" ".join(cell.text for cell in row.cells))
        lines = []
        for para in raw:  # a paragraph can contain manual line breaks
            lines.extend(para.splitlines())

    elif ext == ".pdf":
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            pages = [[ln for ln in (page.extract_text() or "").splitlines() if ln.strip()]
                     for page in pdf.pages]
        lines = remove_headers_footers(pages)

    elif ext == ".txt":
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(SUPPORTED))}")

    return [" ".join(line.split()) for line in lines if line.strip()]


# ---------------------------------------------------------------- PDF clean-up
_PAGE_NO_RE = re.compile(r"^\W*(?:page\s*)?\d{1,3}(?:\s*(?:of|/)\s*\d{1,3})?\W*$", re.IGNORECASE)


def _shape(line: str) -> str:
    """'Science - Luffy 2' and 'Science - Luffy 1' have the same shape."""
    return re.sub(r"\d+", "#", " ".join(line.lower().split()))


def remove_headers_footers(pages: list[list[str]]) -> list[str]:
    """Drop page numbers, and lines repeated at the top or bottom of several pages
    (headers/footers such as 'Science - Luffy 1'). Question labels are never dropped."""
    def edges(lines):
        return set(range(min(2, len(lines)))) | set(range(max(0, len(lines) - 2), len(lines)))

    counts = Counter()
    for lines in pages:
        counts.update({_shape(lines[i]) for i in edges(lines)})
    repeated = {shape for shape, n in counts.items() if n >= 2} if len(pages) >= 2 else set()

    out = []
    for lines in pages:
        edge_idx = edges(lines)
        for i, line in enumerate(lines):
            if parse_label(line):                      # never drop a question label
                out.append(line)
            elif _PAGE_NO_RE.match(line):
                continue
            elif i in edge_idx and _shape(line) in repeated:
                continue
            else:
                out.append(line)
    return out