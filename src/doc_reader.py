"""Read .docx, .pdf or .txt files into a list of text lines."""
from pathlib import Path

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

        lines = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                lines.extend((page.extract_text() or "").splitlines())

    elif ext == ".txt":
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use one of: {', '.join(sorted(SUPPORTED))}")

    return [" ".join(line.split()) for line in lines if line.strip()]
