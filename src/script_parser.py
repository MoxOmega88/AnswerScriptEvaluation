"""Parse a student's answer script and map each answer to a question in the schema.

The student writes only question numbers ("Q1 (a)", "2)", "4 OR." ...) followed by the answer.
Everything after a label, up to the next label, is that answer.

Safety rules against false labels (e.g. a numbered list inside an answer):
  * a label only counts if it exists in the schema,
  * a label that was already used is treated as normal answer text, and
  * a plain "1." / "2)" right after a line ending in ":" (or after another list item)
    is a list item, not a new answer.
The teacher still sees a mapping preview, so any remaining mistakes can be spotted.
"""
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.doc_reader import read_lines
from src.labels import parse_label
from src.schema_parser import Schema

_NUMBER_ONLY_RE = re.compile(r"^\s*(?:(?i:q[a-z]*|ans[a-z]*)\s*[.\-:]?\s*)?(\d{1,2})\s*[.)\:]?\s*(.*)$")

_NAME_RE = re.compile(r"^\s*(?:student\s*)?name\s*[:\-]\s*(.+)$", re.IGNORECASE)
_ROLL_RE = re.compile(r"^\s*(?:roll|reg(?:istration)?|usn)\s*(?:no\.?|number)?\s*[:\-]\s*(\S+)", re.IGNORECASE)


@dataclass
class StudentScript:
    file_name: str
    name: str = ""
    roll_no: str = ""
    answers: dict[str, str] = field(default_factory=dict)        # schema label -> answer text
    unmatched: list[tuple[str, str]] = field(default_factory=list)  # labels not in the schema
    missing: list[str] = field(default_factory=list)              # schema questions not answered
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_script(path: str | Path, schema: Schema) -> StudentScript:
    path = Path(path)
    lines = read_lines(path)
    script = StudentScript(file_name=path.name)
    valid = set(schema.questions)

    current: str | None = None
    buffer: dict[str, list[str]] = {}
    prev_line = ""
    in_list = False

    for line in lines:
        parsed = parse_label(line)
        # "Q1) (b)" reads as "1b"; if the schema has no 1b but has a plain Q1,
        # the "(b)" is the student's answer, not a sub-part.
        if parsed and parsed[0] not in valid and (m := _NUMBER_ONLY_RE.match(line)) and m.group(1) in valid:
            parsed = (m.group(1), m.group(2).strip())
        label = parsed[0] if parsed else None

        # Numbered list inside an answer?  e.g. "Reasons:" followed by "1. ..." "2. ..."
        plain_number = bool(label) and label.isdigit() and line.lstrip()[:1].isdigit()
        if current and plain_number and (prev_line.rstrip().endswith(":") or in_list):
            buffer[current].append(line)
            in_list, prev_line = True, line
            continue
        in_list = False
        prev_line = line

        # A label the schema doesn't know: maybe "1" when the schema has "1a"/"1b",
        # or a typo. Record it so the teacher can see it in the mapping preview.
        if label and label not in valid and current is None:
            script.unmatched.append((label, parsed[1]))
            continue

        if label in valid and label not in buffer:
            current = label
            buffer[current] = [parsed[1]] if parsed[1] else []
            continue

        if current is None:
            # Header area before the first answer: name / roll number
            if m := _NAME_RE.match(line):
                script.name = m.group(1).strip()
            elif m := _ROLL_RE.match(line):
                script.roll_no = m.group(1).strip()
            continue

        if label and label not in valid and line.lstrip()[:1].lower() == "q":
            script.warnings.append(f"Found 'Q{label}' which is not in the schema; kept as part of Q{current}.")
        if label in buffer and label != current:
            script.warnings.append(
                f"'{line[:30]}...' looks like a second answer to Q{label}; kept as part of Q{current}."
            )
        buffer[current].append(line)

    script.answers = {lab: " ".join(parts).strip() for lab, parts in buffer.items()}

    # Missing questions (an OR pair counts as answered if either half was answered)
    for label in schema.questions:
        alts = schema.alternatives(label)
        answered = label in script.answers or any(a in script.answers for a in alts)
        if not answered and not schema.questions[label].is_alternative:
            script.missing.append(label)

    # Both halves of an OR pair answered -> teacher decides which one counts
    seen_bases = set()
    for label in script.answers:
        alts = [a for a in schema.alternatives(label) if a in script.answers]
        base = label.replace("-OR", "")
        if alts and base not in seen_bases:
            seen_bases.add(base)
            script.warnings.append(f"Answered both Q{base} and Q{base}-OR; teacher should choose which to mark.")

    for label, text in script.answers.items():
        if not text:
            script.warnings.append(f"Q{label}: label found but the answer is empty.")

    if not script.answers:
        script.warnings.append("No answers found. Check that answers start with question numbers like 'Q1' or '1)'.")
    return script
