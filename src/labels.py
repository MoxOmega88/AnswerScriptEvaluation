"""Detect and normalise question labels at the start of a line.

Students write the same label in many ways. All of these become the same key:
    "Q1 (a)", "Q1(a)", "1.a)", "1a.", "1 (a)", "Question 1a"  ->  "1a"
    "Q4-OR", "4 OR", "4 (OR)", "Q4(or)"                          ->  "4-OR"
    "2)", "Q2.", "Ans 2:", "Q3)"                                 ->  "2" / "3"
Sub-parts can be letters (a-h) or roman numerals (i-x).

To avoid false matches like "10 percent of energy..." or "2 molecules of...",
a line without a "Q"/"Ans" prefix only counts as a label if the number is followed
by a delimiter ( . ) : ), a sub-part, or OR.
"""
import re

_SUB = r"(?i:viii|vii|vi|iv|ix|iii|ii|x|v|i|[a-h])"
_OR = r"(?:\(\s*(?i:or)\s*\)|-\s*(?i:or)\b|\bOR\b)"
_PREFIX = r"(?i:q(?:ues(?:tion)?|ue)?|ans(?:wer)?)"

_LABEL_RE = re.compile(
    rf"""^\s*
    (?P<prefix>{_PREFIX}\s*[.\-:]?\s*)?          # optional Q / Question / Ans
    (?P<num>\d{{1,2}})                            # question number
    (?:
        (?P<sub_attached>{_SUB})(?![A-Za-z])         # 1a  (letter stuck to number)
      | \s*[.)\:]?\s*[(\[]\s*(?P<sub_paren>{_SUB})\s*[)\]]   # 1 (a) / 1.(a) / 1[a]
      | \s*[.]\s*(?P<sub_dot>{_SUB})\s*[.)\:]     # 1.a)  / 1.a.
    )?
    (?P<delim>\s*[.)\:](?!\d))?                   # 1.  1)  1:   (but not 1.5 or 2:1)
    \s*(?P<or>-?\s*{_OR})?                        # OR / (OR) / -OR
    \s*[.)\:\-–]*\s*
    (?P<rest>.*)$""",
    re.VERBOSE,  # case-insensitive only where marked (?i:...), so "4 or 5" is not an OR label
)


def parse_label(line: str) -> tuple[str, str] | None:
    """Return (normalised_label, rest_of_line) if the line starts with a question label, else None."""
    m = _LABEL_RE.match(line)
    if not m:
        return None

    sub = (m["sub_attached"] or m["sub_paren"] or m["sub_dot"] or "").lower()
    has_or = bool(m["or"])

    # Without a Q/Ans prefix we need some sign that this is really a label
    if not m["prefix"] and not (sub or has_or or m["delim"]):
        return None

    label = f"{int(m['num'])}{sub}" + ("-OR" if has_or else "")
    return label, m["rest"].strip()


def normalise(label: str) -> str:
    """Normalise a label written by itself, e.g. 'Q1 (a)' -> '1a'. Returns the input unchanged if not a label."""
    parsed = parse_label(label + " ")
    return parsed[0] if parsed else label.strip()


def base_label(label: str) -> str:
    """'4-OR' -> '4'  (the question that an OR alternative replaces)."""
    return label[:-3] if label.endswith("-OR") else label
