"""Shared final-answer extraction for tree votes.

One implementation used by BOTH the scheduler runtime (majority-lock and
finalize votes) and the serving layer (branch_answers on the wire), so every
vote in the system is keyed identically.

Layered extraction, most reliable marker first:
1. the LAST ``\\boxed{...}`` (nested-brace aware) - math models' answer form;
2. the LAST ``#### <number>`` - GSM8K corpus form;
3. the LAST ``Answer: <...>`` line - our instructed form. Numeric content is
   canonicalized to a decimal string exactly as before; content carrying
   LaTeX markers (backslash, braces, ^ or _) is kept as a lightly normalized
   expression string so non-numeric answers (``p - q``, ``90^\\circ``) vote
   coherently instead of degrading to a stray digit;
4. the last number anywhere in the text.

Numeric answers always canonicalize through float ("18." == "18.0" == "18"),
which preserves the previous numeric-only behavior byte for byte.
"""

from __future__ import annotations

import re
from typing import Optional

_MARKED_RE = re.compile(r"####\s*([-+]?[\d.,]+)")
_ANSWER_LINE_RE = re.compile(r"[Aa]nswer\s*:\s*(.+)")
_FIRST_NUM_RE = re.compile(r"[-+]?[\d.,]*\d")
_LAST_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_LATEX_MARKERS = ("\\", "{", "}", "^", "_")
_MAX_EXPR_LEN = 80


def last_boxed_content(text: str) -> Optional[str]:
    key = "\\boxed{"
    start = text.rfind(key)
    if start < 0:
        return None
    i = start + len(key)
    depth = 1
    j = i
    while j < len(text):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[i:j]
        j += 1
    return None


def canonical_numeric(raw: str) -> Optional[str]:
    raw = raw.strip().lstrip("$").strip()
    raw = raw.replace(",", "").rstrip(".")
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return str(int(value)) if value == int(value) else str(value)


def canonical_answer(raw: str) -> Optional[str]:
    """Canonical vote key: decimal string for numbers, else a lightly
    normalized expression string (collapsed whitespace, trailing period and
    outer dollar signs stripped, length capped)."""
    raw = raw.strip().strip("$").strip()
    if not raw:
        return None
    numeric = canonical_numeric(raw)
    if numeric is not None:
        return numeric
    raw = " ".join(raw.split()).rstrip(".")
    return raw[:_MAX_EXPR_LEN] or None


def extract_answer_text(text: str) -> Optional[str]:
    boxed = last_boxed_content(text)
    if boxed is not None:
        answer = canonical_answer(boxed)
        if answer:
            return answer
    marked = _MARKED_RE.findall(text)
    if marked:
        answer = canonical_numeric(marked[-1])
        if answer:
            return answer
    lines = _ANSWER_LINE_RE.findall(text)
    if lines:
        # Cut trailing prose at the first sentence boundary; "3.5" survives
        # because the split needs a period FOLLOWED BY a space.
        capture = lines[-1].split(". ")[0].strip()
        if not any(marker in capture for marker in _LATEX_MARKERS):
            first_num = _FIRST_NUM_RE.search(capture)
            if first_num:
                answer = canonical_numeric(first_num.group(0))
                if answer:
                    return answer
        answer = canonical_answer(capture)
        if answer:
            return answer
    numbers = _LAST_NUM_RE.findall(text)
    if numbers:
        answer = canonical_numeric(numbers[-1])
        if answer:
            return answer
    return None
