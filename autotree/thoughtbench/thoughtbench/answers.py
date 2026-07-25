"""Answer extraction, normalization, grading, and majority voting."""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable

_FINAL_MARKER_RE = re.compile(
    r"(?:final\s+answer|answer\s+is|answer)\s*[:=]?",
    flags=re.IGNORECASE,
)
_LATEX_FRACTION_RE = re.compile(
    r"(?P<sign>[+-]?)\\frac\s*\{(?P<num>\d+(?:\.\d+)?)\}"
    r"\s*\{(?P<den>\d+(?:\.\d+)?)\}"
)
_NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    r"(?:[eE][+-]?\d+)?(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?%?"
)


def _numeric_tokens(text: str) -> list[tuple[int, str]]:
    fractions = list(_LATEX_FRACTION_RE.finditer(text))
    spans = [match.span() for match in fractions]
    tokens = [
        (match.start(), match.group(0))
        for match in _NUMBER_RE.finditer(text)
        if not any(start <= match.start() < end for start, end in spans)
    ]
    for match in fractions:
        sign = match.group("sign") or ""
        tokens.append((match.start(), f"{sign}{match.group('num')}/{match.group('den')}"))
    return sorted(tokens, key=lambda item: item[0])


def _boxed_contents(text: str) -> list[str]:
    contents: list[str] = []
    cursor = 0
    while (start := text.find(r"\boxed", cursor)) >= 0:
        brace = text.find("{", start + len(r"\boxed"))
        if brace < 0:
            break
        depth = 0
        for index in range(brace, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    contents.append(text[brace + 1 : index])
                    cursor = index + 1
                    break
        else:
            break
    return contents


def normalize_answer(answer: str) -> str:
    """Apply the same comma and terminal-period normalization as the engine."""

    return answer.strip().replace(",", "").removesuffix(".").strip()


def extract_answer(text: str) -> str | None:
    """Extract a numeric answer using boxed, ####, marker, then last-number order."""

    boxed = _boxed_contents(text)
    if boxed:
        tokens = _numeric_tokens(boxed[-1])
        return normalize_answer(tokens[-1][1]) if tokens else None
    if "####" in text:
        tokens = _numeric_tokens(text.rsplit("####", maxsplit=1)[1])
        return normalize_answer(tokens[-1][1]) if tokens else None
    markers = list(_FINAL_MARKER_RE.finditer(text))
    search_text = text[markers[-1].end() :] if markers else text
    tokens = _numeric_tokens(search_text)
    return normalize_answer(tokens[-1][1]) if tokens else None


def grade_answer(response: str, gold: str) -> tuple[str | None, bool]:
    answer = extract_answer(response)
    normalized_gold = extract_answer(gold) or normalize_answer(gold)
    return answer, answer is not None and answer == normalized_gold


def majority_vote(responses: Iterable[str]) -> str | None:
    """Return the first-seen normalized answer among the tied top vote count."""

    answers = [extract_answer(response) for response in responses]
    present = [answer for answer in answers if answer is not None]
    if not present:
        return None
    counts = Counter(present)
    winning_count = max(counts.values())
    return next(answer for answer in present if counts[answer] == winning_count)
