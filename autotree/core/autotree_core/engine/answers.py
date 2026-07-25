"""Final-answer extraction for branch agreement scoring."""

from __future__ import annotations

import re

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
    fraction_spans = [match.span() for match in fractions]
    tokens = [
        (match.start(), match.group(0))
        for match in _NUMBER_RE.finditer(text)
        if not any(start <= match.start() < end for start, end in fraction_spans)
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


def _normalize_answer(answer: str) -> str:
    return answer.strip().replace(",", "").removesuffix(".")


def extract_final_answer(text: str) -> str | None:
    """Return the final numeric answer, preferring the last boxed value."""

    boxed = _boxed_contents(text)
    if boxed:
        tokens = _numeric_tokens(boxed[-1])
        return _normalize_answer(tokens[-1][1]) if tokens else None

    markers = list(_FINAL_MARKER_RE.finditer(text))
    search_text = text[markers[-1].end() :] if markers else text
    tokens = _numeric_tokens(search_text)
    return _normalize_answer(tokens[-1][1]) if tokens else None
