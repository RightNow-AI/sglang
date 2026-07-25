"""Strict answer graders for fixture and future benchmark tasks."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
import re

from .models import Task

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


def grade_exact(expected: str, response: str) -> bool:
    """Compare exact answers after removing surrounding whitespace only."""

    return response.strip() == expected.strip()


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


def _final_numeric_token(text: str) -> str | None:
    boxed = _boxed_contents(text)
    if boxed:
        tokens = _numeric_tokens(boxed[-1])
        return tokens[-1][1] if tokens else None

    markers = list(_FINAL_MARKER_RE.finditer(text))
    search_text = text[markers[-1].end() :] if markers else text
    tokens = _numeric_tokens(search_text)
    return tokens[-1][1] if tokens else None


def _parse_number(token: str) -> float | None:
    normalized = token.strip().replace(",", "").replace(" ", "")
    percent = normalized.endswith("%")
    if percent:
        normalized = normalized[:-1]
    try:
        if "/" in normalized:
            numerator, denominator = normalized.split("/", maxsplit=1)
            value = float(Decimal(numerator) / Decimal(denominator))
        else:
            value = float(Decimal(normalized))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return value / 100 if percent else value


def grade_numeric(
    expected: str,
    response: str,
    *,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-9,
) -> bool:
    """Compare the expected value with the response's final numeric answer.

    A final ``\\boxed{...}`` value wins, followed by the last number after a
    final-answer marker, followed by the last number in the response.
    """

    expected_token = _final_numeric_token(expected)
    response_token = _final_numeric_token(response)
    if expected_token is None or response_token is None:
        return False
    expected_value = _parse_number(expected_token)
    response_value = _parse_number(response_token)
    if expected_value is None or response_value is None:
        return False
    return math.isclose(
        response_value,
        expected_value,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    )


def grade_task(task: Task, response: str) -> bool:
    """Dispatch to the grader declared by a task."""

    if task.grader == "exact-match":
        return grade_exact(task.answer, response)
    return grade_numeric(task.answer, response)
