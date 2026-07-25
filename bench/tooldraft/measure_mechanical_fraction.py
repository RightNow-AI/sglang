#!/usr/bin/env python3
"""Measure approximate mechanical-output span coverage in reasoning traces.

The detector is intentionally conservative. It counts only the output side of
an explicit deterministic computation. The expression or quantity selected by
the model is setup and remains REASONING. Token counts are approximations:

    max(whitespace-delimited chunks, ceil(non-whitespace characters / 4))

The script uses SymPy when it is importable. Without SymPy it falls back to a
small AST-based arithmetic evaluator; symbolic simplification detection is then
disabled rather than guessed.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import glob
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable


try:
    import sympy
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    HAVE_SYMPY = True
    SYMPY_VERSION = sympy.__version__
except ImportError:
    sympy = None
    HAVE_SYMPY = False
    SYMPY_VERSION = None


AUTO_TRACE_FIELDS = (
    "trace_text",
    "reasoning_text",
    "model_output_text",
    "output_text",
    "generated_text",
    "completion_text",
    "raw_output",
    "full_text",
    "reasoning",
    "trace",
    "completion",
    "model_output",
    "response_text",
    "sample_traces",
    "sample_texts",
    "branch_traces",
    "branch_texts",
    "traces",
)

ANSWER_ONLY_FIELDS = (
    "extracted",
    "gold",
    "sample_answers",
    "branch_answers",
    "branch_answer_leader",
    "small_winner_answer",
)

UNSIGNED_NUMBER = (
    r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?"
)
PLAIN_NUMBER = r"[+-]?" + UNSIGNED_NUMBER
LATEX_FRACTION = (
    r"[+-]?\\frac\s*\{\s*" + UNSIGNED_NUMBER
    + r"\s*\}\s*\{\s*" + UNSIGNED_NUMBER + r"\s*\}"
)
SIMPLE_FRACTION = PLAIN_NUMBER + r"\s*/\s*" + PLAIN_NUMBER
NUMERIC_VALUE = (
    r"(?:" + LATEX_FRACTION + r"|" + SIMPLE_FRACTION + r"|" + PLAIN_NUMBER + r")"
)

UNIT_ALIASES = {
    "mm": ("length", Fraction(1, 1000)),
    "millimeter": ("length", Fraction(1, 1000)),
    "millimeters": ("length", Fraction(1, 1000)),
    "cm": ("length", Fraction(1, 100)),
    "centimeter": ("length", Fraction(1, 100)),
    "centimeters": ("length", Fraction(1, 100)),
    "m": ("length", Fraction(1, 1)),
    "meter": ("length", Fraction(1, 1)),
    "meters": ("length", Fraction(1, 1)),
    "km": ("length", Fraction(1000, 1)),
    "kilometer": ("length", Fraction(1000, 1)),
    "kilometers": ("length", Fraction(1000, 1)),
    "in": ("length", Fraction(127, 5000)),
    "inch": ("length", Fraction(127, 5000)),
    "inches": ("length", Fraction(127, 5000)),
    "ft": ("length", Fraction(381, 1250)),
    "foot": ("length", Fraction(381, 1250)),
    "feet": ("length", Fraction(381, 1250)),
    "yd": ("length", Fraction(1143, 1250)),
    "yard": ("length", Fraction(1143, 1250)),
    "yards": ("length", Fraction(1143, 1250)),
    "mi": ("length", Fraction(201168, 125)),
    "mile": ("length", Fraction(201168, 125)),
    "miles": ("length", Fraction(201168, 125)),
    "ms": ("time", Fraction(1, 1000)),
    "millisecond": ("time", Fraction(1, 1000)),
    "milliseconds": ("time", Fraction(1, 1000)),
    "s": ("time", Fraction(1, 1)),
    "sec": ("time", Fraction(1, 1)),
    "second": ("time", Fraction(1, 1)),
    "seconds": ("time", Fraction(1, 1)),
    "min": ("time", Fraction(60, 1)),
    "minute": ("time", Fraction(60, 1)),
    "minutes": ("time", Fraction(60, 1)),
    "h": ("time", Fraction(3600, 1)),
    "hr": ("time", Fraction(3600, 1)),
    "hour": ("time", Fraction(3600, 1)),
    "hours": ("time", Fraction(3600, 1)),
    "mg": ("mass", Fraction(1, 1_000_000)),
    "milligram": ("mass", Fraction(1, 1_000_000)),
    "milligrams": ("mass", Fraction(1, 1_000_000)),
    "g": ("mass", Fraction(1, 1000)),
    "gram": ("mass", Fraction(1, 1000)),
    "grams": ("mass", Fraction(1, 1000)),
    "kg": ("mass", Fraction(1, 1)),
    "kilogram": ("mass", Fraction(1, 1)),
    "kilograms": ("mass", Fraction(1, 1)),
    "ml": ("volume", Fraction(1, 1000)),
    "milliliter": ("volume", Fraction(1, 1000)),
    "milliliters": ("volume", Fraction(1, 1000)),
    "l": ("volume", Fraction(1, 1)),
    "liter": ("volume", Fraction(1, 1)),
    "liters": ("volume", Fraction(1, 1)),
}

UNIT_PATTERN = "|".join(
    sorted((re.escape(unit) for unit in UNIT_ALIASES), key=len, reverse=True)
)
UNIT_EQUATION_RE = re.compile(
    rf"(?<![\w.])(?P<lhs_value>{NUMERIC_VALUE})\s*"
    rf"(?P<lhs_unit>{UNIT_PATTERN})\b\s*(?:=|->|→)\s*"
    rf"(?P<rhs_value>{NUMERIC_VALUE})\s*"
    rf"(?P<rhs_unit>{UNIT_PATTERN})\b",
    re.IGNORECASE,
)

ARITHMETIC_EQUATION_RE = re.compile(
    rf"(?P<lhs>(?<![\w])(?:\\(?:frac|times|cdot|div|left|right)|"
    rf"[0-9.,{{}}()\[\]\s+\-*/^%×÷·−])+?)"
    rf"\s*(?:=|->|→)\s*(?P<rhs>{NUMERIC_VALUE})",
    re.IGNORECASE,
)

RESTATEMENT_CONSERVATIVE_RE = re.compile(
    rf"\b(?:therefore|thus|hence|so)\b[^.\n;:]{{0,60}}?"
    rf"(?:\b(?:answer|value|result)\b|\b[A-Za-z]\b)\s*"
    rf"(?:is|=|:)\s*(?P<rhs>{NUMERIC_VALUE})",
    re.IGNORECASE,
)
RESTATEMENT_LIBERAL_RE = re.compile(
    rf"\b(?:therefore|thus|hence|so)\b[^.\n;:]{{0,80}}?"
    rf"(?:is|=|:)\s*(?P<rhs>{NUMERIC_VALUE})",
    re.IGNORECASE,
)

SYMBOLIC_CUE_RE = re.compile(
    r"\b(?:simplif(?:y|ies|ied|ication)|expand(?:s|ed|ing)?|"
    r"factor(?:s|ed|ing)?|reduc(?:e|es|ed|ing)|"
    r"evaluat(?:e|es|ed|ing|ion)|comput(?:e|es|ed|ing|ation))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Trace:
    problem_id: str
    source: str
    text: str


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    equation_start: int
    equation_end: int
    kind: str
    exact: bool
    expected: str
    reported: str
    numeric_expected: Any | None = None


@dataclass
class FileAudit:
    path: Path
    records: int = 0
    malformed: int = 0
    trace_values: int = 0
    keys: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.keys is None:
            self.keys = Counter()


@dataclass
class DetectorStats:
    variant: str
    problems: int
    traces: int
    total_tokens: int
    mechanical_tokens: int
    spans: int
    exact_spans: int
    wrong_spans: int
    fraction: float
    ci_low: float
    ci_high: float
    mean_length: float
    kind_counts: Counter[str]
    wrong_examples: list[tuple[Trace, Span]]


class ArithmeticEvaluator:
    def __init__(self) -> None:
        self.backend = (
            "sympy {}".format(SYMPY_VERSION)
            if HAVE_SYMPY
            else "stdlib AST rational evaluator"
        )

    def numeric(self, expression: str) -> Any:
        normalized = normalize_numeric_expression(expression)
        if HAVE_SYMPY:
            value = sympy.sympify(normalized, evaluate=True)
            if value.free_symbols or value.is_number is not True:
                raise ValueError("not a numeric expression")
            if value.is_finite is not True:
                raise ValueError("non-finite numeric expression")
            return value
        return evaluate_ast_fraction(normalized)

    def equal(self, left: Any, right: Any) -> bool:
        if HAVE_SYMPY:
            return bool(sympy.simplify(left - right) == 0)
        return left == right

    def multiply_fraction(self, value: Any, factor: Fraction) -> Any:
        if HAVE_SYMPY:
            return value * sympy.Rational(factor.numerator, factor.denominator)
        return value * factor

    def divide_fraction(self, value: Any, factor: Fraction) -> Any:
        if HAVE_SYMPY:
            return value / sympy.Rational(factor.numerator, factor.denominator)
        return value / factor

    def format(self, value: Any) -> str:
        if HAVE_SYMPY:
            return str(sympy.simplify(value))
        if isinstance(value, Fraction):
            if value.denominator == 1:
                return str(value.numerator)
            return "{}/{}".format(value.numerator, value.denominator)
        return str(value)

    def symbolic_equivalent(self, left: str, right: str) -> bool:
        if not HAVE_SYMPY:
            raise ValueError("symbolic verification requires SymPy")
        left_value = parse_symbolic_expression(left)
        right_value = parse_symbolic_expression(right)
        return bool(sympy.simplify(left_value - right_value) == 0)


def normalize_latex_fractions(text: str) -> str:
    pattern = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    previous = None
    current = text
    while previous != current:
        previous = current
        current = pattern.sub(r"((\1)/(\2))", current)
    return current


def normalize_numeric_expression(expression: str) -> str:
    value = expression.strip().strip("$")
    value = value.replace("−", "-").replace("×", "*").replace("÷", "/")
    value = value.replace("·", "*")
    value = re.sub(r"\\(?:times|cdot)", "*", value)
    value = re.sub(r"\\div", "/", value)
    value = re.sub(r"\\(?:left|right)", "", value)
    value = normalize_latex_fractions(value)
    value = value.replace("{", "(").replace("}", ")")
    value = re.sub(r"(?<=\d),(?=\d)", "", value)
    value = value.replace("^", "**")
    if "%" in value:
        raise ValueError("percent notation is deliberately unsupported")
    if not re.fullmatch(r"[0-9eE+\-*/().\s]+", value):
        raise ValueError("unsupported numeric syntax")
    if len(value) > 160 or "***" in value:
        raise ValueError("numeric expression outside safety limits")
    return value


def evaluate_ast_fraction(expression: str) -> Fraction:
    tree = ast.parse(expression, mode="eval")

    def visit(node: ast.AST, depth: int = 0) -> Fraction:
        if depth > 30:
            raise ValueError("expression nesting too deep")
        if isinstance(node, ast.Expression):
            return visit(node.body, depth + 1)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, int):
                return Fraction(node.value, 1)
            try:
                return Fraction(Decimal(str(node.value)))
            except InvalidOperation as exc:
                raise ValueError("invalid decimal") from exc
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand, depth + 1)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = visit(node.left, depth + 1)
            right = visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Mod):
                if left.denominator != 1 or right.denominator != 1:
                    raise ValueError("fractional modulo is unsupported")
                return Fraction(left.numerator % right.numerator, 1)
            if isinstance(node.op, ast.Pow):
                if right.denominator != 1 or abs(right.numerator) > 20:
                    raise ValueError("power outside safety limits")
                return left ** right.numerator
        raise ValueError("unsupported arithmetic AST node")

    return visit(tree)


def normalize_symbolic_expression(expression: str) -> str:
    value = expression.strip().strip("$")
    value = value.replace("−", "-").replace("×", "*").replace("÷", "/")
    value = value.replace("·", "*")
    value = re.sub(r"\\(?:times|cdot)", "*", value)
    value = re.sub(r"\\div", "/", value)
    value = re.sub(r"\\(?:left|right)", "", value)
    value = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", value)
    value = normalize_latex_fractions(value)
    value = value.replace("{", "(").replace("}", ")")
    value = re.sub(r"(?<=\d),(?=\d)", "", value)
    if len(value) > 160 or "__" in value:
        raise ValueError("symbolic expression outside safety limits")
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/^().,\s]+", value):
        raise ValueError("unsupported symbolic syntax")
    names = set(re.findall(r"[A-Za-z_]+", value))
    if any(name != "sqrt" and (len(name) != 1 or name == "_") for name in names):
        raise ValueError("only single-letter symbols and sqrt are supported")
    return value


def parse_symbolic_expression(expression: str) -> Any:
    value = normalize_symbolic_expression(expression)
    names = set(re.findall(r"[A-Za-z_]+", value))
    local_dict = {
        name: sympy.Symbol(name) for name in names if name != "sqrt"
    }
    local_dict["sqrt"] = sympy.sqrt
    global_dict = {
        "Integer": sympy.Integer,
        "Float": sympy.Float,
        "Rational": sympy.Rational,
        "Symbol": sympy.Symbol,
    }
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    return parse_expr(
        value,
        local_dict=local_dict,
        global_dict=global_dict,
        transformations=transformations,
        evaluate=True,
    )


def approximate_tokens(text: str) -> int:
    if not text or not text.strip():
        return 0
    whitespace_chunks = len(re.findall(r"\S+", text))
    non_whitespace_chars = len(re.sub(r"\s", "", text))
    char_estimate = math.ceil(non_whitespace_chars / 4)
    return max(whitespace_chunks, char_estimate)


def ranges_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def has_arithmetic_operation(expression: str) -> bool:
    try:
        normalized = normalize_numeric_expression(expression)
    except ValueError:
        return False
    unsigned = normalized.lstrip().lstrip("+-").strip()
    return any(operator in unsigned for operator in ("+", "-", "*", "/", "%"))


def make_numeric_span(
    evaluator: ArithmeticEvaluator,
    match: re.Match[str],
    kind: str,
    expected: Any,
    reported_text: str,
    output_start: int,
    output_end: int,
) -> Span | None:
    try:
        reported_value = evaluator.numeric(reported_text)
    except (ValueError, TypeError, ZeroDivisionError, SyntaxError):
        return None
    exact = evaluator.equal(expected, reported_value)
    return Span(
        start=output_start,
        end=output_end,
        equation_start=match.start(),
        equation_end=match.end(),
        kind=kind,
        exact=exact,
        expected=evaluator.format(expected),
        reported=reported_text.strip(),
        numeric_expected=expected,
    )


def detect_unit_conversions(
    text: str, evaluator: ArithmeticEvaluator
) -> list[Span]:
    spans = []
    for match in UNIT_EQUATION_RE.finditer(text):
        lhs_unit = match.group("lhs_unit").lower()
        rhs_unit = match.group("rhs_unit").lower()
        lhs_dimension, lhs_factor = UNIT_ALIASES[lhs_unit]
        rhs_dimension, rhs_factor = UNIT_ALIASES[rhs_unit]
        if lhs_dimension != rhs_dimension or lhs_unit == rhs_unit:
            continue
        try:
            lhs_value = evaluator.numeric(match.group("lhs_value"))
            expected = evaluator.divide_fraction(
                evaluator.multiply_fraction(lhs_value, lhs_factor), rhs_factor
            )
            reported_value_text = match.group("rhs_value")
            reported_value = evaluator.numeric(reported_value_text)
        except (ValueError, TypeError, ZeroDivisionError, SyntaxError):
            continue
        exact = evaluator.equal(expected, reported_value)
        spans.append(
            Span(
                start=match.start("rhs_value"),
                end=match.end("rhs_unit"),
                equation_start=match.start(),
                equation_end=match.end(),
                kind="unit_conversion",
                exact=exact,
                expected="{} {}".format(evaluator.format(expected), rhs_unit),
                reported=text[match.start("rhs_value"):match.end("rhs_unit")].strip(),
                numeric_expected=expected,
            )
        )
    return spans


def detect_arithmetic(
    text: str,
    evaluator: ArithmeticEvaluator,
    occupied_equations: list[tuple[int, int]],
) -> list[Span]:
    spans = []
    for match in ARITHMETIC_EQUATION_RE.finditer(text):
        equation_range = (match.start(), match.end())
        if any(ranges_overlap(equation_range, other) for other in occupied_equations):
            continue
        lhs = match.group("lhs").strip()
        if not has_arithmetic_operation(lhs):
            continue
        try:
            expected = evaluator.numeric(lhs)
        except (ValueError, TypeError, ZeroDivisionError, SyntaxError):
            continue
        span = make_numeric_span(
            evaluator,
            match,
            "arithmetic",
            expected,
            match.group("rhs"),
            match.start("rhs"),
            match.end("rhs"),
        )
        if span is not None:
            spans.append(span)
            occupied_equations.append(equation_range)
    return spans


def mathish_token(token: str) -> bool:
    stripped = token.strip().strip("$,.!?;:")
    if not stripped:
        return False
    stripped = re.sub(r"\\(?:frac|sqrt|times|cdot|div|left|right)", "", stripped)
    if not re.fullmatch(r"[A-Za-z0-9_{}()\[\]+\-*/^.=]+", stripped):
        return False
    names = re.findall(r"[A-Za-z_]+", stripped)
    return all(len(name) == 1 for name in names)


def math_suffix(raw: str) -> tuple[str, int] | None:
    tokens = list(re.finditer(r"\S+", raw))
    selected = []
    for token in reversed(tokens):
        if not mathish_token(token.group(0)):
            break
        selected.append(token)
    if not selected:
        return None
    selected.reverse()
    start = selected[0].start()
    value = raw[start:].strip().strip("$,.!?;:")
    if not value:
        return None
    exact_start = raw.find(value, start)
    return value, exact_start


def math_prefix(raw: str) -> tuple[str, int] | None:
    tokens = list(re.finditer(r"\S+", raw))
    selected = []
    for token in tokens:
        if not mathish_token(token.group(0)):
            break
        selected.append(token)
    if not selected:
        return None
    end = selected[-1].end()
    value = raw[:end].strip().strip("$,.!?;:")
    if not value:
        return None
    return value, raw.find(value)


def symbolic_candidates(text: str) -> Iterable[tuple[int, int, int, int, str, str]]:
    for equals in re.finditer(r"(?<![<>=])=(?!=)", text):
        window_start = max(0, equals.start() - 140)
        window_end = min(len(text), equals.end() + 140)
        left_raw = text[window_start:equals.start()]
        right_raw = text[equals.end():window_end]
        left = math_suffix(left_raw)
        right = math_prefix(right_raw)
        if left is None or right is None:
            continue
        lhs, lhs_relative = left
        rhs, rhs_relative = right
        lhs_start = window_start + lhs_relative
        rhs_start = equals.end() + rhs_relative
        rhs_end = rhs_start + len(rhs)
        yield lhs_start, rhs_end, rhs_start, rhs_end, lhs, rhs


def detect_symbolic(
    text: str,
    evaluator: ArithmeticEvaluator,
    occupied_equations: list[tuple[int, int]],
    liberal: bool,
) -> list[Span]:
    if not HAVE_SYMPY:
        return []
    spans = []
    for equation_start, equation_end, rhs_start, rhs_end, lhs, rhs in symbolic_candidates(text):
        equation_range = (equation_start, equation_end)
        if any(ranges_overlap(equation_range, other) for other in occupied_equations):
            continue
        if not re.search(r"[A-Za-z]", lhs + rhs):
            continue
        if not any(operator in lhs for operator in ("+", "-", "*", "/", "^", "(")):
            continue
        cue_window = text[max(0, equation_start - 100):equation_start]
        has_cue = bool(SYMBOLIC_CUE_RE.search(cue_window))
        if not has_cue and not liberal:
            continue
        try:
            exact = evaluator.symbolic_equivalent(lhs, rhs)
        except (ValueError, TypeError, SyntaxError):
            continue
        if not has_cue and not exact:
            # A bare, false equation may be a condition or setup. Do not label it.
            continue
        spans.append(
            Span(
                start=rhs_start,
                end=rhs_end,
                equation_start=equation_start,
                equation_end=equation_end,
                kind="symbolic_simplification",
                exact=exact,
                expected="equivalent to {}".format(lhs),
                reported=rhs,
                numeric_expected=None,
            )
        )
        occupied_equations.append(equation_range)
    return spans


def detect_restatements(
    text: str,
    evaluator: ArithmeticEvaluator,
    prior_spans: list[Span],
    liberal: bool,
) -> list[Span]:
    pattern = RESTATEMENT_LIBERAL_RE if liberal else RESTATEMENT_CONSERVATIVE_RE
    numeric_prior = [span for span in prior_spans if span.numeric_expected is not None]
    spans = []
    for match in pattern.finditer(text):
        if any(ranges_overlap((match.start("rhs"), match.end("rhs")), (s.start, s.end)) for s in prior_spans):
            continue
        eligible = [
            span
            for span in numeric_prior
            if span.equation_end <= match.start()
            and match.start() - span.equation_end <= 320
        ]
        if not eligible:
            continue
        source = max(eligible, key=lambda span: span.equation_end)
        try:
            reported_value = evaluator.numeric(match.group("rhs"))
        except (ValueError, TypeError, ZeroDivisionError, SyntaxError):
            continue
        exact = evaluator.equal(source.numeric_expected, reported_value)
        spans.append(
            Span(
                start=match.start("rhs"),
                end=match.end("rhs"),
                equation_start=match.start(),
                equation_end=match.end(),
                kind="numeric_restatement",
                exact=exact,
                expected=evaluator.format(source.numeric_expected),
                reported=match.group("rhs").strip(),
                numeric_expected=source.numeric_expected,
            )
        )
    return spans


def detect_spans(
    text: str, evaluator: ArithmeticEvaluator, variant: str
) -> list[Span]:
    liberal = variant == "liberal"
    unit_spans = detect_unit_conversions(text, evaluator)
    occupied = [(span.equation_start, span.equation_end) for span in unit_spans]
    arithmetic_spans = detect_arithmetic(text, evaluator, occupied)
    symbolic_spans = detect_symbolic(text, evaluator, occupied, liberal)
    primary = unit_spans + arithmetic_spans + symbolic_spans
    restatements = detect_restatements(text, evaluator, primary, liberal)
    spans = sorted(primary + restatements, key=lambda span: (span.start, span.end))
    deduplicated = []
    for span in spans:
        if any(ranges_overlap((span.start, span.end), (old.start, old.end)) for old in deduplicated):
            continue
        deduplicated.append(span)
    return deduplicated


def flatten_trace_value(value: Any, label: str) -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        if value.strip():
            yield label, value
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten_trace_value(item, "{}[{}]".format(label, index))
        return
    if isinstance(value, dict):
        leaf_keys = {
            "trace_text",
            "reasoning_text",
            "model_output_text",
            "output_text",
            "generated_text",
            "completion_text",
            "raw_output",
            "full_text",
            "reasoning",
            "trace",
            "completion",
            "model_output",
            "response_text",
            "text",
            "content",
        }
        preferred = [key for key in value if str(key).lower() in leaf_keys]
        if preferred:
            for key in preferred:
                yield from flatten_trace_value(
                    value[key], "{}.{}".format(label, key)
                )
            return
        for key in sorted(value, key=str):
            item = value[key]
            if isinstance(item, str):
                if item.strip():
                    yield "{}.{}".format(label, key), item
            elif isinstance(item, (list, dict)):
                yield from flatten_trace_value(item, "{}.{}".format(label, key))


def get_dot_path(record: dict[str, Any], path: str) -> Any:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def record_problem_id(record: dict[str, Any], path: Path, line_number: int) -> str:
    for key in ("id", "problem_id", "task_id", "item_id"):
        value = record.get(key)
        if isinstance(value, (str, int)) and str(value):
            return str(value)
    return "{}:{}".format(path.name, line_number)


def read_traces(
    paths: list[Path], text_fields: tuple[str, ...]
) -> tuple[list[Trace], list[FileAudit]]:
    traces = []
    audits = []
    seen = set()
    for path in paths:
        audit = FileAudit(path=path)
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                line = raw_line.strip()
                if not line:
                    continue
                audit.records += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    audit.malformed += 1
                    continue
                if not isinstance(record, dict):
                    audit.malformed += 1
                    continue
                audit.keys.update(record.keys())
                problem_id = record_problem_id(record, path, line_number)
                for field in text_fields:
                    value = get_dot_path(record, field)
                    for label, trace_text in flatten_trace_value(value, field):
                        audit.trace_values += 1
                        dedupe_key = (
                            problem_id,
                            record.get("mode"),
                            record.get("seed"),
                            record.get("item_index"),
                            label,
                            trace_text,
                        )
                        if dedupe_key in seen:
                            continue
                        seen.add(dedupe_key)
                        traces.append(
                            Trace(
                                problem_id=problem_id,
                                source="{}:{}:{}".format(path, line_number, label),
                                text=trace_text,
                            )
                        )
        audits.append(audit)
    return traces, audits


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_ci(
    per_problem: dict[str, tuple[int, int]], samples: int, seed: int
) -> tuple[float, float]:
    problem_ids = sorted(per_problem)
    if not problem_ids:
        return 0.0, 0.0
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [rng.choice(problem_ids) for _ in problem_ids]
        mechanical = sum(per_problem[item][0] for item in selected)
        total = sum(per_problem[item][1] for item in selected)
        estimates.append(mechanical / total if total else 0.0)
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def measure_variant(
    traces: list[Trace],
    evaluator: ArithmeticEvaluator,
    variant: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> DetectorStats:
    total_tokens = 0
    mechanical_tokens = 0
    span_count = 0
    exact_spans = 0
    wrong_spans = 0
    kind_counts: Counter[str] = Counter()
    wrong_examples = []
    per_problem_mutable: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for trace in traces:
        trace_tokens = approximate_tokens(trace.text)
        spans = detect_spans(trace.text, evaluator, variant)
        span_tokens = sum(
            approximate_tokens(trace.text[span.start:span.end]) for span in spans
        )
        span_tokens = min(span_tokens, trace_tokens)
        total_tokens += trace_tokens
        mechanical_tokens += span_tokens
        per_problem_mutable[trace.problem_id][0] += span_tokens
        per_problem_mutable[trace.problem_id][1] += trace_tokens
        span_count += len(spans)
        for span in spans:
            kind_counts[span.kind] += 1
            if span.exact:
                exact_spans += 1
            else:
                wrong_spans += 1
                if len(wrong_examples) < 8:
                    wrong_examples.append((trace, span))

    per_problem = {
        key: (value[0], value[1]) for key, value in per_problem_mutable.items()
    }
    fraction = mechanical_tokens / total_tokens if total_tokens else 0.0
    ci_low, ci_high = bootstrap_ci(
        per_problem,
        bootstrap_samples,
        bootstrap_seed + (1 if variant == "liberal" else 0),
    )
    return DetectorStats(
        variant=variant,
        problems=len(per_problem),
        traces=len(traces),
        total_tokens=total_tokens,
        mechanical_tokens=mechanical_tokens,
        spans=span_count,
        exact_spans=exact_spans,
        wrong_spans=wrong_spans,
        fraction=fraction,
        ci_low=ci_low,
        ci_high=ci_high,
        mean_length=(total_tokens / len(traces) if traces else 0.0),
        kind_counts=kind_counts,
        wrong_examples=wrong_examples,
    )


def speedup(mechanical_fraction: float, drafter_cost: float, length: float) -> float:
    if length <= 0:
        return 0.0
    cost = (1.0 - mechanical_fraction) * ((length * drafter_cost + 1.0) / length)
    return math.inf if cost == 0 else 1.0 / cost


def format_percent(value: float) -> str:
    return "{:.2f}%".format(100 * value)


def format_ratio(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return "{:.2f}%".format(100 * numerator / denominator)


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines = []
    lines.append("  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)))
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    return "\n".join(lines)


def resolve_inputs(raw_inputs: list[str]) -> list[Path]:
    paths = []
    for raw in raw_inputs:
        matches = glob.glob(raw, recursive=True)
        candidates = [Path(match) for match in matches] if matches else [Path(raw)]
        for candidate in candidates:
            if candidate.is_dir():
                paths.extend(sorted(candidate.rglob("*items*.jsonl")))
            elif candidate.is_file():
                paths.append(candidate)
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return sorted(unique, key=lambda path: str(path).lower())


def default_inputs() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    roots = [
        repo_root / "AGENTS-GOALs",
        repo_root.parent / "AutoTree" / "AGENTS-GOALs",
    ]
    selected = []
    for root in roots:
        selected.extend(
            [
                root / "results-tputdiag" / "dg_tree_items.jsonl",
                root / "results-tputdiag" / "dg_bo8_items.jsonl",
            ]
        )
        for directory in ("results-headline", "results-largetree"):
            if (root / directory).is_dir():
                selected.extend(sorted((root / directory).glob("*items*.jsonl")))
    return resolve_inputs([str(path) for path in selected if path.is_file()])


def audit_summary(audits: list[FileAudit]) -> tuple[int, int, Counter[str]]:
    records = sum(audit.records for audit in audits)
    malformed = sum(audit.malformed for audit in audits)
    answer_fields: Counter[str] = Counter()
    for audit in audits:
        for field in ANSWER_ONLY_FIELDS:
            if field in audit.keys:
                answer_fields[field] += audit.keys[field]
    return records, malformed, answer_fields


def print_no_trace_report(
    paths: list[Path], audits: list[FileAudit], fields: tuple[str, ...], evaluator: ArithmeticEvaluator
) -> None:
    records, malformed, answer_fields = audit_summary(audits)
    print("data_status: NO_FULL_TRACES")
    print("files_scanned: {}".format(len(paths)))
    print("records_scanned: {}".format(records))
    print("malformed_records: {}".format(malformed))
    print("evaluator: {}".format(evaluator.backend))
    print("recognized_trace_fields: {}".format(", ".join(fields)))
    print("trace_values_found: 0")
    if answer_fields:
        print("answer_only_fields_seen: {}".format(
            ", ".join(sorted(answer_fields))
        ))
    print("result: real mechanical fraction m was not measurable")
    print("result: no real multiplier was computed")
    print(
        "next_capture: persist each raw assistant completion before answer extraction "
        "as trace_text, sample_traces, or branch_traces"
    )


def print_measurement(
    paths: list[Path],
    audits: list[FileAudit],
    traces: list[Trace],
    stats: list[DetectorStats],
    evaluator: ArithmeticEvaluator,
    synthetic: bool,
    sequence_length: float | None,
    show_spans: bool,
) -> None:
    records, malformed, _answer_fields = audit_summary(audits)
    status = "SYNTHETIC_PLUMBING_VALIDATION_ONLY" if synthetic else "REAL_FULL_TRACES"
    print("data_status: {}".format(status))
    print("files_scanned: {}".format(len(paths)))
    print("records_scanned: {}".format(records))
    print("malformed_records: {}".format(malformed))
    print("deduplicated_traces: {}".format(len(traces)))
    print("evaluator: {}".format(evaluator.backend))
    if not HAVE_SYMPY:
        print("symbolic_detection: disabled because SymPy is absent")
    print(
        "token_estimator: APPROX max(whitespace chunks, ceil(non-whitespace chars / 4))"
    )
    print("mechanical_boundary: RHS/result tokens only; setup expression remains REASONING")
    print()

    rows = []
    for item in stats:
        rows.append(
            [
                item.variant,
                str(item.problems),
                str(item.traces),
                "{:.1f}".format(item.mean_length),
                "{}/{}".format(item.mechanical_tokens, item.total_tokens),
                str(item.spans),
                format_percent(item.fraction),
                "[{}, {}]".format(
                    format_percent(item.ci_low), format_percent(item.ci_high)
                ),
                format_ratio(item.exact_spans, item.spans),
                format_ratio(item.wrong_spans, item.spans),
            ]
        )
    print("Approximate mechanical span fraction and verifiability")
    print(
        render_table(
            [
                "detector",
                "problems",
                "traces",
                "L~",
                "mech~/total~",
                "spans",
                "m~",
                "95% problem-bootstrap CI",
                "tool exact",
                "model wrong/tool right",
            ],
            rows,
        )
    )
    print()

    speed_rows = []
    for item in stats:
        length = sequence_length if sequence_length is not None else item.mean_length
        for drafter_cost in (0.12, 0.05, 0.0):
            value = speedup(item.fraction, drafter_cost, length)
            speed_rows.append(
                [
                    item.variant,
                    "{:.2f}".format(drafter_cost),
                    "{:.1f}".format(length),
                    "inf" if math.isinf(value) else "{:.3f}x".format(value),
                ]
            )
    print("Idealized upper-bound speedups")
    print(render_table(["detector", "d", "L~", "speedup upper bound"], speed_rows))
    print()
    print(
        "formula: speedup = 1 / ((1-m~) * (L~*d + 1) / L~)"
    )
    print(
        "honesty_gate: upper bound only; assumes perfect span detection and zero splice overhead"
    )
    print(
        "honesty_gate: the model must set up every computation; setup tokens are never counted mechanical"
    )
    print(
        "bias: both detectors favor false negatives; the liberal result is a range endpoint, not a claim"
    )
    if synthetic:
        print("honesty_gate: synthetic numbers validate plumbing only and are not empirical evidence")

    print()
    print("Detected span kinds")
    kind_rows = []
    all_kinds = sorted(set().union(*(item.kind_counts for item in stats)))
    for kind in all_kinds:
        kind_rows.append([kind] + [str(item.kind_counts[kind]) for item in stats])
    if kind_rows:
        print(render_table(["kind"] + [item.variant for item in stats], kind_rows))
    else:
        print("none")

    if show_spans:
        print()
        print("Model-wrong/tool-right examples")
        emitted = False
        for item in stats:
            for trace, span in item.wrong_examples:
                emitted = True
                print(
                    "{} | {} | {} | reported={!r} | expected={!r} | {}".format(
                        item.variant,
                        trace.problem_id,
                        span.kind,
                        span.reported,
                        span.expected,
                        trace.source,
                    )
                )
        if not emitted:
            print("none")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure approximate mechanical-output fraction in JSONL traces."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "JSONL files, directories, or glob patterns. With no inputs, scan the "
            "requested AGENTS-GOALs result archives."
        ),
    )
    parser.add_argument(
        "--text-field",
        action="append",
        default=[],
        help="Dot path containing a trace string/list/dict. Repeat as needed.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="By-problem bootstrap replicates (default: 10000).",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    parser.add_argument(
        "--sequence-length",
        type=float,
        default=None,
        help="Override L~ in the multiplier formula; default is mean approximate trace length.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Label all output as synthetic plumbing validation only.",
    )
    parser.add_argument(
        "--show-spans",
        action="store_true",
        help="Print model-wrong/tool-right examples.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    if args.sequence_length is not None and args.sequence_length <= 0:
        parser.error("--sequence-length must be positive")

    paths = resolve_inputs(args.inputs) if args.inputs else default_inputs()
    if not paths:
        parser.error("no input JSONL files found")
    fields = tuple(args.text_field) if args.text_field else AUTO_TRACE_FIELDS
    evaluator = ArithmeticEvaluator()
    traces, audits = read_traces(paths, fields)
    if not traces:
        print_no_trace_report(paths, audits, fields, evaluator)
        return 0

    stats = [
        measure_variant(
            traces,
            evaluator,
            variant,
            args.bootstrap_samples,
            args.bootstrap_seed,
        )
        for variant in ("conservative", "liberal")
    ]
    print_measurement(
        paths,
        audits,
        traces,
        stats,
        evaluator,
        args.synthetic,
        args.sequence_length,
        args.show_spans,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
