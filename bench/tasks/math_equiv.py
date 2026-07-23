'''Pure standard-library answer extraction and MATH answer equivalence.'''

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Optional, Tuple


TRAILING_PUNCTUATION = '.,;:!?'
MINUS_TRANSLATION = str.maketrans(
    {
        '\u2212': '-',  # Example: a Unicode minus before 5 becomes '-5'.
        '\u2013': '-',  # Example: an en dash used before 5 becomes '-5'.
        '\u2014': '-',  # Example: a long dash used before 5 becomes '-5'.
        '\ufe63': '-',  # Example: a small minus sign becomes '-'.
        '\uff0d': '-',  # Example: a full-width minus sign becomes '-'.
    }
)


def balanced_group(text: str, opening: int) -> Optional[Tuple[str, int]]:
    '''Return group content and the index after its balanced closing brace.'''
    if opening >= len(text) or text[opening] != '{':
        return None
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == '{':
            depth += 1
        elif text[index] == '}':
            depth -= 1
            if depth == 0:
                return text[opening + 1 : index], index + 1
    return None


def unwrap_text_commands(value: str) -> str:
    '''Remove text-mode commands while preserving their brace content.'''
    for command in ('\\text', '\\mbox', '\\mathrm'):
        search_from = 0
        while True:
            start = value.find(command, search_from)
            if start < 0:
                break
            opening = start + len(command)
            while opening < len(value) and value[opening].isspace():
                opening += 1
            group = balanced_group(value, opening)
            if group is None:
                search_from = start + len(command)
                continue
            content, end = group
            if value[:start].strip() and not value[end:].strip():
                # Example: '7\\text{ widgets}' becomes '7' as a trailing unit.
                value = value[:start]
                search_from = len(value)
                continue
            value = value[:start] + content + value[end:]
            search_from = start + len(content)
    return value


def strip_units(value: str) -> str:
    '''Remove common units when they trail an otherwise complete answer.'''
    unit = (
        r'(?:millimeters|millimeter|centimeters|centimeter|kilometers|kilometer|'
        r'meters|meter|mm|cm|km|m|milligrams|milligram|kilograms|kilogram|mg|kg|'
        r'grams|gram|g|milliliters|milliliter|liters|liter|litres|litre|ml|l|'
        r'inches|inch|in|feet|foot|ft|yards|yard|miles|mile|seconds|second|secs|'
        r'sec|s|minutes|minute|mins|min|hours|hour|hrs|hr|days|day|dollars|'
        r'dollar|cents|cent|mph|kph|percent)'
    )
    power = r'(?:\s*(?:\^\s*\{?[23]\}?|squared|cubed))?'
    # Example: '12 cm' and '12\\text{ cm}' both become '12'.
    pattern = re.compile(r'(?i)^(.*?\d)\s*(?:\\,|~)?\s*' + unit + power + r'\s*$')
    match = pattern.match(value)
    return match.group(1) if match is not None else value


def read_tex_operand(value: str, start: int) -> Optional[Tuple[str, int]]:
    if start >= len(value):
        return None
    if value[start] == '{':
        group = balanced_group(value, start)
        if group is None:
            return None
        content, end = group
        return '{' + content + '}', end
    if value[start] in '+-' and start + 1 < len(value):
        return '{' + value[start : start + 2] + '}', start + 2
    if value[start] == '\\':
        command = re.match(r'\\[A-Za-z]+', value[start:])
        if command is not None:
            token = command.group(0)
            return '{' + token + '}', start + len(token)
    return '{' + value[start] + '}', start + 1


def fix_fracs(value: str) -> str:
    '''Brace unbraced numerator and denominator tokens after frac.'''
    output = []
    position = 0
    while True:
        start = value.find('\\frac', position)
        if start < 0:
            output.append(value[position:])
            break
        output.append(value[position:start])
        numerator = read_tex_operand(value, start + len('\\frac'))
        if numerator is None:
            output.append(value[start:])
            break
        numerator_text, after_numerator = numerator
        denominator = read_tex_operand(value, after_numerator)
        if denominator is None:
            output.append(value[start:])
            break
        denominator_text, after_denominator = denominator
        # Example: '\\frac12' and '\\frac1{2}' become '\\frac{1}{2}'.
        output.append('\\frac' + numerator_text + denominator_text)
        position = after_denominator
    return ''.join(output)


def fix_sqrts(value: str) -> str:
    '''Brace an unbraced operand after a square root command.'''
    output = []
    position = 0
    while True:
        start = value.find('\\sqrt', position)
        if start < 0:
            output.append(value[position:])
            break
        output.append(value[position:start])
        operand_start = start + len('\\sqrt')
        if operand_start < len(value) and value[operand_start] == '{':
            group = balanced_group(value, operand_start)
            if group is None:
                output.append(value[start:])
                break
            _, end = group
            output.append(value[start:end])
            position = end
            continue
        operand = read_tex_operand(value, operand_start)
        if operand is None:
            output.append(value[start:])
            break
        operand_text, end = operand
        # Example: '\\sqrt2' becomes '\\sqrt{2}'.
        output.append('\\sqrt' + operand_text)
        position = end
    return ''.join(output)


def normalize_fraction_sign(value: str) -> str:
    patterns = (
        (r'^\\frac\{-(\d+)\}\{(\d+)\}$', r'-\\frac{\1}{\2}'),
        (r'^\\frac\{(\d+)\}\{-(\d+)\}$', r'-\\frac{\1}{\2}'),
        (r'^-\\frac\{-(\d+)\}\{(\d+)\}$', r'\\frac{\1}{\2}'),
    )
    for pattern, replacement in patterns:
        if re.fullmatch(pattern, value):
            # Example: '\\frac{-1}{2}' becomes '-\\frac{1}{2}'.
            return re.sub(pattern, replacement, value)
    return value


def normalize_plain_number(value: str) -> str:
    if re.fullmatch(r'[+-]?\d+', value):
        # Example: '042' becomes '42', and '-0' becomes '0'.
        return str(int(value))

    if re.fullmatch(r'[+-]?(?:\d+\.\d*|\.\d+)', value):
        try:
            exact = Fraction(Decimal(value))
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return value
        # Example: '3.000' becomes '3', removing insignificant trailing zeros.
        if exact.denominator == 1:
            return str(exact.numerator)
        sign = '-' if exact.numerator < 0 else ''
        numerator = abs(exact.numerator)
        # Example: '0.5' becomes '\\frac{1}{2}' exactly.
        return '{}\\frac{{{}}}{{{}}}'.format(sign, numerator, exact.denominator)

    # Example: '1/2' becomes '\\frac{1}{2}' for single-character operands.
    slash = re.fullmatch(r'([+-]?)([^{}\\/])/([^{}\\/])', value)
    if slash is not None:
        sign, numerator, denominator = slash.groups()
        if sign == '+':
            sign = ''
        return '{}\\frac{{{}}}{{{}}}'.format(sign, numerator, denominator)
    return value


def normalize_answer(s: str) -> str:
    '''Return a clean Hendrycks MATH-style canonical answer string.'''
    if s is None:
        return ''
    value = str(s).strip().translate(MINUS_TRANSLATION)

    # Example: line breaks in '1 / 2' are treated as ordinary spaces.
    value = value.replace('\n', ' ').replace('\r', ' ')
    # Example: '\\dfrac{1}{2}' and '\\tfrac{1}{2}' use canonical '\\frac'.
    value = value.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    # Example: '\\left(2\\right)' becomes '(2)'.
    value = value.replace('\\left', '').replace('\\right', '')
    # Example: '$42$' and '\\$42' become '42'.
    value = value.replace('\\$', '').replace('$', '')
    # Example: '90^\\circ' and a Unicode degree suffix become '90'.
    value = re.sub(r'\^\s*\{?\\circ\}?', '', value)
    value = value.replace('\\circ', '').replace('\u00b0', '')
    # Example: '50\\%' and '50%' become '50'.
    value = value.replace('\\%', '').replace('%', '')
    # Example: '\\text{42}' becomes '42' before unit removal.
    value = unwrap_text_commands(value)
    # Example: '12 meters' and '12\\text{ cm}' both become '12'.
    value = strip_units(value)
    # Example: '1,000' becomes '1000' without changing tuple commas.
    value = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', value)
    # Example: TeX spacing in '1\\, 2' does not affect the answer.
    for spacing in ('\\qquad', '\\quad', '\\!', '\\,', '\\;', '\\:', '~'):
        value = value.replace(spacing, '')
    # Example: ordinary spaces in '\\frac {1} {2}' are removed.
    value = re.sub(r'\s+', '', value)
    # Example: a doubled slash command becomes the corresponding single slash.
    while '\\\\' in value:
        value = value.replace('\\\\', '\\')
    # Example: 'x=42' becomes '42' when the left side is a short label.
    equation = value.split('=')
    if len(equation) == 2 and 0 < len(equation[0]) <= 2:
        value = equation[1]
    # Example: '+-2', '-+2', and '--2' become '-2', '-2', and '2'.
    while any(token in value for token in ('+-', '-+', '--', '++')):
        value = value.replace('+-', '-').replace('-+', '-')
        value = value.replace('--', '').replace('++', '+')
    # Example: a redundant leading plus in '+42' is removed.
    if value.startswith('+'):
        value = value[1:]
    # Example: '42.' becomes '42' when punctuation trails the full answer.
    value = value.rstrip(TRAILING_PUNCTUATION)

    value = fix_sqrts(value)
    value = fix_fracs(value)
    value = normalize_fraction_sign(value)
    value = normalize_plain_number(value)
    return value


def numeric_value(value: str) -> Optional[Fraction]:
    value = value.strip()
    if re.fullmatch(r'[+-]?\d+', value):
        return Fraction(int(value), 1)

    if re.fullmatch(r'[+-]?(?:\d+\.\d*|\.\d+)', value):
        try:
            return Fraction(Decimal(value))
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None

    latex = re.fullmatch(
        r'([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}', value
    )
    if latex is not None:
        leading, numerator, denominator = latex.groups()
        try:
            result = Fraction(int(numerator), int(denominator))
        except ZeroDivisionError:
            return None
        return -result if leading == '-' else result

    slash = re.fullmatch(r'([+-]?\d+)\s*/\s*([+-]?\d+)', value)
    if slash is not None:
        try:
            return Fraction(int(slash.group(1)), int(slash.group(2)))
        except ZeroDivisionError:
            return None
    return None


def is_equiv(a: str, b: str) -> bool:
    '''Compare normalized strings, then simple numbers within 1e-6.'''
    normalized_a = normalize_answer(a)
    normalized_b = normalize_answer(b)
    if normalized_a == normalized_b:
        return True
    numeric_a = numeric_value(normalized_a)
    numeric_b = numeric_value(normalized_b)
    if numeric_a is None or numeric_b is None:
        return False
    return abs(numeric_a - numeric_b) <= Fraction(1, 1_000_000)


def strip_trailing_punctuation(value: str) -> str:
    return value.strip().rstrip(TRAILING_PUNCTUATION).strip()


def extract_final_answer(text: str) -> Optional[str]:
    '''Extract the last boxed answer, Answer line, or final numeric token.'''
    if text is None:
        return None
    value = str(text)

    last_boxed = None
    search_from = 0
    while True:
        start = value.find('\\boxed', search_from)
        if start < 0:
            break
        opening = start + len('\\boxed')
        while opening < len(value) and value[opening].isspace():
            opening += 1
        group = balanced_group(value, opening)
        if group is not None:
            last_boxed, end = group
            search_from = end
        else:
            search_from = opening + 1
    if last_boxed is not None:
        return strip_trailing_punctuation(last_boxed)

    markers = list(re.finditer(r'(?i)\bAnswer\s*:\s*', value))
    if markers:
        start = markers[-1].end()
        newline = value.find('\n', start)
        end = len(value) if newline < 0 else newline
        answer = strip_trailing_punctuation(value[start:end])
        return answer or None

    searchable = value.translate(MINUS_TRANSLATION)
    pattern = re.compile(
        r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?'
        r'|[-+]?\.\d+'
    )
    numbers = list(pattern.finditer(searchable))
    return numbers[-1].group(0) if numbers else None
