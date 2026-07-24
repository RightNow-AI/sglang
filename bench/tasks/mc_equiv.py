#!/usr/bin/env python3
'''Extract and score A through D multiple-choice answers.'''

import re


ANSWER_PATTERN = re.compile(r'\banswer\s*:\s*([A-D])\b', re.IGNORECASE)
TRAILING_PATTERN = re.compile(
    r'(?<![A-Za-z0-9_])([A-D])(?=[\s.!?;,\:)\]\}\'"*_`]*\Z)',
    re.IGNORECASE,
)
GOLD_PATTERN = re.compile(r'\s*([A-D])\s*\Z', re.IGNORECASE)


def extract_choice(text):
    '''Return the preferred A through D choice in model output, or None.'''
    if not isinstance(text, str):
        return None
    answers = ANSWER_PATTERN.findall(text)
    if answers:
        return answers[-1].upper()
    trailing = TRAILING_PATTERN.search(text)
    return trailing.group(1).upper() if trailing else None


def is_choice_correct(pred, gold):
    '''Return whether the extracted prediction equals a strict A through D gold.'''
    if not isinstance(gold, str):
        return False
    match = GOLD_PATTERN.fullmatch(gold)
    if match is None:
        return False
    return extract_choice(pred) == match.group(1).upper()
