#!/usr/bin/env python3
'''Direct assertion tests for math_equiv.py. No test runner is required.'''

from math_equiv import extract_final_answer, is_equiv, normalize_answer


PASS_COUNT = 0


def check_equal(actual, expected, label):
    global PASS_COUNT
    assert actual == expected, '{}: expected {!r}, got {!r}'.format(
        label, expected, actual
    )
    PASS_COUNT += 1


def check_true(value, label):
    global PASS_COUNT
    assert value, '{}: expected true'.format(label)
    PASS_COUNT += 1


def check_false(value, label):
    global PASS_COUNT
    assert not value, '{}: expected false'.format(label)
    PASS_COUNT += 1


def main():
    check_equal(normalize_answer(r'\left( 2 \right)'), '(2)', 'left-right')
    check_equal(normalize_answer('  42  '), '42', 'spaces')
    check_equal(normalize_answer(r'90^\circ'), '90', 'degree latex')
    check_equal(normalize_answer('90\u00b0'), '90', 'degree unicode')
    check_equal(normalize_answer('$42$'), '42', 'dollar signs')
    check_equal(normalize_answer(r'\text{42}'), '42', 'text wrapper')
    check_equal(normalize_answer(r'42\text{ cm}'), '42', 'text unit')
    check_equal(normalize_answer(r'7\text{ widgets}'), '7', 'generic text unit')
    check_equal(normalize_answer('42 meters'), '42', 'plain unit')
    check_equal(normalize_answer('3.000'), '3', 'trailing decimal zeros')
    check_equal(normalize_answer('3.5000'), r'\frac{7}{2}', 'decimal fraction')
    check_equal(normalize_answer('.5'), r'\frac{1}{2}', 'leading decimal')
    check_equal(normalize_answer('-.5'), r'-\frac{1}{2}', 'negative decimal')
    check_equal(normalize_answer('\u22125'), '-5', 'unicode negative')
    check_equal(normalize_answer(r'\frac12'), r'\frac{1}{2}', 'compact frac')
    check_equal(normalize_answer(r'\frac1{2}'), r'\frac{1}{2}', 'left compact frac')
    check_equal(normalize_answer(r'\frac{1}2'), r'\frac{1}{2}', 'right compact frac')
    check_equal(normalize_answer('1/2'), r'\frac{1}{2}', 'slash frac')
    check_equal(normalize_answer(r'\sqrt2'), r'\sqrt{2}', 'compact sqrt')
    check_equal(normalize_answer(r'\sqrt x'), r'\sqrt{x}', 'spaced sqrt')
    check_equal(normalize_answer('042'), '42', 'leading zeros')
    check_equal(
        normalize_answer(r'\frac{-1}{2}'),
        r'-\frac{1}{2}',
        'negative numerator',
    )
    check_equal(
        normalize_answer(r'\frac{1}{-2}'),
        r'-\frac{1}{2}',
        'negative denominator',
    )
    check_equal(normalize_answer(r'\tfrac{1}{2}'), r'\frac{1}{2}', 'tfrac')
    check_equal(normalize_answer('x = 42'), '42', 'short equation label')
    check_equal(normalize_answer('+42'), '42', 'leading plus')
    check_equal(normalize_answer('1,000'), '1000', 'thousands comma')

    check_true(is_equiv('0.5', r'\frac{1}{2}'), 'decimal versus fraction')
    check_true(is_equiv(r'\frac{2}{4}', r'\frac{1}{2}'), 'reduced fractions')
    check_true(is_equiv('042', '42'), 'AIME leading zeros')
    check_true(is_equiv('-0.5', r'-\frac{1}{2}'), 'negative equivalence')
    check_true(is_equiv('1.0000004', '1'), 'numeric tolerance')
    check_true(is_equiv(r'\sqrt2', r'\sqrt{2}'), 'sqrt equivalence')
    check_true(is_equiv(r'42\text{ kg}', '42'), 'unit equivalence')
    check_true(is_equiv('12/3', '4'), 'multi-digit numeric fraction')
    check_false(is_equiv(r'\frac{1}{2}', r'\frac{1}{3}'), 'different fractions')
    check_false(is_equiv('-2', '2'), 'different signs')
    check_false(is_equiv(r'\sqrt{2}', '2'), 'symbolic mismatch')
    check_false(is_equiv('1.000002', '1'), 'outside tolerance')

    check_equal(extract_final_answer(r'Work. \boxed{42}'), '42', 'boxed')
    check_equal(
        extract_final_answer(r'First \boxed{1}; final \boxed{2}.'),
        '2',
        'last boxed',
    )
    check_equal(
        extract_final_answer(r'Result: \boxed{\frac{1}{\sqrt{2}}}.'),
        r'\frac{1}{\sqrt{2}}',
        'nested boxed braces',
    )
    check_equal(extract_final_answer(r'\boxed {007}'), '007', 'spaced boxed')
    check_equal(extract_final_answer('Answer: 42\nDone.'), '42', 'answer line')
    check_equal(
        extract_final_answer('Answer: 1\nCorrection. Answer: 2.'),
        '2',
        'last answer marker',
    )
    check_equal(extract_final_answer('Answer: 3.14.'), '3.14', 'answer punctuation')
    check_equal(extract_final_answer('Values 3, 7, then 11.'), '11', 'last integer')
    check_equal(extract_final_answer('The result is -12.'), '-12', 'last negative')
    check_equal(extract_final_answer('The result is 0.125.'), '0.125', 'last decimal')
    check_equal(extract_final_answer('No numeric result.'), None, 'no answer')
    check_equal(
        extract_final_answer(r'\boxed{oops then Answer: 9.'),
        '9',
        'malformed box fallback',
    )
    check_equal(
        extract_final_answer(r'Earlier \boxed{8}. Later Answer: 9.'),
        '8',
        'boxed precedence',
    )

    print('PASS: {} assertions'.format(PASS_COUNT))


if __name__ == '__main__':
    main()
