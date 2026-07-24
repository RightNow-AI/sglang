#!/usr/bin/env python3
'''Direct assertion tests for mc_equiv.py. No test runner is required.'''

from mc_equiv import extract_choice, is_choice_correct


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
    check_equal(extract_choice('Answer: A'), 'A', 'answer marker')
    check_equal(extract_choice('answer: b'), 'B', 'lowercase answer')
    check_equal(extract_choice('ANSWER:\n C'), 'C', 'multiline answer')
    check_equal(
        extract_choice('Answer: A\nCorrection. Answer: D.'),
        'D',
        'last answer marker',
    )
    check_equal(
        extract_choice('Answer: B\nA final stray C'),
        'B',
        'answer marker beats trailing letter',
    )
    check_equal(extract_choice('Reasoning complete. D'), 'D', 'trailing letter')
    check_equal(extract_choice('The choice is (C).'), 'C', 'parenthesized trailing')
    check_equal(extract_choice('Final: b!'), 'B', 'punctuated trailing')
    check_equal(extract_choice('A'), 'A', 'letter only')
    check_equal(extract_choice('The word CAD'), None, 'embedded letter')
    check_equal(extract_choice('Option A because it fits.'), None, 'not trailing')
    check_equal(extract_choice('Answer: E'), None, 'out of range marker')
    check_equal(extract_choice('No answer is provided.'), None, 'no choice')
    check_equal(extract_choice(''), None, 'empty text')
    check_equal(extract_choice(None), None, 'non-string text')

    check_true(is_choice_correct('Answer: A', 'A'), 'correct marker')
    check_true(is_choice_correct('conclusion: c', 'C'), 'correct trailing')
    check_true(is_choice_correct('Answer: d', ' D '), 'normalized gold case')
    check_false(is_choice_correct('Answer: A', 'B'), 'wrong choice')
    check_false(is_choice_correct('Answer: A', 'E'), 'invalid gold')
    check_false(is_choice_correct('No final choice', 'A'), 'missing prediction')
    check_false(is_choice_correct('Answer: B', None), 'non-string gold')

    print('PASS: {} assertions'.format(PASS_COUNT))


if __name__ == '__main__':
    main()
