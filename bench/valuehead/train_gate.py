#!/usr/bin/env python3
"""Train and apply the CascadeTree small-tree correctness gate."""

import argparse
import collections
import json
import math
import os
import random
import re
import sys


FEATURE_NAMES = (
    "leader_count",
    "voter_count",
    "leader_share",
    "n_distinct_answers",
    "second_count",
    "margin",
    "null_fraction",
    "leader_matches_winner",
    "small_tokens_per_branch",
)
PREDICTION_FIELD = "p_small_correct"
MODEL_VERSION = 1
GD_STEPS = 1800
GD_INITIAL_LR = 0.25
GD_DECAY_STEPS = 200.0
PROBABILITY_EPSILON = 1e-15


def normalize_number(value):
    """Return a canonical decimal string, or None when value is not numeric."""
    if value is None:
        return None
    cleaned = re.sub(r"\s+", "", str(value))
    cleaned = cleaned.replace("$", "").replace(",", "").replace("%", "")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", cleaned):
        return None
    sign = ""
    if cleaned[0] in "+-":
        sign = cleaned[0]
        cleaned = cleaned[1:]
    if "." in cleaned:
        integer, fraction = cleaned.split(".", 1)
    else:
        integer, fraction = cleaned, ""
    integer = integer.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    normalized = integer if not fraction else integer + "." + fraction
    if sign == "-" and normalized != "0":
        normalized = "-" + normalized
    return normalized


def nonnegative_int(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def nonnegative_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0.0 else None


def require_cascade_features(record, path, line_number):
    where = "{} line {}".format(path, line_number)
    if record.get("mode") != "cascade":
        raise ValueError("{} field 'mode' must be 'cascade'".format(where))
    branch_answers = record.get("branch_answers")
    if not isinstance(branch_answers, dict):
        raise ValueError("{} field 'branch_answers' must be an object".format(where))
    branches = len(branch_answers)
    if branches <= 0:
        raise ValueError("{} has no branches in 'branch_answers'".format(where))

    votes = []
    for branch_id, answer in branch_answers.items():
        if answer is None:
            continue
        normalized = normalize_number(answer)
        if normalized is None:
            raise ValueError(
                "{} branch {!r} has a non-numeric answer".format(where, branch_id)
            )
        votes.append(normalized)
    voter_count = nonnegative_int(record.get("voter_count"))
    if voter_count is None or voter_count != len(votes):
        raise ValueError(
            "{} voter_count must match numeric branch answers".format(where)
        )
    counts = collections.Counter(votes)
    ranked_counts = sorted(counts.values(), reverse=True)
    computed_leader_count = ranked_counts[0] if ranked_counts else 0
    leader_count = nonnegative_int(record.get("leader_count"))
    if leader_count is None or leader_count != computed_leader_count:
        raise ValueError("{} leader_count must match branch answers".format(where))
    leader_matches_winner = record.get("leader_matches_winner")
    if not isinstance(leader_matches_winner, bool):
        raise ValueError("{} leader_matches_winner must be bool".format(where))
    small_tokens = nonnegative_number(record.get("small_tokens"))
    if small_tokens is None:
        raise ValueError("{} small_tokens must be nonnegative".format(where))

    second_count = ranked_counts[1] if len(ranked_counts) > 1 else 0
    voter_denominator = max(voter_count, 1)
    return [
        float(leader_count),
        float(voter_count),
        leader_count / voter_denominator,
        float(len(counts)),
        float(second_count),
        (leader_count - second_count) / voter_denominator,
        (branches - voter_count) / branches,
        1.0 if leader_matches_winner else 0.0,
        (small_tokens / branches) / 512.0,
    ], {
        "leader_count": leader_count,
        "small_tokens": small_tokens,
    }


def training_example(record, path, line_number):
    features, details = require_cascade_features(record, path, line_number)
    where = "{} line {}".format(path, line_number)
    gold = normalize_number(record.get("gold"))
    if gold is None:
        raise ValueError("{} gold is not numeric".format(where))
    small_answer = normalize_number(record.get("small_winner_answer"))
    escalated = record.get("escalated")
    correct = record.get("correct")
    if not isinstance(escalated, bool) or not isinstance(correct, bool):
        raise ValueError("{} escalated and correct must be bool".format(where))
    details.update(
        {
            "features": features,
            "label": int(small_answer is not None and small_answer == gold),
            "escalated": escalated,
            "large_correct": int(correct),
            "large_tokens": nonnegative_number(record.get("large_tokens")),
        }
    )
    return details


def read_training_items(paths):
    examples = []
    source_stats = []
    total_count = 0
    skipped_error_count = 0
    for path in paths:
        file_total = 0
        file_usable = 0
        file_skipped = 0
        with open(path, "r", encoding="utf-8-sig") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                line = raw_line.strip()
                if not line:
                    continue
                file_total += 1
                total_count += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "invalid JSON in {} line {}: {}".format(
                            path, line_number, exc
                        )
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        "{} line {} is not a JSON object".format(path, line_number)
                    )
                if record.get("error") is not None:
                    file_skipped += 1
                    skipped_error_count += 1
                    continue
                examples.append(training_example(record, path, line_number))
                file_usable += 1
        source_stats.append(
            {
                "path": path,
                "total": file_total,
                "usable": file_usable,
                "skipped_error": file_skipped,
            }
        )
    return examples, {
        "total": total_count,
        "usable": len(examples),
        "skipped_error": skipped_error_count,
        "sources": source_stats,
    }


def fit_standardizer(rows):
    count = len(rows)
    means = [
        sum(row[index] for row in rows) / count
        for index in range(len(FEATURE_NAMES))
    ]
    variances = [
        sum((row[index] - means[index]) ** 2 for row in rows) / count
        for index in range(len(FEATURE_NAMES))
    ]
    return means, [math.sqrt(value) for value in variances]


def standardize_rows(rows, means, stds):
    return [
        [
            (value - means[index]) / (stds[index] if stds[index] > 0.0 else 1.0)
            for index, value in enumerate(row)
        ]
        for row in rows
    ]


def sigmoid(value):
    if value >= 0.0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def fit_logistic(rows, labels, l2):
    """Fit mean log-loss plus l2/(2*n)*sum(weight^2) by gradient descent."""
    count = len(rows)
    width = len(FEATURE_NAMES)
    weights = [0.0] * width
    intercept = 0.0
    for step in range(GD_STEPS):
        intercept_gradient = 0.0
        weight_gradients = [0.0] * width
        for row, label in zip(rows, labels):
            probability = sigmoid(
                intercept + sum(weight * value for weight, value in zip(weights, row))
            )
            error = probability - label
            intercept_gradient += error
            for index, value in enumerate(row):
                weight_gradients[index] += error * value
        learning_rate = GD_INITIAL_LR / math.sqrt(1.0 + step / GD_DECAY_STEPS)
        intercept -= learning_rate * intercept_gradient / count
        for index in range(width):
            gradient = (
                weight_gradients[index] / count + l2 * weights[index] / count
            )
            weights[index] -= learning_rate * gradient
    return intercept, weights


def predict_probabilities(rows, intercept, weights):
    return [
        sigmoid(
            intercept + sum(weight * value for weight, value in zip(weights, row))
        )
        for row in rows
    ]


def binary_accuracy(labels, probabilities):
    correct = sum(
        (probability >= 0.5) == bool(label)
        for label, probability in zip(labels, probabilities)
    )
    return correct / len(labels), correct


def log_loss(labels, probabilities):
    total = 0.0
    for label, probability in zip(labels, probabilities):
        clipped = min(
            max(probability, PROBABILITY_EPSILON), 1.0 - PROBABILITY_EPSILON
        )
        total -= label * math.log(clipped) + (1 - label) * math.log(1.0 - clipped)
    return total / len(labels)


def rank_auc(labels, probabilities):
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ordered = sorted(zip(probabilities, labels), key=lambda pair: pair[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        stop = index + 1
        while stop < len(ordered) and ordered[stop][0] == ordered[index][0]:
            stop += 1
        average_rank = ((index + 1) + stop) / 2.0
        positive_rank_sum += average_rank * sum(
            label for _score, label in ordered[index:stop]
        )
        index = stop
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def cross_validate(rows, labels, requested_folds, l2, seed):
    effective_folds = min(requested_folds, len(rows))
    if effective_folds < 2:
        raise ValueError("at least 2 usable records are required for cross-validation")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    fold_indices = [
        indices[offset::effective_folds] for offset in range(effective_folds)
    ]
    out_of_fold = [None] * len(rows)
    fold_results = []
    all_indices = set(indices)
    for fold_number, test_indices in enumerate(fold_indices, 1):
        train_indices = sorted(all_indices.difference(test_indices))
        train_rows = [rows[index] for index in train_indices]
        train_labels = [labels[index] for index in train_indices]
        test_rows = [rows[index] for index in test_indices]
        test_labels = [labels[index] for index in test_indices]
        means, stds = fit_standardizer(train_rows)
        standardized_train = standardize_rows(train_rows, means, stds)
        standardized_test = standardize_rows(test_rows, means, stds)
        intercept, weights = fit_logistic(standardized_train, train_labels, l2)
        probabilities = predict_probabilities(standardized_test, intercept, weights)
        for index, probability in zip(test_indices, probabilities):
            out_of_fold[index] = probability
        accuracy, correct_count = binary_accuracy(test_labels, probabilities)
        fold_results.append(
            {
                "fold": fold_number,
                "train_n": len(train_indices),
                "test_n": len(test_indices),
                "positive_n": sum(test_labels),
                "correct_n": correct_count,
                "accuracy": accuracy,
                "auc": rank_auc(test_labels, probabilities),
                "log_loss": log_loss(test_labels, probabilities),
            }
        )
    if any(probability is None for probability in out_of_fold):
        raise RuntimeError("cross-validation did not score every record")
    return out_of_fold, fold_results, effective_folds


def calibration_rows(labels, probabilities):
    bins = [[] for _unused in range(10)]
    for label, probability in zip(labels, probabilities):
        bins[min(int(probability * 10.0), 9)].append((probability, label))
    rows = []
    for index, values in enumerate(bins):
        count = len(values)
        positive_n = sum(label for _probability, label in values)
        rows.append(
            {
                "bin": index,
                "n": count,
                "positive_n": positive_n,
                "mean_predicted": (
                    sum(probability for probability, _label in values) / count
                    if count
                    else None
                ),
                "observed": positive_n / count if count else None,
            }
        )
    return rows


def large_reference(examples, assumed_large_tokens):
    escalated = [example for example in examples if example["escalated"]]
    if not escalated:
        return {
            "n": 0,
            "correct_n": 0,
            "accuracy": None,
            "mean_tokens": assumed_large_tokens,
            "recorded_tokens_n": 0,
            "assumed_tokens_n": 0,
        }
    token_values = []
    recorded_tokens_n = 0
    for example in escalated:
        if example["large_tokens"] is None:
            token_values.append(assumed_large_tokens)
        else:
            token_values.append(example["large_tokens"])
            recorded_tokens_n += 1
    correct_n = sum(example["large_correct"] for example in escalated)
    return {
        "n": len(escalated),
        "correct_n": correct_n,
        "accuracy": correct_n / len(escalated),
        "mean_tokens": sum(token_values) / len(token_values),
        "recorded_tokens_n": recorded_tokens_n,
        "assumed_tokens_n": len(escalated) - recorded_tokens_n,
    }


def policy_result(accept_mask, examples, large, small_cost, large_cost):
    total_n = len(examples)
    accept_indices = [
        index for index, accepted in enumerate(accept_mask) if accepted
    ]
    accept_n = len(accept_indices)
    escalated_n = total_n - accept_n
    accept_correct_n = sum(examples[index]["label"] for index in accept_indices)
    accept_accuracy = accept_correct_n / accept_n if accept_n else None
    small_total_cost = (
        sum(example["small_tokens"] for example in examples) * small_cost
    )
    total_cost = (
        small_total_cost + escalated_n * large["mean_tokens"] * large_cost
    )
    if large["accuracy"] is None:
        projected_correct = None
        projected_accuracy = None
        cost_per_correct = None
    else:
        projected_correct = accept_correct_n + escalated_n * large["accuracy"]
        projected_accuracy = projected_correct / total_n
        cost_per_correct = (
            total_cost / projected_correct if projected_correct > 0.0 else None
        )
    return {
        "total_n": total_n,
        "accept_n": accept_n,
        "accept_correct_n": accept_correct_n,
        "accept_accuracy": accept_accuracy,
        "escalated_n": escalated_n,
        "escalation_rate": escalated_n / total_n,
        "projected_correct": projected_correct,
        "projected_accuracy": projected_accuracy,
        "cost_per_correct": cost_per_correct,
    }


def decision_analysis(examples, probabilities, large, small_cost, large_cost):
    threshold_rows = []
    for step in range(1, 20):
        threshold = step * 0.05
        row = policy_result(
            [probability >= threshold for probability in probabilities],
            examples,
            large,
            small_cost,
            large_cost,
        )
        row["threshold"] = threshold
        threshold_rows.append(row)
    accuracy_floor = (
        None if large["accuracy"] is None else large["accuracy"] - 0.015
    )
    eligible = [
        row
        for row in threshold_rows
        if accuracy_floor is not None
        and row["projected_accuracy"] is not None
        and row["projected_accuracy"] >= accuracy_floor
        and row["cost_per_correct"] is not None
    ]
    best = (
        min(
            eligible,
            key=lambda row: (
                row["cost_per_correct"],
                row["escalation_rate"],
                row["threshold"],
            ),
        )
        if eligible
        else None
    )
    fixed = policy_result(
        [example["leader_count"] >= 6 for example in examples],
        examples,
        large,
        small_cost,
        large_cost,
    )
    return threshold_rows, best, fixed, accuracy_floor


def format_rate(value):
    return "n/a" if value is None else "{:.6%}".format(value)


def format_float(value, digits=6):
    return "n/a" if value is None else ("{:.%df}" % digits).format(value)


def projected_fraction(row):
    if row["projected_correct"] is None:
        return "n/a/{}".format(row["total_n"])
    return "{:.3f}/{}".format(row["projected_correct"], row["total_n"])


def build_report(
    examples,
    input_stats,
    fold_results,
    effective_folds,
    labels,
    probabilities,
    large,
    threshold_rows,
    best,
    fixed,
    accuracy_floor,
    args,
):
    lines = ["ESCALATION-GATE CALIBRATOR", "", "DATA"]
    lines.append(
        "records: total_n={} usable_n={} skipped_error_n={}".format(
            input_stats["total"],
            input_stats["usable"],
            input_stats["skipped_error"],
        )
    )
    for source in input_stats["sources"]:
        lines.append(
            "source: {} total_n={} usable_n={} skipped_error_n={}".format(
                source["path"],
                source["total"],
                source["usable"],
                source["skipped_error"],
            )
        )
    positive_n = sum(labels)
    lines.append(
        "labels: correct_small_n={} incorrect_small_n={} total_n={}".format(
            positive_n, len(labels) - positive_n, len(labels)
        )
    )
    lines.append(
        "label_rule: normalize small_winner_answer and gold with the cascade "
        "canonical-decimal helper; label 1 only when the normalized small answer "
        "is non-null and equals normalized gold"
    )
    lines.append("label_exclusion: record.correct is never used as the training label")
    if len(examples) < 100:
        lines.extend(
            [
                "",
                "!!!!!!!!!!!!!!!! LOW-DATA WARNING !!!!!!!!!!!!!!!!",
                "usable_n={} is below 100; calibration and threshold selection "
                "are unstable".format(len(examples)),
                "!!!!!!!!!!!!!!!! LOW-DATA WARNING !!!!!!!!!!!!!!!!",
            ]
        )

    lines.extend(["", "CROSS-VALIDATION"])
    lines.append(
        "folds: requested_n={} effective_n={} shuffled_seed={} "
        "probabilities=out-of-fold".format(args.folds, effective_folds, args.seed)
    )
    lines.append(
        "fold  train_n  test_n  positive_n  correct_n  accuracy  auc  log_loss"
    )
    for row in fold_results:
        lines.append(
            "{fold:>4}  {train_n:>7}  {test_n:>6}  {positive_n:>10}  "
            "{correct_n:>9}  {accuracy:>11}  {auc:>8}  {loss:>8}".format(
                fold=row["fold"],
                train_n=row["train_n"],
                test_n=row["test_n"],
                positive_n=row["positive_n"],
                correct_n=row["correct_n"],
                accuracy=format_rate(row["accuracy"]),
                auc=format_float(row["auc"]),
                loss=format_float(row["log_loss"]),
            )
        )
    overall_accuracy, overall_correct = binary_accuracy(labels, probabilities)
    lines.append(
        "overall_oof: n={} positive_n={} correct_n={} accuracy={} auc={} "
        "log_loss={}".format(
            len(labels),
            positive_n,
            overall_correct,
            format_rate(overall_accuracy),
            format_float(rank_auc(labels, probabilities)),
            format_float(log_loss(labels, probabilities)),
        )
    )

    lines.extend(
        ["", "CALIBRATION (out-of-fold, 10 equal-width probability bins)"]
    )
    lines.append("bin         n  positive_n  mean_predicted  observed_rate")
    for row in calibration_rows(labels, probabilities):
        closing = "]" if row["bin"] == 9 else ")"
        bin_label = "[{:.1f},{:.1f}{}".format(
            row["bin"] / 10.0, (row["bin"] + 1) / 10.0, closing
        )
        lines.append(
            "{:<9} {:>4} {:>11} {:>15} {:>14}".format(
                bin_label,
                row["n"],
                row["positive_n"],
                format_float(row["mean_predicted"]),
                format_rate(row["observed"]),
            )
        )

    lines.extend(["", "DECISION ANALYSIS (out-of-fold probabilities)"])
    lines.append(
        "large_reference: correct_n={} escalated_input_n={} accuracy={} "
        "mean_large_tokens={} recorded_token_n={} assumed_token_n={}".format(
            large["correct_n"],
            large["n"],
            format_rate(large["accuracy"]),
            format_float(large["mean_tokens"], 3),
            large["recorded_tokens_n"],
            large["assumed_tokens_n"],
        )
    )
    lines.append(
        "projection: accepted items use observed small labels; projected "
        "escalations use the aggregate input large-answer accuracy because "
        "counterfactual large answers are unavailable"
    )
    lines.append(
        "cost: all recorded small_tokens * {:.6f} plus projected escalated_n * "
        "mean_large_tokens * {:.6f}".format(args.small_cost, args.large_cost)
    )
    lines.append(
        "accuracy_constraint: projected_accuracy >= input_large_accuracy - "
        "1.5pt = {} (reference_n={})".format(
            format_rate(accuracy_floor), large["n"]
        )
    )
    lines.append(
        "threshold  escalated_n/total_n  escalation_rate  "
        "accept_correct_n/accept_n  accept_accuracy  "
        "projected_correct_n/total_n  projected_accuracy  cost_per_correct"
    )
    for row in threshold_rows:
        lines.append(
            "{threshold:>9.2f}  {escalated:>19}  {escalation_rate:>15}  "
            "{accepted:>25}  {accept_accuracy:>15}  {projected:>27}  "
            "{projected_accuracy:>18}  {cost_per_correct:>16}".format(
                threshold=row["threshold"],
                escalated="{}/{}".format(row["escalated_n"], row["total_n"]),
                escalation_rate=format_rate(row["escalation_rate"]),
                accepted="{}/{}".format(
                    row["accept_correct_n"], row["accept_n"]
                ),
                accept_accuracy=format_rate(row["accept_accuracy"]),
                projected=projected_fraction(row),
                projected_accuracy=format_rate(row["projected_accuracy"]),
                cost_per_correct=format_float(row["cost_per_correct"], 3),
            )
        )
    eligible_n = sum(
        accuracy_floor is not None
        and row["projected_accuracy"] is not None
        and row["projected_accuracy"] >= accuracy_floor
        for row in threshold_rows
    )
    if best is None:
        lines.append(
            "best_threshold: none (eligible_n=0 threshold_n={}; no threshold "
            "met a computable accuracy constraint)".format(len(threshold_rows))
        )
    else:
        lines.append(
            "best_threshold: t={:.2f} eligible_n={} threshold_n={} "
            "escalated_n={}/{} projected_accuracy={} cost_per_correct={}".format(
                best["threshold"],
                eligible_n,
                len(threshold_rows),
                best["escalated_n"],
                best["total_n"],
                format_rate(best["projected_accuracy"]),
                format_float(best["cost_per_correct"], 3),
            )
        )

    lines.extend(["", "FIXED RULE COMPARISON"])
    lines.append("rule: accept small when leader_count >= 6")
    lines.append(
        "fixed_rule: escalated_n={} total_n={} escalation_rate={} "
        "accept_correct_n={} accept_n={} accept_accuracy={} "
        "projected_correct={} projected_accuracy={} cost_per_correct={}".format(
            fixed["escalated_n"],
            fixed["total_n"],
            format_rate(fixed["escalation_rate"]),
            fixed["accept_correct_n"],
            fixed["accept_n"],
            format_rate(fixed["accept_accuracy"]),
            projected_fraction(fixed),
            format_rate(fixed["projected_accuracy"]),
            format_float(fixed["cost_per_correct"], 3),
        )
    )
    return "\n".join(lines) + "\n"


def model_document(
    examples, means, stds, intercept, weights, args, input_paths, effective_folds
):
    labels = [example["label"] for example in examples]
    return {
        "feature_names": list(FEATURE_NAMES),
        "intercept": intercept,
        "model_type": "logistic_regression",
        "prediction_field": PREDICTION_FIELD,
        "standardization": {
            "means": {
                name: means[index] for index, name in enumerate(FEATURE_NAMES)
            },
            "stds": {
                name: stds[index] for index, name in enumerate(FEATURE_NAMES)
            },
            "zero_std_scale": 1.0,
        },
        "target": "small_tree_winner_correct",
        "training": {
            "effective_folds": effective_folds,
            "feature_small_tokens_scale": 512.0,
            "gradient_descent": {
                "initial_learning_rate": GD_INITIAL_LR,
                "schedule": "lr / sqrt(1 + step / {})".format(GD_DECAY_STEPS),
                "steps": GD_STEPS,
            },
            "input_paths": list(input_paths),
            "l2": args.l2,
            "l2_objective": "mean_log_loss + l2/(2*n)*sum(weights^2)",
            "small_cost": args.small_cost,
            "large_cost": args.large_cost,
            "assumed_large_tokens": args.assumed_large_tokens,
            "label_rule": "normalize small_winner_answer and gold; label 1 "
            "only for a non-null exact canonical match; never use "
            "record.correct as label",
            "n": len(examples),
            "negative_n": len(examples) - sum(labels),
            "positive_n": sum(labels),
            "requested_folds": args.folds,
            "seed": args.seed,
        },
        "version": MODEL_VERSION,
        "weights": {
            name: weights[index] for index, name in enumerate(FEATURE_NAMES)
        },
    }


def ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_text_atomic(path, value):
    ensure_parent(path)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json_atomic(path, value):
    ensure_parent(path)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def train_and_report(examples, input_stats, args, input_paths):
    if len(examples) < 2:
        raise ValueError(
            "at least 2 usable records are required; got {}".format(len(examples))
        )
    rows = [example["features"] for example in examples]
    labels = [example["label"] for example in examples]
    probabilities, fold_results, effective_folds = cross_validate(
        rows, labels, args.folds, args.l2, args.seed
    )
    large = large_reference(examples, args.assumed_large_tokens)
    threshold_rows, best, fixed, accuracy_floor = decision_analysis(
        examples, probabilities, large, args.small_cost, args.large_cost
    )
    report = build_report(
        examples,
        input_stats,
        fold_results,
        effective_folds,
        labels,
        probabilities,
        large,
        threshold_rows,
        best,
        fixed,
        accuracy_floor,
        args,
    )
    means, stds = fit_standardizer(rows)
    intercept, weights = fit_logistic(
        standardize_rows(rows, means, stds), labels, args.l2
    )
    model = model_document(
        examples,
        means,
        stds,
        intercept,
        weights,
        args,
        input_paths,
        effective_folds,
    )
    return model, report, rank_auc(labels, probabilities)


def load_model(path):
    with open(path, "r", encoding="utf-8") as handle:
        model = json.load(handle)
    if not isinstance(model, dict):
        raise ValueError("{} does not contain a JSON object".format(path))
    if (
        model.get("model_type") != "logistic_regression"
        or model.get("version") != MODEL_VERSION
        or model.get("feature_names") != list(FEATURE_NAMES)
    ):
        raise ValueError("{} is not a supported gate model".format(path))
    standardization = model.get("standardization")
    weights_by_name = model.get("weights")
    if not isinstance(standardization, dict) or not isinstance(
        weights_by_name, dict
    ):
        raise ValueError("{} is missing standardization or weights".format(path))
    try:
        means = [
            float(standardization["means"][name]) for name in FEATURE_NAMES
        ]
        stds = [
            float(standardization["stds"][name]) for name in FEATURE_NAMES
        ]
        weights = [float(weights_by_name[name]) for name in FEATURE_NAMES]
        intercept = float(model["intercept"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("{} has invalid model parameters".format(path)) from exc
    values = means + stds + weights + [intercept]
    if not all(math.isfinite(value) for value in values) or any(
        value < 0.0 for value in stds
    ):
        raise ValueError("{} has non-finite parameters or negative stds".format(path))
    return means, stds, intercept, weights


def score_items(model_path, item_paths, out_path):
    means, stds, intercept, weights = load_model(model_path)
    input_abspaths = {
        os.path.normcase(os.path.abspath(path)) for path in item_paths
    }
    if os.path.normcase(os.path.abspath(out_path)) in input_abspaths:
        raise ValueError("--out-scored must differ from every --items path")
    ensure_parent(out_path)
    temporary = out_path + ".tmp"
    total_n = 0
    scored_n = 0
    input_error_n = 0
    with open(temporary, "w", encoding="utf-8", newline="\n") as output:
        for path in item_paths:
            with open(path, "r", encoding="utf-8-sig") as handle:
                for line_number, raw_line in enumerate(handle, 1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    total_n += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            "invalid JSON in {} line {}: {}".format(
                                path, line_number, exc
                            )
                        ) from exc
                    if not isinstance(record, dict):
                        raise ValueError(
                            "{} line {} is not a JSON object".format(
                                path, line_number
                            )
                        )
                    scored_record = dict(record)
                    if record.get("error") is not None:
                        probability = None
                        input_error_n += 1
                    else:
                        features, _details = require_cascade_features(
                            record, path, line_number
                        )
                        row = standardize_rows([features], means, stds)
                        probability = predict_probabilities(
                            row, intercept, weights
                        )[0]
                        scored_n += 1
                    scored_record[PREDICTION_FIELD] = probability
                    output.write(
                        json.dumps(
                            scored_record,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                    )
                    output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, out_path)
    print(
        "prediction: total_n={} scored_n={} input_error_n={} field={} out={}".format(
            total_n,
            scored_n,
            input_error_n,
            PREDICTION_FIELD,
            os.path.abspath(out_path),
        )
    )


def distribute_vote_counts(voter_count, leader_count, rng):
    remaining = voter_count - leader_count
    counts = [leader_count]
    while remaining > 0:
        next_count = rng.randint(1, min(leader_count, remaining))
        counts.append(next_count)
        remaining -= next_count
    return counts


def synthetic_demo_records(seed, count=500):
    rng = random.Random(seed)
    records = []
    for index in range(count):
        branches = 8
        voter_count = rng.randint(4, branches)
        leader_count = rng.randint(2, voter_count)
        vote_counts = distribute_vote_counts(voter_count, leader_count, rng)
        ranked = sorted(vote_counts, reverse=True)
        second_count = ranked[1] if len(ranked) > 1 else 0
        leader_share = leader_count / voter_count
        margin = (leader_count - second_count) / voter_count
        planted_logit = -7.0 + 6.0 * leader_share + 7.0 * margin
        small_is_correct = rng.random() < sigmoid(planted_logit)
        gold = "42"
        small_answer = gold if small_is_correct else "41"
        leader_matches = rng.random() < 0.75
        leader_answer = small_answer if leader_matches else "40"

        answers = []
        for answer_index, answer_count in enumerate(vote_counts):
            answer = (
                leader_answer if answer_index == 0 else str(100 + answer_index)
            )
            answers.extend([answer] * answer_count)
        answers.extend([None] * (branches - voter_count))
        branch_answers = {
            str(branch): answer for branch, answer in enumerate(answers)
        }
        escalated = leader_count < 6
        large_is_correct = rng.random() < 0.88
        records.append(
            {
                "id": "demo-{:04d}".format(index),
                "mode": "cascade",
                "correct": large_is_correct if escalated else small_is_correct,
                "escalated": escalated,
                "small_winner_answer": small_answer,
                "branch_answers": branch_answers,
                "branch_answer_leader": leader_answer,
                "leader_count": leader_count,
                "voter_count": voter_count,
                "leader_matches_winner": leader_matches,
                "small_tokens": branches * rng.randint(64, 512),
                "large_tokens": rng.randint(120, 280) if escalated else 0,
                "gold": gold,
                "error": None,
            }
        )
    return records


def run_demo(args):
    records = synthetic_demo_records(args.seed)
    examples = [
        training_example(record, "<synthetic-demo>", index + 1)
        for index, record in enumerate(records)
    ]
    input_stats = {
        "total": len(records),
        "usable": len(records),
        "skipped_error": 0,
        "sources": [
            {
                "path": "<synthetic-demo>",
                "total": len(records),
                "usable": len(records),
                "skipped_error": 0,
            }
        ],
    }
    _model, report, auc = train_and_report(
        examples, input_stats, args, ["<synthetic-demo>"]
    )
    print(report, end="")
    if auc is None or auc <= 0.8:
        raise AssertionError(
            "demo held-out OOF AUC must exceed 0.8; got {}".format(
                format_float(auc)
            )
        )
    print(
        "DEMO PASS: held-out OOF AUC {} > 0.800000 (n={})".format(
            format_float(auc), len(records)
        )
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train or apply the offline CascadeTree escalation gate."
    )
    parser.add_argument("--items", nargs="+", help="cascade item JSONL path(s)")
    parser.add_argument("--out-model", help="trained model JSON output")
    parser.add_argument("--out-report", help="text report output")
    parser.add_argument(
        "--predict", metavar="MODEL_JSON", help="score items with a saved model"
    )
    parser.add_argument("--out-scored", help="prediction JSONL output")
    parser.add_argument(
        "--demo", action="store_true", help="run the seeded 500-row self-test"
    )
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--small-cost", type=float, default=1.0)
    parser.add_argument("--large-cost", type=float, default=10.0)
    parser.add_argument("--assumed-large-tokens", type=float, default=200.0)
    return parser


def validate_args(parser, args):
    if args.l2 < 0.0 or not math.isfinite(args.l2):
        parser.error("--l2 must be finite and nonnegative")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    for name in ("small_cost", "large_cost", "assumed_large_tokens"):
        value = getattr(args, name)
        if value < 0.0 or not math.isfinite(value):
            parser.error(
                "--{} must be finite and nonnegative".format(
                    name.replace("_", "-")
                )
            )
    if args.demo:
        if (
            args.predict
            or args.items
            or args.out_scored
            or args.out_model
            or args.out_report
        ):
            parser.error("--demo cannot be combined with file input or output modes")
        return
    if args.predict:
        if not args.items or not args.out_scored:
            parser.error("--predict requires --items and --out-scored")
        if args.out_model or args.out_report:
            parser.error(
                "--predict cannot be combined with --out-model or --out-report"
            )
        return
    if not args.items or not args.out_model or not args.out_report:
        parser.error("training requires --items, --out-model, and --out-report")
    if args.out_scored:
        parser.error("--out-scored requires --predict")
    output_paths = {
        os.path.normcase(os.path.abspath(args.out_model)),
        os.path.normcase(os.path.abspath(args.out_report)),
    }
    if len(output_paths) != 2:
        parser.error("--out-model and --out-report must be different paths")
    item_paths = {
        os.path.normcase(os.path.abspath(path)) for path in args.items
    }
    if output_paths.intersection(item_paths):
        parser.error("training outputs must differ from every --items path")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    try:
        if args.demo:
            run_demo(args)
            return 0
        if args.predict:
            score_items(args.predict, args.items, args.out_scored)
            return 0
        examples, input_stats = read_training_items(args.items)
        model, report, _auc = train_and_report(
            examples, input_stats, args, args.items
        )
        write_json_atomic(args.out_model, model)
        write_text_atomic(args.out_report, report)
        print(report, end="")
        print("model_json: {}".format(os.path.abspath(args.out_model)))
        print("report_text: {}".format(os.path.abspath(args.out_report)))
        return 0
    except (
        OSError,
        ValueError,
        RuntimeError,
        AssertionError,
        json.JSONDecodeError,
    ) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
