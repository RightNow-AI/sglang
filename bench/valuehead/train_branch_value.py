#!/usr/bin/env python3
"""Train and apply a per-branch correctness value model."""

import argparse
import importlib.util
import json
import math
import os
import random
import re
import sys


MODEL_TYPE = "branch_value_logistic"
FEATURE_SET_A = ("mean_logprob", "tokens_norm")
FEATURE_SET_B = (
    "answer_agreement",
    "is_leader",
    "n_distinct_answers_norm",
)
PREDICTION_FIELD = "p_correct"
TOKEN_SCALE = 512.0
PROBABILITY_EPSILON = 1e-15
MEANINGFUL_AUC_MARGIN = 0.02
GD_STEPS = 1600
GD_INITIAL_LR = 0.25
GD_DECAY_STEPS = 250.0

_ANSWER_MODE = "numeric"
_MATH_EQUIV = None


def set_answer_mode(mode):
    global _ANSWER_MODE, _MATH_EQUIV
    _ANSWER_MODE = mode
    if mode == "math" and _MATH_EQUIV is None:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "tasks",
            "math_equiv.py",
        )
        spec = importlib.util.spec_from_file_location("math_equiv", path)
        if spec is None or spec.loader is None:
            raise ValueError("could not load {}".format(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MATH_EQUIV = module


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


def answer_text(value):
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def answers_match(candidate, reference):
    if candidate is None or reference is None:
        return False
    if _ANSWER_MODE == "math":
        try:
            return bool(_MATH_EQUIV.is_equiv(candidate, reference))
        except Exception:
            return False
    candidate_number = normalize_number(candidate)
    reference_number = normalize_number(reference)
    return (
        candidate_number is not None
        and reference_number is not None
        and candidate_number == reference_number
    )


def answers_share_vote(candidate, reference):
    if _ANSWER_MODE == "math":
        return answers_match(candidate, reference)
    candidate_number = normalize_number(candidate)
    reference_number = normalize_number(reference)
    if candidate_number is not None and reference_number is not None:
        return candidate_number == reference_number
    return candidate.strip() == reference.strip()


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def positive_number(value):
    number = finite_number(value)
    return number if number is not None and number > 0.0 else None


def mapping_value(mapping, branch_id):
    if not isinstance(mapping, dict):
        return None
    if branch_id in mapping:
        return mapping[branch_id]
    text_id = str(branch_id)
    if text_id in mapping:
        return mapping[text_id]
    return None


def record_containers(record):
    containers = [(record, "record")]
    tree = record.get("tree")
    if isinstance(tree, dict):
        containers.append((tree, "tree"))
    return containers


def find_mapping(record, names):
    for container, prefix in record_containers(record):
        for name in names:
            value = container.get(name)
            if isinstance(value, dict):
                return value, "{}.{}".format(prefix, name)
    return None, None


def find_branch_metrics(record, branch_id):
    for container, prefix in record_containers(record):
        metrics = container.get("branch_metrics")
        entry = mapping_value(metrics, branch_id)
        if isinstance(entry, dict):
            return entry, "{}.branch_metrics".format(prefix)
    return None, None


def feature_a_for_branch(record, branch_id):
    """Return engine-usable features and their source, or None."""
    metrics, metrics_source = find_branch_metrics(record, branch_id)
    mean_logprob = None
    tokens = None
    mean_source = None
    token_source = None

    if metrics is not None:
        mean_logprob = finite_number(metrics.get("mean_logprob"))
        tokens = positive_number(
            metrics.get("tokens", metrics.get("token_count"))
        )
        if mean_logprob is not None:
            mean_source = metrics_source + ".mean_logprob"
        if tokens is not None:
            token_source = metrics_source + ".tokens"

    if tokens is None:
        token_map, token_map_source = find_mapping(
            record,
            (
                "tokens_spent_per_branch",
                "branch_token_counts",
                "branch_tokens",
                "tokens_per_branch",
            ),
        )
        tokens = positive_number(mapping_value(token_map, branch_id))
        if tokens is not None:
            token_source = token_map_source

    if mean_logprob is None:
        mean_map, mean_map_source = find_mapping(
            record,
            (
                "branch_mean_logprobs",
                "mean_logprobs_per_branch",
                "mean_logprob_per_branch",
                "mean_logprobs",
            ),
        )
        mean_logprob = finite_number(mapping_value(mean_map, branch_id))
        if mean_logprob is not None:
            mean_source = mean_map_source

    if mean_logprob is None and tokens is not None:
        score_map, score_map_source = find_mapping(
            record,
            (
                "final_scores",
                "cumulative_logprobs",
                "branch_cumulative_logprobs",
            ),
        )
        cumulative = finite_number(mapping_value(score_map, branch_id))
        if cumulative is not None:
            mean_logprob = cumulative / tokens
            mean_source = score_map_source + "/tokens"

    if mean_logprob is None or tokens is None:
        return None
    return {
        "features": [mean_logprob, tokens / TOKEN_SCALE],
        "mean_logprob": mean_logprob,
        "source": "{} + {}".format(mean_source, token_source),
    }


def group_answers(answered):
    representatives = []
    group_indices = []
    counts = []
    for _branch_id, answer in answered:
        group_index = None
        for index, representative in enumerate(representatives):
            if answers_share_vote(answer, representative):
                group_index = index
                break
        if group_index is None:
            group_index = len(representatives)
            representatives.append(answer)
            counts.append(0)
        counts[group_index] += 1
        group_indices.append(group_index)
    return group_indices, counts


def branch_rows_for_record(record, path, line_number, require_labels):
    branch_answers = record.get("branch_answers")
    if not isinstance(branch_answers, dict) or not branch_answers:
        return []
    problem_id = record.get("id")
    if problem_id is None:
        raise ValueError(
            "{} line {} with branch answers is missing id".format(path, line_number)
        )
    problem_id = str(problem_id)
    gold = answer_text(record.get("gold"))
    if require_labels and gold is None:
        raise ValueError(
            "{} line {} with branch answers is missing string gold".format(
                path, line_number
            )
        )

    answered = []
    for branch_id, value in branch_answers.items():
        answer = answer_text(value)
        if answer is not None:
            answered.append((str(branch_id), answer))
    if not answered:
        return []

    group_indices, counts = group_answers(answered)
    voters = len(answered)
    leader_count = max(counts)
    distinct = len(counts)
    rows = []
    for (branch_id, answer), group_index in zip(answered, group_indices):
        agreement_count = counts[group_index]
        feature_a = feature_a_for_branch(record, branch_id)
        label = None
        if gold is not None:
            label = int(answers_match(answer, gold))
        rows.append(
            {
                "problem_id": problem_id,
                "seed": record.get("seed"),
                "branch_id": branch_id,
                "answer": answer,
                "label": label,
                "feature_a": feature_a["features"] if feature_a else None,
                "feature_b": [
                    agreement_count / voters,
                    1.0 if agreement_count == leader_count else 0.0,
                    distinct / voters,
                ],
                "mean_logprob": (
                    feature_a["mean_logprob"] if feature_a else None
                ),
                "feature_a_source": feature_a["source"] if feature_a else None,
                "source_path": path,
                "source_line": line_number,
                "record_error": record.get("error"),
            }
        )
    return rows


def empty_input_stats():
    return {
        "total_records": 0,
        "records_with_rows": 0,
        "records_with_error_used": 0,
        "skipped_without_branch_rows": 0,
        "branch_rows": 0,
        "feature_a_rows": 0,
        "feature_a_sources": {},
        "sources": [],
    }


def add_rows_to_stats(stats, rows):
    stats["branch_rows"] += len(rows)
    for row in rows:
        if row["feature_a"] is not None:
            stats["feature_a_rows"] += 1
            source = row["feature_a_source"]
            stats["feature_a_sources"][source] = (
                stats["feature_a_sources"].get(source, 0) + 1
            )


def read_branch_rows(paths, require_labels):
    rows = []
    stats = empty_input_stats()
    for path in paths:
        source = {
            "path": path,
            "total_records": 0,
            "records_with_rows": 0,
            "branch_rows": 0,
        }
        with open(path, "r", encoding="utf-8-sig") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                line = raw_line.strip()
                if not line:
                    continue
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
                stats["total_records"] += 1
                source["total_records"] += 1
                record_rows = branch_rows_for_record(
                    record, path, line_number, require_labels
                )
                if record_rows:
                    stats["records_with_rows"] += 1
                    source["records_with_rows"] += 1
                    source["branch_rows"] += len(record_rows)
                    if record.get("error") is not None:
                        stats["records_with_error_used"] += 1
                    rows.extend(record_rows)
                    add_rows_to_stats(stats, record_rows)
                else:
                    stats["skipped_without_branch_rows"] += 1
        stats["sources"].append(source)
    return rows, stats


def rows_from_memory_records(records, require_labels):
    rows = []
    stats = empty_input_stats()
    source = {
        "path": "<synthetic-demo>",
        "total_records": 0,
        "records_with_rows": 0,
        "branch_rows": 0,
    }
    for line_number, record in enumerate(records, 1):
        stats["total_records"] += 1
        source["total_records"] += 1
        record_rows = branch_rows_for_record(
            record, "<synthetic-demo>", line_number, require_labels
        )
        if record_rows:
            stats["records_with_rows"] += 1
            source["records_with_rows"] += 1
            source["branch_rows"] += len(record_rows)
            rows.extend(record_rows)
            add_rows_to_stats(stats, record_rows)
        else:
            stats["skipped_without_branch_rows"] += 1
    stats["sources"].append(source)
    return rows, stats


def select_feature_set(rows, requested):
    available_a = sum(row["feature_a"] is not None for row in rows)
    if requested == "a":
        if available_a != len(rows):
            raise ValueError(
                "feature set A requested, but only {}/{} branch rows have "
                "per-branch mean logprob and token count".format(
                    available_a, len(rows)
                )
            )
        return "A", FEATURE_SET_A
    if requested == "b":
        return "B", FEATURE_SET_B
    if available_a == len(rows):
        return "A", FEATURE_SET_A
    return "B", FEATURE_SET_B


def fit_standardizer(rows):
    count = len(rows)
    width = len(rows[0])
    means = [
        sum(row[index] for row in rows) / count for index in range(width)
    ]
    variances = [
        sum((row[index] - means[index]) ** 2 for row in rows) / count
        for index in range(width)
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
    """Fit mean log-loss plus l2/(2*n)*sum(weights^2)."""
    count = len(rows)
    width = len(rows[0])
    weights = [0.0] * width
    bias = 0.0
    for step in range(GD_STEPS):
        bias_gradient = 0.0
        weight_gradients = [0.0] * width
        for row, label in zip(rows, labels):
            probability = sigmoid(
                bias + sum(weight * value for weight, value in zip(weights, row))
            )
            error = probability - label
            bias_gradient += error
            for index, value in enumerate(row):
                weight_gradients[index] += error * value
        learning_rate = GD_INITIAL_LR / math.sqrt(
            1.0 + step / GD_DECAY_STEPS
        )
        bias -= learning_rate * bias_gradient / count
        for index in range(width):
            gradient = (
                weight_gradients[index] / count + l2 * weights[index] / count
            )
            weights[index] -= learning_rate * gradient
    return bias, weights


def predict_probabilities(rows, bias, weights):
    return [
        sigmoid(
            bias + sum(weight * value for weight, value in zip(weights, row))
        )
        for row in rows
    ]


def rank_auc(labels, scores):
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0])
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


def log_loss(labels, probabilities):
    total = 0.0
    for label, probability in zip(labels, probabilities):
        clipped = min(
            max(probability, PROBABILITY_EPSILON), 1.0 - PROBABILITY_EPSILON
        )
        total -= label * math.log(clipped) + (1 - label) * math.log(1.0 - clipped)
    return total / len(labels)


def binary_accuracy(labels, probabilities):
    correct = sum(
        (probability >= 0.5) == bool(label)
        for label, probability in zip(labels, probabilities)
    )
    return correct / len(labels), correct


def assign_group_folds(groups, requested_folds, seed):
    by_group = {}
    for index, group in enumerate(groups):
        by_group.setdefault(group, []).append(index)
    effective_folds = min(requested_folds, len(by_group))
    if effective_folds < 2:
        raise ValueError(
            "at least 2 distinct problem ids are required for grouped cross-validation"
        )
    grouped = list(by_group.items())
    random.Random(seed).shuffle(grouped)
    grouped.sort(key=lambda item: len(item[1]), reverse=True)
    fold_indices = [[] for _unused in range(effective_folds)]
    fold_group_counts = [0] * effective_folds
    fold_row_counts = [0] * effective_folds
    for _group, indices in grouped:
        fold = min(
            range(effective_folds),
            key=lambda value: (
                fold_row_counts[value],
                fold_group_counts[value],
                value,
            ),
        )
        fold_indices[fold].extend(indices)
        fold_group_counts[fold] += 1
        fold_row_counts[fold] += len(indices)
    return fold_indices, fold_group_counts, effective_folds


def cross_validate(rows, labels, groups, baseline_scores, requested_folds, l2, seed):
    fold_indices, fold_group_counts, effective_folds = assign_group_folds(
        groups, requested_folds, seed
    )
    all_indices = set(range(len(rows)))
    out_of_fold = [None] * len(rows)
    fold_results = []
    for fold_number, test_indices in enumerate(fold_indices, 1):
        test_indices = sorted(test_indices)
        train_indices = sorted(all_indices.difference(test_indices))
        train_rows = [rows[index] for index in train_indices]
        train_labels = [labels[index] for index in train_indices]
        test_rows = [rows[index] for index in test_indices]
        test_labels = [labels[index] for index in test_indices]
        means, stds = fit_standardizer(train_rows)
        standardized_train = standardize_rows(train_rows, means, stds)
        standardized_test = standardize_rows(test_rows, means, stds)
        bias, weights = fit_logistic(standardized_train, train_labels, l2)
        probabilities = predict_probabilities(standardized_test, bias, weights)
        for index, probability in zip(test_indices, probabilities):
            out_of_fold[index] = probability
        accuracy, correct_n = binary_accuracy(test_labels, probabilities)
        fold_baseline = None
        if baseline_scores is not None:
            fold_baseline = rank_auc(
                test_labels, [baseline_scores[index] for index in test_indices]
            )
        fold_results.append(
            {
                "fold": fold_number,
                "group_n": fold_group_counts[fold_number - 1],
                "train_n": len(train_indices),
                "test_n": len(test_indices),
                "positive_n": sum(test_labels),
                "correct_n": correct_n,
                "accuracy": accuracy,
                "auc": rank_auc(test_labels, probabilities),
                "baseline_auc": fold_baseline,
                "log_loss": log_loss(test_labels, probabilities),
            }
        )
    if any(value is None for value in out_of_fold):
        raise RuntimeError("cross-validation did not score every branch row")
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
                "observed_rate": positive_n / count if count else None,
            }
        )
    return rows


def format_float(value, digits=6):
    return "NA" if value is None else ("{:.%df}" % digits).format(value)


def decisive_finding(model_auc, baseline_auc):
    if model_auc is None:
        return (
            "DECISIVE FINDING: INCONCLUSIVE. The evaluated rows do not contain "
            "both correctness classes, so AUC is undefined."
        )
    if baseline_auc is None:
        return (
            "DECISIVE FINDING: INCONCLUSIVE. Raw mean_logprob is not available "
            "for every evaluated branch, so the required same-row comparison "
            "cannot be made."
        )
    delta = model_auc - baseline_auc
    if delta < MEANINGFUL_AUC_MARGIN:
        return (
            "DECISIVE FINDING: RETIRE THIS LEVER. The learned model did not beat "
            "raw mean_logprob by the predefined meaningful AUC margin of "
            "{:.3f}. Learned AUC={}, raw logprob AUC={}, delta={}."
        ).format(
            MEANINGFUL_AUC_MARGIN,
            format_float(model_auc),
            format_float(baseline_auc),
            format_float(delta),
        )
    return (
        "DECISIVE FINDING: LEARNED SIGNAL CLEARS THE COMPARISON. The learned "
        "model beat raw mean_logprob by at least {:.3f} AUC. Learned AUC={}, "
        "raw logprob AUC={}, delta={}."
    ).format(
        MEANINGFUL_AUC_MARGIN,
        format_float(model_auc),
        format_float(baseline_auc),
        format_float(delta),
    )


def build_report(
    raw_rows,
    stats,
    feature_set,
    feature_names,
    labels,
    probabilities,
    baseline_scores,
    fold_results,
    effective_folds,
    args,
):
    model_auc = rank_auc(labels, probabilities)
    baseline_auc = (
        rank_auc(labels, baseline_scores) if baseline_scores is not None else None
    )
    accuracy, correct_n = binary_accuracy(labels, probabilities)
    lines = [
        "BRANCH VALUE MODEL REPORT",
        decisive_finding(model_auc, baseline_auc),
        "",
        "INPUT",
        "records={} records_with_branch_rows={} branch_rows={} positive_n={} "
        "negative_n={}".format(
            stats["total_records"],
            stats["records_with_rows"],
            len(raw_rows),
            sum(labels),
            len(labels) - sum(labels),
        ),
        "records_with_error_used={} skipped_without_branch_rows={}".format(
            stats["records_with_error_used"],
            stats["skipped_without_branch_rows"],
        ),
        "problem_groups={} requested_folds={} effective_folds={} seed={}".format(
            len(set(row["problem_id"] for row in raw_rows)),
            args.folds,
            effective_folds,
            args.seed,
        ),
        "answer_mode={}".format(args.answer_mode),
        "",
        "FEATURES",
        "selected_set={} engine_usable={} names={}".format(
            feature_set,
            str(feature_set == "A").lower(),
            ",".join(feature_names),
        ),
        "feature_a_available_rows={}/{} token_scale={}".format(
            stats["feature_a_rows"], len(raw_rows), format_float(TOKEN_SCALE, 1)
        ),
    ]
    if stats["feature_a_sources"]:
        lines.append(
            "feature_a_sources={}".format(
                "; ".join(
                    "{}:{}".format(source, count)
                    for source, count in sorted(
                        stats["feature_a_sources"].items()
                    )
                )
            )
        )
    if feature_set == "B":
        lines.append(
            "set_b_note=analysis-only because agreement and leader status need "
            "sibling final answers"
        )
    lines.extend(
        [
            "",
            "GROUPED OUT-OF-FOLD EVALUATION",
            "fold  groups  train_n  test_n  positive_n  accuracy  model_auc  "
            "raw_logprob_auc  log_loss",
        ]
    )
    for result in fold_results:
        lines.append(
            "{fold:>4}  {group_n:>6}  {train_n:>7}  {test_n:>6}  "
            "{positive_n:>10}  {accuracy:>8}  {auc:>9}  "
            "{baseline_auc:>15}  {log_loss:>8}".format(
                fold=result["fold"],
                group_n=result["group_n"],
                train_n=result["train_n"],
                test_n=result["test_n"],
                positive_n=result["positive_n"],
                accuracy=format_float(result["accuracy"]),
                auc=format_float(result["auc"]),
                baseline_auc=format_float(result["baseline_auc"]),
                log_loss=format_float(result["log_loss"]),
            )
        )
    delta = (
        model_auc - baseline_auc
        if model_auc is not None and baseline_auc is not None
        else None
    )
    lines.extend(
        [
            "overall_oof: n={} correct_n={} accuracy={} auc={} log_loss={}".format(
                len(labels),
                correct_n,
                format_float(accuracy),
                format_float(model_auc),
                format_float(log_loss(labels, probabilities)),
            ),
            "raw_mean_logprob_same_rows: auc={}".format(
                format_float(baseline_auc)
            ),
            "learned_minus_raw_auc={} meaningful_margin={}".format(
                format_float(delta), format_float(MEANINGFUL_AUC_MARGIN)
            ),
            "grouping_rule=all seeds and branches for the same problem id stay "
            "in one fold",
            "",
            "CALIBRATION (out-of-fold, 10 equal-width probability bins)",
            "bin         n  positive_n  mean_predicted  observed_rate",
        ]
    )
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
                format_float(row["observed_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "TRAINING",
            "optimizer=full-batch gradient descent steps={} initial_lr={} "
            "schedule=lr/sqrt(1+step/{}) l2={}".format(
                GD_STEPS,
                format_float(GD_INITIAL_LR),
                format_float(GD_DECAY_STEPS, 1),
                format_float(args.l2),
            ),
            "l2_objective=mean_log_loss + l2/(2*n)*sum(weights^2); bias is "
            "not regularized",
            "label=branch answer is equivalent to gold",
            "caveat=the model is trained on final branch statistics; an engine "
            "using it during generation applies it to partial branches",
        ]
    )
    return "\n".join(lines) + "\n", model_auc, baseline_auc


def train_and_report(raw_rows, stats, args):
    if not raw_rows:
        raise ValueError("no answered branch rows were found")
    if any(row["label"] is None for row in raw_rows):
        raise ValueError("training rows must all have correctness labels")
    feature_set, feature_names = select_feature_set(
        raw_rows, args.feature_set
    )
    feature_key = "feature_a" if feature_set == "A" else "feature_b"
    rows = [row[feature_key] for row in raw_rows]
    labels = [row["label"] for row in raw_rows]
    groups = [row["problem_id"] for row in raw_rows]
    baseline_scores = None
    if all(row["mean_logprob"] is not None for row in raw_rows):
        baseline_scores = [row["mean_logprob"] for row in raw_rows]
    probabilities, fold_results, effective_folds = cross_validate(
        rows,
        labels,
        groups,
        baseline_scores,
        args.folds,
        args.l2,
        args.seed,
    )
    means, stds = fit_standardizer(rows)
    standardized = standardize_rows(rows, means, stds)
    bias, weights = fit_logistic(standardized, labels, args.l2)
    model = {
        "type": MODEL_TYPE,
        "feature_names": list(feature_names),
        "weights": weights,
        "bias": bias,
        "standardization": {"mean": means, "std": stds},
        "engine_usable": feature_set == "A",
    }
    report, model_auc, baseline_auc = build_report(
        raw_rows,
        stats,
        feature_set,
        feature_names,
        labels,
        probabilities,
        baseline_scores,
        fold_results,
        effective_folds,
        args,
    )
    return model, report, model_auc, baseline_auc


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


def load_model(path):
    with open(path, "r", encoding="utf-8") as handle:
        model = json.load(handle)
    if not isinstance(model, dict) or model.get("type") != MODEL_TYPE:
        raise ValueError("{} is not a branch value logistic model".format(path))
    feature_names = model.get("feature_names")
    if feature_names == list(FEATURE_SET_A):
        feature_set = "A"
    elif feature_names == list(FEATURE_SET_B):
        feature_set = "B"
    else:
        raise ValueError("{} has unsupported feature_names".format(path))
    if model.get("engine_usable") is not (feature_set == "A"):
        raise ValueError("{} has inconsistent engine_usable".format(path))
    standardization = model.get("standardization")
    if not isinstance(standardization, dict):
        raise ValueError("{} is missing standardization".format(path))
    try:
        means = [float(value) for value in standardization["mean"]]
        stds = [float(value) for value in standardization["std"]]
        weights = [float(value) for value in model["weights"]]
        bias = float(model["bias"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("{} has invalid model parameters".format(path)) from exc
    width = len(feature_names)
    if not (len(means) == len(stds) == len(weights) == width):
        raise ValueError("{} has inconsistent parameter lengths".format(path))
    values = means + stds + weights + [bias]
    if not all(math.isfinite(value) for value in values) or any(
        value < 0.0 for value in stds
    ):
        raise ValueError("{} has non-finite parameters or negative std".format(path))
    return feature_set, feature_names, means, stds, weights, bias


def score_rows(model_path, item_paths, out_path):
    feature_set, feature_names, means, stds, weights, bias = load_model(
        model_path
    )
    input_abspaths = {
        os.path.normcase(os.path.abspath(path)) for path in item_paths
    }
    if os.path.normcase(os.path.abspath(out_path)) in input_abspaths:
        raise ValueError("--out-predictions must differ from every --items path")
    raw_rows, _stats = read_branch_rows(item_paths, require_labels=False)
    ensure_parent(out_path)
    temporary = out_path + ".tmp"
    scored_n = 0
    missing_n = 0
    feature_key = "feature_a" if feature_set == "A" else "feature_b"
    with open(temporary, "w", encoding="utf-8", newline="\n") as output:
        for row in raw_rows:
            features = row[feature_key]
            probability = None
            prediction_error = None
            if features is None:
                prediction_error = "missing_feature_set_a_inputs"
                missing_n += 1
            else:
                standardized = standardize_rows([features], means, stds)
                probability = predict_probabilities(
                    standardized, bias, weights
                )[0]
                scored_n += 1
            output_row = {
                "id": row["problem_id"],
                "seed": row["seed"],
                "branch_id": row["branch_id"],
                "answer": row["answer"],
                "feature_names": feature_names,
                "features": features,
                PREDICTION_FIELD: probability,
                "prediction_error": prediction_error,
                "source_path": row["source_path"],
                "source_line": row["source_line"],
            }
            if row["label"] is not None:
                output_row["correct"] = bool(row["label"])
            output.write(
                json.dumps(
                    output_row,
                    separators=(",", ":"),
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, out_path)
    print(
        "prediction: branch_rows={} scored_n={} missing_feature_n={} field={} "
        "out={}".format(
            len(raw_rows),
            scored_n,
            missing_n,
            PREDICTION_FIELD,
            os.path.abspath(out_path),
        )
    )


def synthetic_demo_records(seed, problem_count=240):
    rng = random.Random(seed)
    records = []
    for problem_index in range(problem_count):
        problem_bias = rng.gauss(0.0, 0.25)
        gold = str(1000 + problem_index)
        for run_seed in (0, 1):
            branch_answers = {}
            final_scores = {}
            tokens_spent = {}
            for branch_index in range(6):
                tokens = rng.randint(32, 512)
                mean_logprob = rng.gauss(-1.45, 0.82)
                planted_logit = (
                    3.0
                    + 2.35 * mean_logprob
                    + 3.4 * (tokens / TOKEN_SCALE - 0.5)
                    + problem_bias
                )
                correct = rng.random() < sigmoid(planted_logit)
                if correct:
                    answer = gold
                else:
                    answer = "wrong-{}-{}".format(
                        problem_index, branch_index % 3
                    )
                branch_id = str(branch_index)
                branch_answers[branch_id] = answer
                tokens_spent[branch_id] = tokens
                final_scores[branch_id] = mean_logprob * tokens
            records.append(
                {
                    "id": "demo-{:04d}".format(problem_index),
                    "seed": run_seed,
                    "gold": gold,
                    "branch_answers": branch_answers,
                    "tokens_spent_per_branch": tokens_spent,
                    "final_scores": final_scores,
                    "error": None,
                }
            )
    return records


def run_demo(args):
    records = synthetic_demo_records(args.seed)
    rows, stats = rows_from_memory_records(records, require_labels=True)
    model, report, auc, _baseline_auc = train_and_report(rows, stats, args)
    print(report, end="")
    if model["feature_names"] != list(FEATURE_SET_A):
        raise AssertionError("demo did not select feature set A")
    if auc is None or auc <= 0.8:
        raise AssertionError(
            "demo held-out grouped OOF AUC must exceed 0.8; got {}".format(
                format_float(auc)
            )
        )
    print(
        "DEMO PASS: grouped held-out OOF AUC {} > 0.800000 "
        "(branch_rows={} problem_groups={})".format(
            format_float(auc),
            len(rows),
            len(set(row["problem_id"] for row in rows)),
        )
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train or apply a grouped-CV branch correctness model."
    )
    parser.add_argument("--items", nargs="+", help="cascade per-item JSONL path(s)")
    parser.add_argument("--out-model", help="trained model JSON output")
    parser.add_argument("--out-report", help="text evaluation report output")
    parser.add_argument(
        "--feature-set",
        choices=("auto", "a", "b"),
        default="auto",
        help="auto prefers A only when every answered branch has live features",
    )
    parser.add_argument(
        "--answer-mode", choices=("numeric", "math"), default="numeric"
    )
    parser.add_argument(
        "--predict", metavar="MODEL_JSON", help="score branch rows with a model"
    )
    parser.add_argument(
        "--out-predictions", help="per-branch JSONL with appended p_correct"
    )
    parser.add_argument(
        "--demo", action="store_true", help="run the seeded grouped-CV self-test"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1729)
    return parser


def validate_args(parser, args):
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.l2 < 0.0 or not math.isfinite(args.l2):
        parser.error("--l2 must be finite and nonnegative")
    if args.demo:
        if any(
            (
                args.items,
                args.out_model,
                args.out_report,
                args.predict,
                args.out_predictions,
            )
        ):
            parser.error("--demo cannot be combined with file input or output modes")
        if args.feature_set == "b":
            parser.error("--demo requires feature set auto or a")
        return
    if args.predict:
        if not args.items or not args.out_predictions:
            parser.error("--predict requires --items and --out-predictions")
        if args.out_model or args.out_report:
            parser.error(
                "--predict cannot be combined with --out-model or --out-report"
            )
        return
    if not args.items or not args.out_model or not args.out_report:
        parser.error("training requires --items, --out-model, and --out-report")
    if args.out_predictions:
        parser.error("--out-predictions requires --predict")
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
        set_answer_mode(args.answer_mode)
        if args.demo:
            run_demo(args)
            return 0
        if args.predict:
            score_rows(args.predict, args.items, args.out_predictions)
            return 0
        rows, stats = read_branch_rows(args.items, require_labels=True)
        model, report, _auc, _baseline_auc = train_and_report(rows, stats, args)
        write_json_atomic(args.out_model, model)
        write_text_atomic(args.out_report, report)
        print(report, end="")
        print("model_json: {}".format(os.path.abspath(args.out_model)))
        print("report_text: {}".format(os.path.abspath(args.out_report)))
        return 0
    except (OSError, ValueError, AssertionError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
