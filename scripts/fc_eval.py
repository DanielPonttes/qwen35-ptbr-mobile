#!/usr/bin/env python3
"""Calcula métricas reproduzíveis para predições de function calling."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fc_common import canonical_target, load_registry, parse_prediction_record, validate_target


def read_jsonl(path: Path) -> list[Any]:
    records: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                records.append({"__line__": line_number, "__error__": exc.msg})
    return records


def rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def wilson_interval(successes: int, denominator: int, z: float = 1.959963984540054) -> dict[str, float] | None:
    """Intervalo de confiança de 95% pelo score de Wilson."""

    if denominator == 0:
        return None
    proportion = successes / denominator
    denominator_adjusted = 1.0 + (z * z / denominator)
    center = (proportion + (z * z / (2.0 * denominator))) / denominator_adjusted
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / denominator
            + (z * z / (4.0 * denominator * denominator))
        )
        / denominator_adjusted
    )
    return {
        "lower": round(max(0.0, center - half_width), 6),
        "upper": round(min(1.0, center + half_width), 6),
    }


def bootstrap_abstention_f1(
    rows: list[tuple[bool, bool]], samples: int = 2000, seed: int = 20260830
) -> dict[str, float] | None:
    """IC95% bootstrap determinístico para F1 de abstenção."""

    if not rows:
        return None
    generator = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        true_positive = false_positive = false_negative = 0
        for _ in rows:
            gold_abstain, predicted_abstain = rows[generator.randrange(len(rows))]
            if gold_abstain and predicted_abstain:
                true_positive += 1
            elif not gold_abstain and predicted_abstain:
                false_positive += 1
            elif gold_abstain and not predicted_abstain:
                false_negative += 1
        precision = rate(true_positive, true_positive + false_positive)
        recall = rate(true_positive, true_positive + false_negative)
        if precision is None or recall is None or precision + recall == 0:
            values.append(0.0)
        else:
            values.append(2.0 * precision * recall / (precision + recall))
    values.sort()
    lower_index = int(0.025 * (len(values) - 1))
    upper_index = int(0.975 * (len(values) - 1))
    return {"lower": round(values[lower_index], 6), "upper": round(values[upper_index], 6)}


def _f1_from_counts(true_positive: int, false_positive: int, false_negative: int) -> float:
    precision = rate(true_positive, true_positive + false_positive)
    recall = rate(true_positive, true_positive + false_negative)
    if precision is None or recall is None or precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def summarize_groups(
    group_rows: dict[str, list[tuple[bool, bool, bool, bool]]]
) -> dict[str, Any]:
    """Summarize item outcomes after giving each leakage group equal weight.

    A phrase can occur for many speakers. Item-level confidence intervals then
    overstate the effective sample size. The macro values below are descriptive
    group-level statistics; cluster bootstrap intervals are reported separately.
    """

    if not group_rows:
        return {
            "groups": 0,
            "records": 0,
            "group_size": {},
            "macro": {},
            "strict_group_exact_match_rate": None,
        }
    exact_rates: list[float] = []
    action_rates: list[float] = []
    abstention_f1_values: list[float] = []
    strict_exact = 0
    sizes = [len(rows) for rows in group_rows.values()]
    for rows in group_rows.values():
        exact_rates.append(sum(int(row[0]) for row in rows) / len(rows))
        action_rates.append(sum(int(row[1]) for row in rows) / len(rows))
        tp = sum(int(row[2] and row[3]) for row in rows)
        fp = sum(int((not row[2]) and row[3]) for row in rows)
        fn = sum(int(row[2] and (not row[3])) for row in rows)
        abstention_f1_values.append(_f1_from_counts(tp, fp, fn))
        if all(row[0] for row in rows):
            strict_exact += 1

    return {
        "groups": len(group_rows),
        "records": sum(sizes),
        "group_size": {
            "min": min(sizes),
            "median": sorted(sizes)[len(sizes) // 2],
            "max": max(sizes),
        },
        "macro": {
            "exact_match_rate": round(sum(exact_rates) / len(exact_rates), 6),
            "action_accuracy": round(sum(action_rates) / len(action_rates), 6),
            "abstention_f1": round(
                sum(abstention_f1_values) / len(abstention_f1_values), 6
            ),
        },
        "strict_group_exact_match_rate": rate(strict_exact, len(group_rows)),
    }


def bootstrap_group_intervals(
    group_rows: dict[str, list[tuple[bool, bool, bool, bool]]],
    samples: int = 2000,
    seed: int = 20260830,
) -> dict[str, dict[str, float] | None]:
    """Cluster bootstrap over leakage groups, not individual records."""

    if not group_rows:
        return {
            "exact_match_rate": None,
            "action_accuracy": None,
            "abstention_f1": None,
        }
    generator = random.Random(seed)
    groups = list(group_rows.values())
    exact_values: list[float] = []
    action_values: list[float] = []
    abstention_values: list[float] = []
    for _ in range(samples):
        sampled = [groups[generator.randrange(len(groups))] for _ in groups]
        total = sum(len(rows) for rows in sampled)
        exact = sum(sum(int(row[0]) for row in rows) for rows in sampled)
        action = sum(sum(int(row[1]) for row in rows) for rows in sampled)
        tp = sum(sum(int(row[2] and row[3]) for row in rows) for rows in sampled)
        fp = sum(sum(int((not row[2]) and row[3]) for row in rows) for rows in sampled)
        fn = sum(sum(int(row[2] and (not row[3])) for row in rows) for rows in sampled)
        exact_values.append(exact / total)
        action_values.append(action / total)
        abstention_values.append(_f1_from_counts(tp, fp, fn))

    def interval(values: list[float]) -> dict[str, float]:
        values.sort()
        lower_index = int(0.025 * (len(values) - 1))
        upper_index = int(0.975 * (len(values) - 1))
        return {
            "lower": round(values[lower_index], 6),
            "upper": round(values[upper_index], 6),
        }

    return {
        "exact_match_rate": interval(exact_values),
        "action_accuracy": interval(action_values),
        "abstention_f1": interval(abstention_values),
    }


def evaluate(
    dataset_path: Path,
    predictions_path: Path,
    registry_path: Path,
    split: str | None = None,
    group_field: str = "template_id",
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    dataset_records = read_jsonl(dataset_path)
    if split is not None:
        dataset_records = [record for record in dataset_records if record.get("split") == split]
    prediction_records = read_jsonl(predictions_path)
    dataset = {
        record.get("id"): record
        for record in dataset_records
        if isinstance(record, dict) and isinstance(record.get("id"), str)
    }
    predictions: dict[str, Any] = {}
    duplicate_predictions = 0
    for record in prediction_records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            continue
        if record["id"] in predictions:
            duplicate_predictions += 1
        predictions[record["id"]] = record

    total = len(dataset)
    parsed_count = 0
    json_valid_count = 0
    canonical_valid_count = 0
    exact_count = 0
    action_correct = 0
    call_total = 0
    tool_correct = 0
    argument_exact = 0
    argument_exact_given_tool = 0
    abstain_total = 0
    abstain_true_positive = 0
    abstain_false_positive = 0
    abstain_false_negative = 0
    invalid_examples: list[dict[str, Any]] = []
    by_tool: dict[str, Counter[str]] = defaultdict(Counter)
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    by_mapping: dict[str, Counter[str]] = defaultdict(Counter)
    abstention_rows: list[tuple[bool, bool]] = []
    group_rows: dict[str, list[tuple[bool, bool, bool, bool]]] = defaultdict(list)

    for identifier, gold_record in dataset.items():
        gold = gold_record.get("target")
        split = gold_record.get("split", "unknown")
        metadata = gold_record.get("metadata", {})
        mapping_group = (
            metadata.get("mapping_rule", "unknown")
            if isinstance(metadata, dict)
            else "unknown"
        )
        by_mapping[mapping_group]["total"] += 1
        gold_action = gold.get("action") if isinstance(gold, dict) else None
        if gold_action == "call":
            call_total += 1
            by_tool[gold.get("tool")]["total"] += 1
            by_mapping[mapping_group]["call_gold"] += 1
        elif gold_action == "abstain":
            abstain_total += 1
            by_mapping[mapping_group]["abstain_gold"] += 1

        prediction_record = predictions.get(identifier)
        if prediction_record is None:
            parsed = None
            is_json = False
            parse_error = "predição ausente"
        else:
            parsed, is_json, parse_error = parse_prediction_record(prediction_record)
        if parsed is not None:
            parsed_count += 1
            if is_json:
                json_valid_count += 1
        validation_errors = validate_target(parsed, registry) if parsed is not None else [parse_error or "predição nula"]
        valid = parsed is not None and not validation_errors
        if valid:
            canonical_valid_count += 1
            by_mapping[mapping_group]["valid"] += 1
        if not valid:
            if len(invalid_examples) < 25:
                invalid_examples.append({"id": identifier, "errors": validation_errors})
            predicted_action = None
        else:
            predicted_action = parsed["action"]
            by_split[split]["valid"] += 1

        item_exact = valid and canonical_target(parsed) == canonical_target(gold)
        item_action = valid and predicted_action == gold_action
        item_predicted_abstain = valid and predicted_action == "abstain"
        group_id = (
            metadata.get(group_field)
            if isinstance(metadata, dict)
            else None
        )
        if not isinstance(group_id, str) or not group_id:
            group_id = identifier
        group_rows[group_id].append(
            (
                item_exact,
                item_action,
                gold_action == "abstain",
                item_predicted_abstain,
            )
        )
        if item_exact:
            exact_count += 1
            by_split[split]["exact"] += 1
            by_mapping[mapping_group]["exact"] += 1
        if item_action:
            action_correct += 1
            by_mapping[mapping_group]["action_correct"] += 1
        abstention_rows.append(
            (gold_action == "abstain", valid and predicted_action == "abstain")
        )
        if gold_action == "call":
            if valid and parsed["tool"] == gold.get("tool"):
                tool_correct += 1
                by_tool[gold["tool"]]["tool_correct"] += 1
                by_mapping[mapping_group]["tool_correct"] += 1
                if parsed["arguments"] == gold.get("arguments", {}):
                    argument_exact_given_tool += 1
                    by_tool[gold["tool"]]["argument_exact_given_tool"] += 1
                    by_mapping[mapping_group]["argument_exact_given_tool"] += 1
            if valid and parsed["tool"] == gold.get("tool") and parsed["arguments"] == gold.get("arguments", {}):
                argument_exact += 1
                by_tool[gold["tool"]]["argument_exact"] += 1
                by_mapping[mapping_group]["argument_exact"] += 1
        if gold_action == "abstain":
            if item_predicted_abstain:
                abstain_true_positive += 1
                by_mapping[mapping_group]["abstain_true_positive"] += 1
            else:
                abstain_false_negative += 1
                by_mapping[mapping_group]["abstain_false_negative"] += 1
        elif item_predicted_abstain:
            abstain_false_positive += 1
            by_mapping[mapping_group]["abstain_false_positive"] += 1

    abstention_precision = rate(abstain_true_positive, abstain_true_positive + abstain_false_positive)
    abstention_recall = rate(abstain_true_positive, abstain_true_positive + abstain_false_negative)
    if abstention_precision is None or abstention_recall is None or abstention_precision + abstention_recall == 0:
        abstention_f1 = None
    else:
        abstention_f1 = round(
            2 * abstention_precision * abstention_recall / (abstention_precision + abstention_recall),
            6,
        )

    per_tool = {}
    for tool in sorted(registry):
        counts = by_tool[tool]
        per_tool[tool] = {
            "count": counts["total"],
            "tool_selection_accuracy": rate(counts["tool_correct"], counts["total"]),
            "argument_exact_accuracy": rate(counts["argument_exact"], counts["total"]),
            "argument_exact_given_tool": rate(
                counts["argument_exact_given_tool"], counts["tool_correct"]
            ),
        }

    per_mapping = {}
    for mapping_rule in sorted(by_mapping):
        counts = by_mapping[mapping_rule]
        group_total = counts["total"]
        group_call_total = counts["call_gold"]
        group_tool_correct = counts["tool_correct"]
        group_abstain_tp = counts["abstain_true_positive"]
        group_abstain_fp = counts["abstain_false_positive"]
        group_abstain_fn = counts["abstain_false_negative"]
        group_precision = rate(group_abstain_tp, group_abstain_tp + group_abstain_fp)
        group_recall = rate(group_abstain_tp, group_abstain_tp + group_abstain_fn)
        if (
            group_precision is None
            or group_recall is None
            or group_precision + group_recall == 0
        ):
            group_f1 = None
        else:
            group_f1 = round(
                2 * group_precision * group_recall / (group_precision + group_recall),
                6,
            )
        per_mapping[mapping_rule] = {
            "records": group_total,
            "call_gold": group_call_total,
            "abstain_gold": counts["abstain_gold"],
            "canonical_valid_rate": rate(counts["valid"], group_total),
            "exact_match_rate": rate(counts["exact"], group_total),
            "action_accuracy": rate(counts["action_correct"], group_total),
            "tool_selection_accuracy": rate(group_tool_correct, group_call_total),
            "argument_exact_accuracy": rate(
                counts["argument_exact"], group_call_total
            ),
            "argument_exact_given_tool": rate(
                counts["argument_exact_given_tool"], group_tool_correct
            ),
            "abstention_precision": group_precision,
            "abstention_recall": group_recall,
            "abstention_f1": group_f1,
        }

    metrics = {
        "parseable_rate": rate(parsed_count, total),
        "json_valid_rate": rate(json_valid_count, total),
        "canonical_valid_rate": rate(canonical_valid_count, total),
        "exact_match_rate": rate(exact_count, total),
        "action_accuracy": rate(action_correct, total),
        "tool_selection_accuracy": rate(tool_correct, call_total),
        "argument_exact_accuracy": rate(argument_exact, call_total),
        "argument_exact_given_tool": rate(argument_exact_given_tool, tool_correct),
        "abstention_precision": abstention_precision,
        "abstention_recall": abstention_recall,
        "abstention_f1": abstention_f1,
    }
    confidence_intervals_95 = {
        "parseable_rate": wilson_interval(parsed_count, total),
        "json_valid_rate": wilson_interval(json_valid_count, total),
        "canonical_valid_rate": wilson_interval(canonical_valid_count, total),
        "exact_match_rate": wilson_interval(exact_count, total),
        "action_accuracy": wilson_interval(action_correct, total),
        "tool_selection_accuracy": wilson_interval(tool_correct, call_total),
        "argument_exact_accuracy": wilson_interval(argument_exact, call_total),
        "argument_exact_given_tool": wilson_interval(argument_exact_given_tool, tool_correct),
        "abstention_precision": wilson_interval(
            abstain_true_positive, abstain_true_positive + abstain_false_positive
        ),
        "abstention_recall": wilson_interval(
            abstain_true_positive, abstain_true_positive + abstain_false_negative
        ),
        "abstention_f1": bootstrap_abstention_f1(abstention_rows),
    }

    grouped_metrics = summarize_groups(group_rows)
    return {
        "dataset": str(dataset_path),
        "predictions": str(predictions_path),
        "split": split,
        "group_field": group_field,
        "records": total,
        "missing_predictions": len(set(dataset) - set(predictions)),
        "extra_predictions": len(set(predictions) - set(dataset)),
        "duplicate_predictions": duplicate_predictions,
        "invalid_examples_sample": invalid_examples,
        "metrics": metrics,
        "confidence_intervals_95": confidence_intervals_95,
        "grouped_metrics": grouped_metrics,
        "cluster_confidence_intervals_95": bootstrap_group_intervals(group_rows),
        "statistical_methods": {
            "proportion_intervals": "Wilson score, 95%",
            "abstention_f1_interval": "deterministic bootstrap percentile, 2000 resamples, seed 20260830",
        },
        "counts": {
            "call_gold": call_total,
            "abstain_gold": abstain_total,
            "abstain_true_positive": abstain_true_positive,
            "abstain_false_positive": abstain_false_positive,
            "abstain_false_negative": abstain_false_negative,
        },
        "per_tool": per_tool,
        "per_mapping_rule": per_mapping,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "data" / "generated" / "fc_dataset.jsonl",
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=root / "data" / "tools" / "android_tools.json",
    )
    parser.add_argument("--split", choices=["train", "dev", "test"], default=None)
    parser.add_argument(
        "--group-field",
        default="template_id",
        help="metadata field used for cluster-aware summaries and bootstrap",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate(
        args.dataset,
        args.predictions,
        args.registry,
        args.split,
        args.group_field,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
