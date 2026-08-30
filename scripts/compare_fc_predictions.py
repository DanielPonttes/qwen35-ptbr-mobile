#!/usr/bin/env python3
"""Compara duas predições pareadas e calcula McNemar exato."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from fc_common import canonical_target, load_registry, parse_prediction_record, validate_target
from fc_eval import wilson_interval


def read_jsonl(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prediction_map(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for record in read_jsonl(path):
        if isinstance(record, dict) and isinstance(record.get("id"), str):
            result[record["id"]] = record
    return result


def prediction_outcomes(
    record: Any, gold: dict[str, Any], registry: dict[str, dict[str, Any]]
) -> tuple[bool, bool, bool]:
    """Retorna (exact_match, action_correct, canonical_valid)."""

    parsed, _, _ = parse_prediction_record(record)
    if parsed is None:
        return False, False, False
    errors = validate_target(parsed, registry)
    if errors:
        return False, False, False
    return (
        canonical_target(parsed) == canonical_target(gold),
        parsed.get("action") == gold.get("action"),
        True,
    )


def exact_mcnemar_p_value(b: int, c: int) -> float:
    """p bilateral exato de McNemar usando a distribuição binomial."""

    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(b, c) + 1))
    # Preserve very small p-values instead of rounding them to 0.0 in the
    # report (e.g. 2 * 0.5**66 is small but scientifically meaningful).
    return min(1.0, 2.0 * tail / (2.0**discordant))


def paired_comparison(
    dataset_path: Path,
    predictions_a_path: Path,
    predictions_b_path: Path,
    registry_path: Path,
    split: str | None,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    predictions_a = prediction_map(predictions_a_path)
    predictions_b = prediction_map(predictions_b_path)
    records = [
        record
        for record in read_jsonl(dataset_path)
        if isinstance(record, dict) and (split is None or record.get("split") == split)
    ]
    totals = {"exact_match": [0, 0], "action_accuracy": [0, 0], "canonical_valid": [0, 0]}
    discordant = {
        "exact_match": {"b_a_correct_b_incorrect": 0, "c_a_incorrect_b_correct": 0},
        "action_accuracy": {"b_a_correct_b_incorrect": 0, "c_a_incorrect_b_correct": 0},
    }
    for gold_record in records:
        identifier = gold_record["id"]
        gold = gold_record["target"]
        outcome_a = prediction_outcomes(predictions_a.get(identifier), gold, registry)
        outcome_b = prediction_outcomes(predictions_b.get(identifier), gold, registry)
        exact_a, action_a, valid_a = outcome_a
        exact_b, action_b, valid_b = outcome_b
        for key, value_a, value_b in (
            ("exact_match", exact_a, exact_b),
            ("action_accuracy", action_a, action_b),
            ("canonical_valid", valid_a, valid_b),
        ):
            totals[key][0] += int(value_a)
            totals[key][1] += int(value_b)
            if key in discordant:
                if value_a and not value_b:
                    discordant[key]["b_a_correct_b_incorrect"] += 1
                elif not value_a and value_b:
                    discordant[key]["c_a_incorrect_b_correct"] += 1

    comparisons: dict[str, Any] = {}
    for metric, (correct_a, correct_b) in totals.items():
        denominator = len(records)
        result: dict[str, Any] = {
            "correct_a": correct_a,
            "correct_b": correct_b,
            "rate_a": None if denominator == 0 else round(correct_a / denominator, 6),
            "rate_b": None if denominator == 0 else round(correct_b / denominator, 6),
            "ci95_a": wilson_interval(correct_a, denominator),
            "ci95_b": wilson_interval(correct_b, denominator),
        }
        if metric in discordant:
            b = discordant[metric]["b_a_correct_b_incorrect"]
            c = discordant[metric]["c_a_incorrect_b_correct"]
            result["mcnemar_exact"] = {
                "b_a_correct_b_incorrect": b,
                "c_a_incorrect_b_correct": c,
                "discordant_pairs": b + c,
                "p_value_two_sided": exact_mcnemar_p_value(b, c),
            }
        comparisons[metric] = result

    return {
        "dataset": str(dataset_path),
        "predictions_a": str(predictions_a_path),
        "predictions_b": str(predictions_b_path),
        "split": split,
        "pairs": len(records),
        "comparisons": comparisons,
        "statistical_methods": {
            "proportion_intervals": "Wilson score, 95%",
            "paired_test": "McNemar exact bilateral binomial test",
        },
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "data/generated/fc_dataset.jsonl")
    parser.add_argument("--predictions-a", type=Path, required=True)
    parser.add_argument("--predictions-b", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=root / "data/tools/android_tools.json")
    parser.add_argument("--split", choices=["train", "dev", "test"], default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = paired_comparison(
        args.dataset, args.predictions_a, args.predictions_b, args.registry, args.split
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
