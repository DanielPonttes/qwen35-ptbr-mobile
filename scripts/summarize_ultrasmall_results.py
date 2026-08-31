#!/usr/bin/env python3
"""Create a compact, versionable summary from benchmark metric reports."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


MODEL_ORDER = [
    "qwen25_0_5b",
    "qwen35_0_8b",
    "tinyllama_1_1b",
    "smollm2_1_7b",
    "qwen35_2b",
]
PROTOCOLS = ["official", "phrase_disjoint"]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latency_summary(path: Path) -> dict[str, float]:
    values = [
        json.loads(line)["latency_ms"]
        for line in path.open("r", encoding="utf-8")
        if line.strip()
    ]
    ordered = sorted(values)
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 3),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=root / "benchmarks" / "ultrasmall_models.json")
    parser.add_argument("--results", type=Path, default=root / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_manifest = load(args.models)
    models = {model["name"]: model for model in model_manifest["models"]}
    summary: dict[str, Any] = {
        "benchmark": model_manifest["benchmark"],
        "parameter_cap": model_manifest["parameter_cap"],
        "models": [],
        "baselines": {},
        "completeness": {"expected_model_protocol_pairs": len(MODEL_ORDER) * len(PROTOCOLS)},
    }
    completed = 0
    for name in MODEL_ORDER:
        model_entry: dict[str, Any] = {
            "name": name,
            "hf_id": models[name]["hf_id"],
            "family": models[name]["family"],
            "parameters": models[name]["parameters"],
            "revision": models[name]["revision"],
            "protocols": {},
        }
        for protocol in PROTOCOLS:
            prefix = args.results / f"fsc_ultrasmall_{name}_base_{protocol}"
            metric_path = prefix.with_suffix(".metrics.json")
            prediction_path = prefix.with_suffix(".predictions.jsonl")
            report = load(metric_path)
            model_entry["protocols"][protocol] = {
                "records": report["records"],
                "missing_predictions": report["missing_predictions"],
                "duplicate_predictions": report["duplicate_predictions"],
                "metrics": report["metrics"],
                "grouped_metrics": report["grouped_metrics"],
                "cluster_confidence_intervals_95": report[
                    "cluster_confidence_intervals_95"
                ],
                "latency": latency_summary(prediction_path),
            }
            completed += 1
        summary["models"].append(model_entry)

    for baseline in ("always_abstain", "lexical"):
        summary["baselines"][baseline] = {}
        for protocol in PROTOCOLS:
            report = load(args.results / f"fsc_ultrasmall_{baseline}_{protocol}.metrics.json")
            summary["baselines"][baseline][protocol] = {
                "records": report["records"],
                "metrics": report["metrics"],
                "grouped_metrics": report["grouped_metrics"],
                "cluster_confidence_intervals_95": report[
                    "cluster_confidence_intervals_95"
                ],
            }
    summary["completeness"]["completed_model_protocol_pairs"] = completed
    summary["completeness"]["all_predictions_complete"] = all(
        protocol["missing_predictions"] == 0 and protocol["duplicate_predictions"] == 0
        for model in summary["models"]
        for protocol in model["protocols"].values()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"summary={args.output} pairs={completed} complete={summary['completeness']['all_predictions_complete']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
