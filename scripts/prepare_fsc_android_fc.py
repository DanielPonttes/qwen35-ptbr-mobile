#!/usr/bin/env python3
"""Derive an auditable command-routing benchmark from Fluent Speech Commands.

FSC's native labels describe smart-home/assistant intents, not the benchmark
operations. This script therefore applies a deliberately conservative,
deterministic mapping:

* activate music      -> media_control(play)
* deactivate music    -> media_control(pause)
* increase volume     -> volume_adjust(up)
* decrease volume     -> volume_adjust(down)

Every other source record is retained only as an explicit unsupported/OOD
example with an abstention target. The derived target is not a native FSC gold
label and must be reported as such.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


GENERATOR_VERSION = "fsc-command-benchmark/0.2.0"
SOURCE_SPLITS = ("train", "valid", "test")
OUTPUT_SPLITS = {"train": "train", "valid": "dev", "test": "test"}

DIRECT_RULES: dict[tuple[str, str, str], tuple[str, dict[str, Any], str]] = {
    ("activate", "music", "none"): (
        "media_control",
        {"action": "play"},
        "activate_music_to_media_play",
    ),
    ("deactivate", "music", "none"): (
        "media_control",
        {"action": "pause"},
        "deactivate_music_to_media_pause",
    ),
    ("increase", "volume", "none"): (
        "volume_adjust",
        {"direction": "up"},
        "increase_volume_to_volume_up",
    ),
    ("decrease", "volume", "none"): (
        "volume_adjust",
        {"direction": "down"},
        "decrease_volume_to_volume_down",
    ),
}

ABSTAIN_RULE = "unsupported_fsc_semantics_to_policy_abstain"


def normalize_template(text: str) -> str:
    """Normalize surface form for leakage auditing without changing stored text."""

    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^\w\s']", "", normalized, flags=re.UNICODE)
    return normalized


def stable_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def read_source_rows(fsc_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_split in SOURCE_SPLITS:
        csv_path = fsc_root / "data" / f"{source_split}_data.csv"
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                path = row["path"]
                source_id = f"{source_split}:{path}"
                rows.append(
                    {
                        "source_id": source_id,
                        "source_split": source_split,
                        "source_row": row_number,
                        "audio": path,
                        "speaker_id": row["speakerId"],
                        "text": row["transcription"],
                        "native_action": row["action"],
                        "native_object": row["object"],
                        "native_location": row["location"],
                        "template_id": stable_digest(normalize_template(row["transcription"])),
                    }
                )
    return rows


def derive_target(row: dict[str, Any]) -> tuple[dict[str, Any], str, str | None]:
    key = (row["native_action"], row["native_object"], row["native_location"])
    mapped = DIRECT_RULES.get(key)
    if mapped is None:
        return (
            {"action": "abstain", "tool": None, "arguments": {}},
            ABSTAIN_RULE,
            None,
        )
    tool, arguments, rule = mapped
    return (
        {"action": "call", "tool": tool, "arguments": arguments},
        rule,
        tool,
    )


def materialize_record(
    row: dict[str, Any],
    split: str,
    split_strategy: str,
    sample_group: str,
) -> dict[str, Any]:
    target, mapping_rule, tool = derive_target(row)
    mapping_status = "supported" if target["action"] == "call" else "unsupported"
    return {
        "id": f"fsc_{stable_digest(row['source_id'])}",
        "split": split,
        "locale": "en-US",
        "text": row["text"],
        "target": target,
        "metadata": {
            "kind": target["action"],
            "case_id": f"fsc_case_{stable_digest(row['source_id'])}",
            "variant_id": 0,
            "generator_version": GENERATOR_VERSION,
            "tool": tool,
            "source": "fluent_speech_commands",
            "source_id": row["source_id"],
            "source_split": row["source_split"],
            "source_row": row["source_row"],
            "speaker_id": row["speaker_id"],
            "audio": row["audio"],
            "native_action": row["native_action"],
            "native_object": row["native_object"],
            "native_location": row["native_location"],
            "template_id": row["template_id"],
            "mapping_status": mapping_status,
            "mapping_rule": mapping_rule,
            "split_strategy": split_strategy,
            "sample_group": sample_group,
        },
    }


def sort_key(row: dict[str, Any], seed: int, salt: str) -> str:
    return hashlib.sha256(
        f"{seed}|{salt}|{row['source_id']}".encode("utf-8")
    ).hexdigest()


def balance_rows(
    rows: Iterable[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[OUTPUT_SPLITS[row["source_split"]]].append(row)
    selected: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        split_rows = grouped[split]
        supported = [row for row in split_rows if derive_target(row)[0]["action"] == "call"]
        unsupported = [row for row in split_rows if derive_target(row)[0]["action"] == "abstain"]
        supported.sort(key=lambda row: row["source_id"])
        unsupported.sort(key=lambda row: sort_key(row, seed, split))
        negative_count = min(len(supported), len(unsupported))
        selected.extend(
            materialize_record(row, split, "official_speaker", "balanced_1to1")
            for row in supported
        )
        selected.extend(
            materialize_record(row, split, "official_speaker", "balanced_1to1")
            for row in unsupported[:negative_count]
        )
    return sorted(selected, key=lambda record: record["id"])


def assign_phrase_splits(rows: list[dict[str, Any]], seed: int) -> dict[str, str]:
    groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (
            row["native_action"],
            row["native_object"],
            row["native_location"],
        )
        groups[key].add(row["template_id"])
    assignments: dict[str, str] = {}
    for key, templates in sorted(groups.items()):
        ordered = sorted(
            templates,
            key=lambda template: hashlib.sha256(
                f"{seed}|{key}|{template}".encode("utf-8")
            ).hexdigest(),
        )
        n = len(ordered)
        train_end = max(1, round(n * 0.70))
        dev_end = min(n - 1, train_end + max(1, round(n * 0.15)))
        for index, template in enumerate(ordered):
            if index < train_end:
                split = "train"
            elif index < dev_end:
                split = "dev"
            else:
                split = "test"
            assignments[
                f"{key[0]}|{key[1]}|{key[2]}|{template}"
            ] = split
    return assignments


def phrase_disjoint_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    assignments = assign_phrase_splits(rows, seed)
    annotated: list[tuple[dict[str, Any], str]] = []
    for row in rows:
        key = (
            f"{row['native_action']}|{row['native_object']}|"
            f"{row['native_location']}|{row['template_id']}"
        )
        annotated.append((row, assignments[key]))

    selected: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        split_rows = [row for row, assigned in annotated if assigned == split]
        supported = [row for row in split_rows if derive_target(row)[0]["action"] == "call"]
        unsupported = [row for row in split_rows if derive_target(row)[0]["action"] == "abstain"]
        supported.sort(key=lambda row: row["source_id"])
        unsupported.sort(key=lambda row: sort_key(row, seed, f"phrase-{split}"))
        negative_count = min(len(supported), len(unsupported))
        selected.extend(
            materialize_record(row, split, "phrase_disjoint", "balanced_1to1")
            for row in supported
        )
        selected.extend(
            materialize_record(row, split, "phrase_disjoint", "balanced_1to1")
            for row in unsupported[:negative_count]
        )
    return sorted(selected, key=lambda record: record["id"])


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    speakers: dict[str, set[str]] = defaultdict(set)
    templates: dict[str, set[str]] = defaultdict(set)
    rules: Counter[str] = Counter()
    for record in records:
        split = record["split"]
        metadata = record["metadata"]
        by_split[split][metadata["mapping_status"]] += 1
        speakers[split].add(metadata["speaker_id"])
        templates[split].add(metadata["template_id"])
        rules[metadata["mapping_rule"]] += 1
    return {
        "records": len(records),
        "by_split": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(by_split.items())
        },
        "speakers_by_split": {split: len(ids) for split, ids in sorted(speakers.items())},
        "templates_by_split": {
            split: len(ids) for split, ids in sorted(templates.items())
        },
        "mapping_rules": dict(sorted(rules.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fsc-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="official speaker-disjoint balanced derived dataset",
    )
    parser.add_argument(
        "--phrase-output",
        type=Path,
        required=True,
        help="phrase-disjoint balanced derived dataset",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    rows = read_source_rows(args.fsc_root)
    official = balance_rows(rows, args.seed)
    phrase = phrase_disjoint_rows(rows, args.seed)
    official_digest = write_jsonl(args.output, official)
    phrase_digest = write_jsonl(args.phrase_output, phrase)

    source_files = {}
    for source_split in SOURCE_SPLITS:
        path = args.fsc_root / "data" / f"{source_split}_data.csv"
        source_files[source_split] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    official_speakers = {
        split: {record["metadata"]["speaker_id"] for record in official if record["split"] == split}
        for split in ("train", "dev", "test")
    }
    speaker_intersections = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        speaker_intersections[f"{left}-{right}"] = len(
            official_speakers[left] & official_speakers[right]
        )
    template_sets = {
        split: {record["metadata"]["template_id"] for record in official if record["split"] == split}
        for split in ("train", "dev", "test")
    }
    template_intersections = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        template_intersections[f"{left}-{right}"] = len(
            template_sets[left] & template_sets[right]
        )

    manifest = {
        "dataset_name": "Fluent Speech Commands derived command-routing benchmark",
        "generator_version": GENERATOR_VERSION,
        "locale": "en-US",
        "source_dataset": "Fluent Speech Commands",
        "source_root": str(args.fsc_root),
        "source_archive_sha256": "c9fd67f2efa078daa84daddcad2de937eb96581c140e3131ed8cd06fbae9ba1b",
        "source_files": source_files,
        "source_license": "CC BY-NC-ND 4.0; academic research only",
        "derived_contract": "data/tools/fsc_command_benchmark.json",
        "mapping_policy": {
            "supported_native_labels": [
                ["activate", "music", "none"],
                ["deactivate", "music", "none"],
                ["increase", "volume", "none"],
                ["decrease", "volume", "none"]
            ],
            "unsupported_policy": "all other FSC labels become abstain; this is a derived policy label, not native FSC gold"
        },
        "seed": args.seed,
        "official_speaker_disjoint": {
            "path": str(args.output),
            "sha256": official_digest,
            "summary": summarize(official),
            "speaker_intersections": speaker_intersections,
            "template_intersections": template_intersections
        },
        "phrase_disjoint": {
            "path": str(args.phrase_output),
            "sha256": phrase_digest,
            "summary": summarize(phrase),
            "note": "Phrase groups are disjoint across splits; speakers may occur in multiple splits by design."
        },
        "limitations": [
            "FSC is smart-home/assistant speech, not a native command-routing corpus.",
            "The four supported mappings are manually specified semantic bridges.",
            "The source repeats exact phrasings across speakers; official splits are speaker-disjoint but not phrase-disjoint.",
            "The benchmark consumes transcripts; it does not perform speech recognition or execute actions."
        ]
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
