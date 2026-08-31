#!/usr/bin/env python3
"""Generate deterministic controls for the FSC command-routing benchmark.

The controls use only the transcript and the public benchmark contract. They
never inspect native FSC labels, metadata, or the target field.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


ABSTAIN = {"action": "abstain", "tool": None, "arguments": {}}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def lexical_target(text: str) -> dict[str, Any]:
    """A deliberately transparent keyword baseline for the two-tool contract."""

    normalized = normalize(text)
    has_media = bool(re.search(r"\b(music|song|songs|audio|track|tune|playlist)\b", normalized))
    has_volume = bool(re.search(r"\b(volume|sound|loud|quiet)\b", normalized))
    if has_media and re.search(
        r"\b(play|playing|start|activate|turn on|put on|resume|unpause)\b",
        normalized,
    ):
        return {"action": "call", "tool": "media_control", "arguments": {"action": "play"}}
    if has_media and re.search(
        r"\b(pause|stop|deactivate|turn off|halt|quiet)\b",
        normalized,
    ):
        return {"action": "call", "tool": "media_control", "arguments": {"action": "pause"}}
    if has_volume and re.search(r"\b(increase|raise|up|louder|loud)\b", normalized):
        return {"action": "call", "tool": "volume_adjust", "arguments": {"direction": "up"}}
    if has_volume and re.search(r"\b(decrease|lower|down|quieter|quiet)\b", normalized):
        return {"action": "call", "tool": "volume_adjust", "arguments": {"direction": "down"}}
    return dict(ABSTAIN)


def read_records(path: Path, split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record.get("split") == split:
                    records.append(record)
    return records


def write_predictions(records: list[dict[str, Any]], baseline: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            target = dict(ABSTAIN) if baseline == "always_abstain" else lexical_target(record["text"])
            handle.write(
                json.dumps(
                    {
                        "id": record["id"],
                        "raw": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
                        "model": "deterministic-control",
                        "baseline": baseline,
                        "locale": record.get("locale"),
                        "device": "cpu",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        choices=["always_abstain", "lexical"],
        required=True,
    )
    args = parser.parse_args()
    records = read_records(args.dataset, args.split)
    if not records:
        raise SystemExit("nenhum registro selecionado")
    write_predictions(records, args.baseline, args.output)
    print(f"baseline={args.baseline} split={args.split} records={len(records)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
