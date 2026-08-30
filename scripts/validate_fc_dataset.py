#!/usr/bin/env python3
"""Valida JSONL do dataset contra o catálogo Android e o contrato canônico."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from fc_common import load_registry, validate_target


REQUIRED_RECORD_KEYS = {"id", "split", "locale", "text", "target", "metadata"}
REQUIRED_METADATA_KEYS = {"kind", "case_id", "variant_id", "generator_version"}


def validate_record(record: Any, registry: dict[str, dict[str, Any]], line_number: int) -> list[str]:
    prefix = f"linha {line_number}"
    if not isinstance(record, dict):
        return [f"{prefix}: esperado objeto"]
    errors: list[str] = []
    missing = REQUIRED_RECORD_KEYS - set(record)
    extra = set(record) - REQUIRED_RECORD_KEYS
    errors.extend(f"{prefix}: campo ausente: {key}" for key in sorted(missing))
    errors.extend(f"{prefix}: campo não permitido: {key}" for key in sorted(extra))
    if missing:
        return errors
    if not isinstance(record["id"], str) or not record["id"]:
        errors.append(f"{prefix}.id: deve ser string não vazia")
    if record["split"] not in {"train", "dev", "test"}:
        errors.append(f"{prefix}.split: valor inválido")
    if record["locale"] != "pt-BR":
        errors.append(f"{prefix}.locale: esperado pt-BR")
    if not isinstance(record["text"], str) or len(record["text"].strip()) < 3:
        errors.append(f"{prefix}.text: texto vazio ou curto")
    errors.extend(f"{prefix}.{error}" for error in validate_target(record["target"], registry))

    metadata = record["metadata"]
    if not isinstance(metadata, dict):
        errors.append(f"{prefix}.metadata: esperado objeto")
        return errors
    missing_metadata = REQUIRED_METADATA_KEYS - set(metadata)
    extra_metadata = set(metadata) - (REQUIRED_METADATA_KEYS | {"tool"})
    errors.extend(f"{prefix}.metadata: campo ausente: {key}" for key in sorted(missing_metadata))
    errors.extend(f"{prefix}.metadata: campo não permitido: {key}" for key in sorted(extra_metadata))
    if metadata.get("kind") not in {"call", "abstain"}:
        errors.append(f"{prefix}.metadata.kind: valor inválido")
    if not isinstance(metadata.get("case_id"), str) or not metadata.get("case_id"):
        errors.append(f"{prefix}.metadata.case_id: inválido")
    if isinstance(metadata.get("variant_id"), bool) or not isinstance(metadata.get("variant_id"), int):
        errors.append(f"{prefix}.metadata.variant_id: esperado integer")
    if not isinstance(metadata.get("generator_version"), str):
        errors.append(f"{prefix}.metadata.generator_version: esperado string")
    target = record["target"]
    if isinstance(target, dict) and metadata.get("kind") != target.get("action"):
        errors.append(f"{prefix}: metadata.kind não coincide com target.action")
    if isinstance(target, dict) and metadata.get("tool") != target.get("tool"):
        errors.append(f"{prefix}: metadata.tool não coincide com target.tool")
    return errors


def validate_dataset(
    dataset_path: Path,
    registry_path: Path,
    expected_total: int | None = None,
) -> tuple[list[str], Counter[str], Counter[str]]:
    registry = load_registry(registry_path)
    errors: list[str] = []
    ids: set[str] = set()
    texts: set[str] = set()
    splits: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    case_splits: dict[str, set[str]] = {}
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                errors.append(f"linha {line_number}: linha vazia")
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"linha {line_number}: JSON inválido: {exc.msg}")
                continue
            errors.extend(validate_record(record, registry, line_number))
            if isinstance(record, dict):
                identifier = record.get("id")
                text = record.get("text")
                if identifier in ids:
                    errors.append(f"linha {line_number}: id duplicado: {identifier}")
                if text in texts:
                    errors.append(f"linha {line_number}: texto duplicado")
                ids.add(identifier)
                texts.add(text)
                splits[record.get("split")] += 1
                metadata = record.get("metadata", {})
                if isinstance(metadata, dict):
                    kinds[metadata.get("kind")] += 1
                    case_id = metadata.get("case_id")
                    split = record.get("split")
                    if isinstance(case_id, str) and isinstance(split, str):
                        case_splits.setdefault(case_id, set()).add(split)
    for case_id, case_split_values in sorted(case_splits.items()):
        if len(case_split_values) > 1:
            errors.append(
                f"case_id atravessa splits: {case_id} -> {sorted(case_split_values)}"
            )
    if expected_total is not None and len(ids) != expected_total:
        errors.append(f"total esperado {expected_total}, encontrado {len(ids)}")
    return errors, splits, kinds


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=root / "data" / "generated" / "fc_dataset.jsonl",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=root / "data" / "tools" / "android_tools.json",
    )
    parser.add_argument("--expected-total", type=int, default=None)
    args = parser.parse_args()
    errors, splits, kinds = validate_dataset(args.dataset, args.registry, args.expected_total)
    if errors:
        print(f"status=INVALID errors={len(errors)}")
        for error in errors[:50]:
            print(error)
        if len(errors) > 50:
            print(f"... {len(errors) - 50} erros adicionais")
        return 1
    print("status=VALID")
    print(f"records={sum(splits.values())}")
    print(f"splits={dict(sorted(splits.items()))}")
    print(f"kinds={dict(sorted(kinds.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
