#!/usr/bin/env python3
"""Convert the Phase 1 Qwen3.5 PEFT adapter to a llama.cpp GGUF adapter.

The Qwen3.5 checkpoint currently exposes the conditional-generation
architecture at the top level while leaving ``text_config.architectures`` as
``null``.  The llama.cpp converter supports the architecture, but its LoRA
entry point indexes that nested field directly.  This wrapper creates a
temporary config-only view with the nested architecture filled in, then calls
the upstream converter unchanged.

The temporary view contains no model weights.  This is sufficient for LoRA
export and avoids modifying the cached Hugging Face checkpoint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Hugging Face base checkpoint directory")
    parser.add_argument("--lora", type=Path, required=True, help="PEFT adapter directory")
    parser.add_argument("--outfile", type=Path, required=True, help="Output GGUF adapter path")
    parser.add_argument("--llama-cpp", type=Path, default=Path("/home/daniel/llama.cpp"))
    parser.add_argument("--outtype", choices=("f32", "f16", "bf16", "q8_0", "auto"), default="f16")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_config_view(base: Path, temp_root: Path) -> Path:
    source = base / "config.json"
    with source.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    top_architectures = config.get("architectures") or []
    text_config = config.setdefault("text_config", {})
    nested_architectures = text_config.get("architectures") or top_architectures
    if not nested_architectures:
        raise ValueError(f"No architecture found in {source}")
    text_config["architectures"] = nested_architectures

    config_view = temp_root / "config-view"
    config_view.mkdir()
    with (config_view / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return config_view


def main() -> int:
    args = parse_args()
    converter = args.llama_cpp / "convert_lora_to_gguf.py"
    if not converter.is_file():
        raise FileNotFoundError(converter)
    if not (args.base / "config.json").is_file():
        raise FileNotFoundError(args.base / "config.json")
    if not args.lora.is_dir():
        raise FileNotFoundError(args.lora)

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qwen35-lora-config-") as temporary:
        config_view = make_config_view(args.base, Path(temporary))
        command = [
            sys.executable,
            str(converter),
            "--base",
            str(config_view),
            "--outfile",
            str(args.outfile),
            "--outtype",
            args.outtype,
        ]
        if args.dry_run:
            command.append("--dry-run")
        command.append(str(args.lora))
        return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
