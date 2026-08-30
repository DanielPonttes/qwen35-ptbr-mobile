#!/usr/bin/env python3
"""Run one canonical Phase 1 example through llama.cpp + a GGUF LoRA adapter."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from fc_common import load_registry, parse_json_object, validate_target
from run_qwen_fc_baseline import CANONICAL_SYSTEM_PROMPT, compact_catalog


DEFAULT_CUDA_LIBS = (
    "/home/daniel/Área de trabalho/swarm-emotions-tag/python-ml/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib",
    "/home/daniel/Área de trabalho/swarm-emotions-tag/python-ml/.venv/lib/python3.12/site-packages/nvidia/cublas/lib",
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-hf", type=Path, required=True)
    parser.add_argument("--base-gguf", type=Path, required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=root / "data/generated/fc_dataset.jsonl")
    parser.add_argument("--registry", type=Path, default=root / "data/tools/android_tools.json")
    parser.add_argument("--schema", type=Path, default=root / "data/schema/fc_output.schema.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-id", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--llama-completion", type=Path, default=Path("/home/daniel/llama.cpp/build/bin/llama-completion"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--cuda-lib", action="append", default=[])
    return parser.parse_args()


def load_record(path: Path, split: str, index: int, record_id: str | None) -> dict[str, Any]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record_id is not None:
                if record.get("id") == record_id:
                    return record
            elif record.get("split") == split:
                records.append(record)
    if record_id is not None:
        raise ValueError(f"registro inexistente: {record_id}")
    if not 0 <= index < len(records):
        raise IndexError(f"índice {index} fora do split {split} ({len(records)} registros)")
    return records[index]


def build_prompt(record: dict[str, Any], registry_path: Path, base_hf: Path) -> str:
    from transformers import AutoTokenizer

    registry = load_registry(registry_path)
    system = CANONICAL_SYSTEM_PROMPT.replace("{catalog}", compact_catalog(registry))
    tokenizer = AutoTokenizer.from_pretrained(base_hf)
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": record["text"]},
        ],
        add_generation_prompt=True,
        tokenize=False,
        chat_template_kwargs={"enable_thinking": False},
    )


def main() -> int:
    args = parse_args()
    record = load_record(args.dataset, args.split, args.index, args.record_id)
    registry = load_registry(args.registry)
    prompt = build_prompt(record, args.registry, args.base_hf)

    library_paths = args.cuda_lib or list(DEFAULT_CUDA_LIBS)
    env = os.environ.copy()
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join([*library_paths, existing] if existing else library_paths)

    command = [
        str(args.llama_completion),
        "-m",
        str(args.base_gguf),
        "--lora",
        str(args.lora),
        "-f",
        "PROMPT_FILE",
        "-no-cnv",
        "-n",
        str(args.max_new_tokens),
        "--ctx-size",
        str(args.ctx_size),
        "-ngl",
        str(args.ngl),
        "--temp",
        "0",
        "--seed",
        "1",
        "--json-schema-file",
        str(args.schema),
        "--no-display-prompt",
        "--log-verbosity",
        "0",
        "--color",
        "off",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="llama-fc-smoke-") as temporary:
        prompt_path = Path(temporary) / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        actual_command = [str(prompt_path) if value == "PROMPT_FILE" else value for value in command]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                actual_command,
                env=env,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            timed_out = False
        except subprocess.TimeoutExpired as error:
            completed = subprocess.CompletedProcess(
                actual_command,
                returncode=124,
                stdout=error.stdout or "",
                stderr=error.stderr or "",
            )
            timed_out = True
        elapsed_ms = (time.perf_counter() - started) * 1000.0

    stdout = completed.stdout or ""
    prediction, is_json, parse_error = parse_json_object(stdout)
    validation_errors = validate_target(prediction, registry) if prediction is not None else [parse_error or "predição ausente"]
    report = {
        "record": record,
        "prediction": prediction,
        "json_parseable": is_json,
        "validation_errors": validation_errors,
        "raw_stdout": stdout,
        "stderr": completed.stderr or "",
        "returncode": completed.returncode,
        "timed_out": timed_out,
        "elapsed_ms": round(elapsed_ms, 3),
        "command": [str(args.llama_completion), "-m", str(args.base_gguf), "--lora", str(args.lora)],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if completed.returncode == 0 and not validation_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
