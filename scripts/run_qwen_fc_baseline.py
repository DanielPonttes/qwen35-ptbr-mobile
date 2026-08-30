#!/usr/bin/env python3
"""Executa baseline zero-shot do Qwen em um dataset FC JSONL.

O script registra a resposta bruta para que o parser e as métricas possam ser
auditados separadamente da geração.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from fc_common import load_json, load_registry
from peft import PeftModel


SYSTEM_PROMPT = """Você é um roteador estrito de comandos Android em português brasileiro.
Escolha no máximo uma ferramenta do catálogo fornecido.
Responda somente com a chamada de ferramenta exigida pelo template ou com:
{\"action\":\"abstain\",\"tool\":null,\"arguments\":{}}.
Abstenha-se quando o pedido for ambíguo, estiver incompleto, fora do catálogo ou não puder ser executado com segurança.
Não explique sua decisão e não invente argumentos."""

CANONICAL_SYSTEM_PROMPT = """Você é um roteador estrito de comandos Android em português brasileiro.
Escolha no máximo uma ferramenta do catálogo.
Responda somente com JSON válido e exatamente estes campos: action, tool, arguments.
Para uma ação válida use action=call, o nome da ferramenta e os argumentos exigidos.
Para pedido ambíguo, incompleto, fora do catálogo ou inseguro use action=abstain, tool=null e arguments={}.
Não explique a decisão, não use markdown e não invente argumentos.
Catálogo de ferramentas:
{catalog}
"""


def as_openai_tools(registry: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["arguments"],
            },
        }
        for tool in registry.values()
    ]


def compact_catalog(registry: dict[str, dict[str, Any]]) -> str:
    return json.dumps(
        [
            {"name": tool["name"], "arguments": tool["arguments"]}
            for tool in registry.values()
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def load_records(path: Path, split: str | None, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if split is not None and record.get("split") != split:
                continue
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def move_encoding_to_device(encoded: Any, device: torch.device) -> dict[str, torch.Tensor]:
    if isinstance(encoded, torch.Tensor):
        encoded = {
            "input_ids": encoded,
            "attention_mask": torch.ones_like(encoded),
        }
    return {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }


def tokenize_request(
    tokenizer: Any,
    text: str,
    tools: list[dict[str, Any]],
    registry: dict[str, dict[str, Any]],
    prompt_mode: str,
) -> tuple[dict[str, torch.Tensor], str]:
    if prompt_mode == "canonical":
        canonical_system = CANONICAL_SYSTEM_PROMPT.replace(
            "{catalog}", compact_catalog(registry)
        )
        messages = [
            {"role": "system", "content": canonical_system},
            {"role": "user", "content": text},
        ]
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            return move_encoding_to_device(encoded, torch.device("cpu")), "canonical_no_thinking"
        except Exception:
            fallback = canonical_system + "\nComando do usuário: " + text + "\nResposta:"
            return move_encoding_to_device(tokenizer(fallback, return_tensors="pt"), torch.device("cpu")), "canonical_plain"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        return move_encoding_to_device(encoded, torch.device("cpu")), "tools_no_thinking"
    except (TypeError, ValueError, KeyError):
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
            return move_encoding_to_device(encoded, torch.device("cpu")), "tools_default"
        except Exception:
            fallback = (
                SYSTEM_PROMPT
                + "\nCatálogo JSON:\n"
                + json.dumps(tools, ensure_ascii=False)
                + "\nComando do usuário: "
                + text
                + "\nResposta:"
            )
            return move_encoding_to_device(tokenizer(fallback, return_tensors="pt"), torch.device("cpu")), "plain_prompt"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="diretório ou identificador HF local")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "data" / "generated" / "fc_dataset.jsonl",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=root / "data" / "tools" / "android_tools.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--prompt-mode", choices=["tools", "canonical"], default="tools")
    args = parser.parse_args()

    registry = load_registry(args.registry)
    records = load_records(args.dataset, args.split, args.limit)
    if not records:
        raise SystemExit("nenhum registro selecionado")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"records={len(records)}")
    print(f"model={args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device)
    model.eval()
    tools = as_openai_tools(registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            encoded_cpu, template_mode = tokenize_request(
                tokenizer, record["text"], tools, registry, args.prompt_mode
            )
            encoded = {key: value.to(device) for key, value in encoded_cpu.items()}
            prompt_tokens = int(encoded["input_ids"].shape[-1])
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            completion = generated[0, prompt_tokens:]
            raw = tokenizer.decode(completion, skip_special_tokens=False)
            output_record = {
                "id": record["id"],
                "raw": raw,
                "model": str(args.model),
                "adapter": args.adapter,
                "prompt_mode": args.prompt_mode,
                "device": str(device),
                "template_mode": template_mode,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": int(completion.shape[-1]),
                "latency_ms": round(elapsed_ms, 3),
            }
            handle.write(json.dumps(output_record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(f"{index}/{len(records)} id={record['id']} latency_ms={elapsed_ms:.1f} template={template_mode}")
    print(f"predictions={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
