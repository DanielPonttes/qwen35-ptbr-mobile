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
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

from fc_common import load_json, load_registry
from peft import PeftModel


SYSTEM_PROMPT_PTBR = """Você é um roteador estrito de comandos em português brasileiro.
Escolha no máximo uma ferramenta do catálogo fornecido.
Responda somente com a chamada de ferramenta exigida pelo template ou com:
{\"action\":\"abstain\",\"tool\":null,\"arguments\":{}}.
Abstenha-se quando o pedido for ambíguo, estiver incompleto, fora do catálogo ou não puder ser executado com segurança.
Não explique sua decisão e não invente argumentos."""

CANONICAL_SYSTEM_PROMPT_PTBR = """Você é um roteador estrito de comandos em um benchmark de modelos pequenos.
Escolha no máximo uma ferramenta do catálogo.
Responda somente com JSON válido e exatamente estes campos: action, tool, arguments.
Para uma ação válida use action=call, o nome da ferramenta e os argumentos exigidos.
Para pedido ambíguo, incompleto, fora do catálogo ou inseguro use action=abstain, tool=null e arguments={}.
Não explique a decisão, não use markdown e não invente argumentos.
Catálogo de ferramentas:
{catalog}
"""

SYSTEM_PROMPT_EN = """You are a strict command router for a small-model benchmark.
Choose at most one tool from the provided catalog.
Respond only with the tool call required by the template or:
{\"action\":\"abstain\",\"tool\":null,\"arguments\":{}}.
Abstain when the request is ambiguous, incomplete, unsupported, out of catalog, or unsafe.
Do not explain your decision or invent arguments."""

CANONICAL_SYSTEM_PROMPT_EN = """You are a strict command router for a small-model benchmark.
Choose at most one tool from the catalog.
Respond only with valid JSON containing exactly these fields: action, tool, arguments.
For a valid request use action=call, the tool name, and the required arguments.
For an ambiguous, incomplete, unsupported, out-of-catalog, or unsafe request use action=abstain, tool=null, and arguments={}.
Do not explain your decision, use markdown, or invent arguments.
Operation catalog:
{catalog}
"""


def load_text_model(model_id: str, dtype: torch.dtype) -> tuple[Any, str]:
    """Load transcript-capable checkpoints without requiring one architecture."""

    config = AutoConfig.from_pretrained(model_id)
    architectures = getattr(config, "architectures", None) or []
    if getattr(config, "model_type", "") == "qwen3_5" or any(
        "ConditionalGeneration" in architecture for architecture in architectures
    ):
        model_class = AutoModelForImageTextToText
    else:
        model_class = AutoModelForCausalLM
    model = model_class.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )
    return model, model_class.__name__


def build_system_prompt(locale: str, canonical: bool, catalog: str) -> str:
    if locale == "en-US":
        template = CANONICAL_SYSTEM_PROMPT_EN if canonical else SYSTEM_PROMPT_EN
    elif locale == "pt-BR":
        template = CANONICAL_SYSTEM_PROMPT_PTBR if canonical else SYSTEM_PROMPT_PTBR
    else:
        raise ValueError(f"locale sem prompt suportado: {locale}")
    return template.replace("{catalog}", catalog) if canonical else template


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
    locale: str,
) -> tuple[dict[str, torch.Tensor], str]:
    if prompt_mode == "canonical":
        canonical_system = build_system_prompt(
            locale, True, compact_catalog(registry)
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
            user_label = "User command" if locale == "en-US" else "Comando do usuário"
            fallback = canonical_system + f"\n{user_label}: " + text + "\nResposta:"
            return move_encoding_to_device(tokenizer(fallback, return_tensors="pt"), torch.device("cpu")), "canonical_plain"
    system_prompt = build_system_prompt(locale, False, compact_catalog(registry))
    messages = [
        {"role": "system", "content": system_prompt},
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
                system_prompt
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
        default=root / "data" / "tools" / "fsc_command_benchmark.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--prompt-mode", choices=["tools", "canonical"], default="tools")
    parser.add_argument("--locale", choices=["pt-BR", "en-US"], default="pt-BR")
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
    model, model_class = load_text_model(args.model, dtype)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device)
    model.eval()
    tools = as_openai_tools(registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    partial_output = args.output.with_name(args.output.name + ".partial")
    with partial_output.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            encoded_cpu, template_mode = tokenize_request(
                tokenizer, record["text"], tools, registry, args.prompt_mode, args.locale
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
                "model_class": model_class,
                "adapter": args.adapter,
                "locale": args.locale,
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
    partial_output.replace(args.output)
    print(f"predictions={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
