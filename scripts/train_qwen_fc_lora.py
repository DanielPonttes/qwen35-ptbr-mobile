#!/usr/bin/env python3
"""Treina um adapter LoRA para a saída canônica de FC no 5090.

O treino é deliberadamente simples e auditável: não depende de Trainer,
datasets ou jsonschema e mascara a perda no prompt, calculando-a apenas na
resposta canônica.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoTokenizer

from fc_common import load_registry
from peft import LoraConfig, get_peft_model


SYSTEM_PROMPT_PTBR = """Você é um roteador estrito de comandos Android em português brasileiro.
Escolha no máximo uma ferramenta do catálogo.
Responda somente com JSON válido e exatamente estes campos: action, tool, arguments.
Para uma ação válida use action=call, o nome da ferramenta e os argumentos exigidos.
Para pedido ambíguo, incompleto, fora do catálogo ou inseguro use action=abstain, tool=null e arguments={}.
Não explique a decisão, não use markdown e não invente argumentos.
Catálogo de ferramentas:
{catalog}
"""

SYSTEM_PROMPT_EN = """You are a strict Android command router.
Choose at most one tool from the catalog.
Respond only with valid JSON containing exactly these fields: action, tool, arguments.
For a valid request use action=call, the tool name, and the required arguments.
For an ambiguous, incomplete, unsupported, out-of-catalog, or unsafe request use action=abstain, tool=null, and arguments={}.
Do not explain your decision, use markdown, or invent arguments.
Tool catalog:
{catalog}
"""


def build_system_prompt(locale: str, catalog: str) -> str:
    if locale == "en-US":
        return SYSTEM_PROMPT_EN.replace("{catalog}", catalog)
    if locale == "pt-BR":
        return SYSTEM_PROMPT_PTBR.replace("{catalog}", catalog)
    raise ValueError(f"locale sem prompt suportado: {locale}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact_catalog(registry: dict[str, dict[str, Any]]) -> str:
    return json.dumps(
        [
            {"name": tool["name"], "arguments": tool["arguments"]}
            for tool in registry.values()
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def prompt_ids(tokenizer: Any, text: str, system_prompt: str) -> list[int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            chat_template_kwargs={"enable_thinking": False},
        )
    except (TypeError, ValueError, KeyError):
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
        except Exception:
            encoded = tokenizer(
                system_prompt + "\nComando do usuário: " + text + "\nResposta:",
                add_special_tokens=True,
            )["input_ids"]
    if isinstance(encoded, torch.Tensor):
        return encoded.tolist()
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        input_ids = encoded["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            return input_ids.tolist()
        return list(input_ids)
    return list(encoded)


class FCDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, system_prompt: str, max_length: int):
        self.items: list[dict[str, Any]] = []
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("tokenizer sem eos_token_id")
        for record in records:
            prefix = prompt_ids(tokenizer, record["text"], system_prompt)
            target_text = json.dumps(
                record["target"], ensure_ascii=False, separators=(",", ":")
            )
            target = tokenizer(target_text, add_special_tokens=False)["input_ids"] + [eos]
            input_ids = prefix + target
            labels = [-100] * len(prefix) + target
            if len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]
                labels = labels[-max_length:]
            self.items.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                    "id": record["id"],
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    max_length = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.full((len(batch), max_length), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
    for row, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[row, :length] = item["input_ids"]
        labels[row, :length] = item["labels"]
        attention_mask[row, :length] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def run_eval(model: Any, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key in {"input_ids", "labels", "attention_mask"}
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = model(**inputs)
            losses.append(float(outputs.loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, default=root / "data" / "generated" / "fc_dataset.jsonl")
    parser.add_argument("--registry", type=Path, default=root / "data" / "tools" / "android_tools.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--locale", choices=["pt-BR", "en-US"], default="pt-BR")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("treinamento desta fase exige a RTX 5090; CUDA não está disponível")

    registry = load_registry(args.registry)
    records = read_jsonl(args.dataset)
    train_records = [record for record in records if record.get("split") == "train"]
    dev_records = [record for record in records if record.get("split") == "dev"]
    if not train_records or not dev_records:
        raise SystemExit("dataset precisa conter train e dev")
    pad_token_id = None
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if pad_token_id is None:
        raise SystemExit("tokenizer sem pad/eos token")
    system_prompt = build_system_prompt(args.locale, compact_catalog(registry))
    train_dataset = FCDataset(train_records, tokenizer, system_prompt, args.max_length)
    dev_dataset = FCDataset(dev_records, tokenizer, system_prompt, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=lambda batch: collate(batch, pad_token_id),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate(batch, pad_token_id),
    )

    print(f"device={device}")
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"train={len(train_dataset)} dev={len(dev_dataset)}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    global_step = 0
    start_time = time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
                loss = outputs.loss / args.gradient_accumulation
            loss.backward()
            running += float(loss.detach().cpu()) * args.gradient_accumulation
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0 or step == len(train_loader):
                    print(f"epoch={epoch + 1} step={global_step} train_loss={running / step:.4f}")
        train_loss = running / len(train_loader)
        dev_loss = run_eval(model, dev_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "optimizer_steps": global_step,
                "train_loss": round(train_loss, 6),
                "dev_loss": round(dev_loss, 6),
            }
        )
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} dev_loss={dev_loss:.4f}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    dataset_digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    manifest = {
        "model": str(args.model),
        "locale": args.locale,
        "dataset_sha256": dataset_digest,
        "dataset": str(args.dataset),
        "registry": str(args.registry),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "elapsed_seconds": round(time.perf_counter() - start_time, 3),
        "history": history,
    }
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"adapter={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
