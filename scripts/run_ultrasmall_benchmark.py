#!/usr/bin/env python3
"""Run the reproducible multi-model FSC ultra-small benchmark.

The runner keeps model outputs and adapters server-local while versioning only
aggregate metrics and this command manifest. It is intentionally transcript
only: no ADB, phone, audio capture, or Android execution is involved.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROTOCOLS = {
    "official": "fluent_speech_commands_command_benchmark.jsonl",
    "phrase_disjoint": "fluent_speech_commands_command_benchmark_phrase_disjoint.jsonl",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_command(command: list[str], root: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=root,
            env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"comando falhou ({completed.returncode}); veja {log_path}")


def maybe_run(
    command: list[str],
    root: Path,
    log_path: Path,
    outputs: list[Path],
    skip_existing: bool,
) -> None:
    if skip_existing and all(path.exists() for path in outputs):
        print("skip-existing: " + ", ".join(str(path) for path in outputs), flush=True)
        return
    run_command(command, root, log_path)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=Path,
        default=root / "benchmarks" / "ultrasmall_models.json",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=None,
        help="run only this manifest name; repeat for several models",
    )
    parser.add_argument(
        "--protocol",
        action="append",
        choices=sorted(PROTOCOLS),
        default=None,
        help="protocol to run; repeat or omit for both",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=root / "data" / "tools" / "fsc_command_benchmark.json",
    )
    parser.add_argument("--output-root", type=Path, default=root / "results")
    parser.add_argument("--log-root", type=Path, default=root / "logs" / "ultrasmall")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    manifest = load_json(args.models)
    cap = int(manifest["parameter_cap"])
    selected_names = set(args.model_name or [model["name"] for model in manifest["models"]])
    selected_models = [model for model in manifest["models"] if model["name"] in selected_names]
    if len(selected_models) != len(selected_names):
        missing = sorted(selected_names - {model["name"] for model in selected_models})
        raise SystemExit(f"modelos não encontrados no manifesto: {missing}")
    for model in selected_models:
        if int(model["parameters"]) > cap:
            raise SystemExit(f"modelo acima do teto: {model['name']}")
    protocols = args.protocol or list(PROTOCOLS)
    python = sys.executable
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.log_root.mkdir(parents=True, exist_ok=True)

    run_manifest: dict[str, Any] = {
        "benchmark": manifest["benchmark"],
        "models_manifest": str(args.models),
        "parameter_cap": cap,
        "registry": str(args.registry),
        "protocols": protocols,
        "seed": args.seed,
        "settings": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "prompt_mode": "canonical",
            "locale": "en-US",
            "device": "cuda",
            "training_requested": not args.skip_training,
            "baseline_requested": not args.skip_baseline,
        },
        "jobs": [],
    }
    try:
        import torch

        run_manifest["environment"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover - diagnostics only
        run_manifest["environment_error"] = str(exc)

    for protocol in protocols:
        dataset = root / "data" / "external" / PROTOCOLS[protocol]
        group_field = "template_id"
        for model in selected_models:
            name = model["name"]
            tag = f"fsc_ultrasmall_{name}"
            base_predictions = args.output_root / f"{tag}_base_{protocol}.predictions.jsonl"
            base_metrics = args.output_root / f"{tag}_base_{protocol}.metrics.json"
            adapter = args.output_root / f"{tag}_lora_{protocol}_seed{args.seed}"
            lora_predictions = args.output_root / f"{tag}_lora_{protocol}.predictions.jsonl"
            lora_metrics = args.output_root / f"{tag}_lora_{protocol}.metrics.json"
            job: dict[str, Any] = {
                "model": model,
                "protocol": protocol,
                "dataset": str(dataset),
                "base_predictions": str(base_predictions),
                "base_metrics": str(base_metrics),
                "training": {
                    "enabled": not args.skip_training,
                    "adapter": str(adapter),
                    "predictions": str(lora_predictions),
                    "metrics": str(lora_metrics),
                },
                "status": "started",
                "started_at": time.time(),
            }
            run_manifest["jobs"].append(job)
            log_path = args.log_root / f"{name}_{protocol}.log"
            try:
                if not args.skip_baseline:
                    maybe_run(
                        [
                            python,
                            "scripts/run_qwen_fc_baseline.py",
                            "--model",
                            model["hf_id"],
                            "--dataset",
                            str(dataset),
                            "--registry",
                            str(args.registry),
                            "--output",
                            str(base_predictions),
                            "--split",
                            "test",
                            "--max-new-tokens",
                            str(args.max_new_tokens),
                            "--device",
                            "cuda",
                            "--prompt-mode",
                            "canonical",
                            "--locale",
                            "en-US",
                        ],
                        root,
                        log_path,
                        [base_predictions],
                        args.skip_existing,
                    )
                    maybe_run(
                        [
                            python,
                            "scripts/fc_eval.py",
                            "--dataset",
                            str(dataset),
                            "--predictions",
                            str(base_predictions),
                            "--registry",
                            str(args.registry),
                            "--split",
                            "test",
                            "--group-field",
                            group_field,
                            "--output",
                            str(base_metrics),
                        ],
                        root,
                        log_path,
                        [base_metrics],
                        args.skip_existing,
                    )
                if not args.skip_training:
                    maybe_run(
                        [
                            python,
                            "scripts/train_qwen_fc_lora.py",
                            "--model",
                            model["hf_id"],
                            "--dataset",
                            str(dataset),
                            "--registry",
                            str(args.registry),
                            "--output-dir",
                            str(adapter),
                            "--epochs",
                            str(args.epochs),
                            "--batch-size",
                            str(args.batch_size),
                            "--gradient-accumulation",
                            str(args.gradient_accumulation),
                            "--learning-rate",
                            str(args.learning_rate),
                            "--max-length",
                            str(args.max_length),
                            "--seed",
                            str(args.seed),
                            "--locale",
                            "en-US",
                        ],
                        root,
                        log_path,
                        [adapter / "adapter_model.safetensors", adapter / "training_manifest.json"],
                        args.skip_existing,
                    )
                    if not args.skip_baseline:
                        maybe_run(
                            [
                                python,
                                "scripts/run_qwen_fc_baseline.py",
                                "--model",
                                model["hf_id"],
                                "--dataset",
                                str(dataset),
                                "--registry",
                                str(args.registry),
                                "--output",
                                str(lora_predictions),
                                "--split",
                                "test",
                                "--max-new-tokens",
                                str(args.max_new_tokens),
                                "--device",
                                "cuda",
                                "--adapter",
                                str(adapter),
                                "--prompt-mode",
                                "canonical",
                                "--locale",
                                "en-US",
                            ],
                            root,
                            log_path,
                            [lora_predictions],
                            args.skip_existing,
                        )
                        maybe_run(
                            [
                                python,
                                "scripts/fc_eval.py",
                                "--dataset",
                                str(dataset),
                                "--predictions",
                                str(lora_predictions),
                                "--registry",
                                str(args.registry),
                                "--split",
                                "test",
                                "--group-field",
                                group_field,
                                "--output",
                                str(lora_metrics),
                            ],
                            root,
                            log_path,
                            [lora_metrics],
                            args.skip_existing,
                        )
                job["status"] = "completed"
            except Exception as exc:
                job["status"] = "failed"
                job["error"] = str(exc)
                print(f"JOB FAILED: {name}/{protocol}: {exc}", file=sys.stderr, flush=True)
            job["finished_at"] = time.time()
            args.output_root.joinpath("fsc_ultrasmall_benchmark_manifest.json").write_text(
                json.dumps(run_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    failures = [job for job in run_manifest["jobs"] if job["status"] != "completed"]
    print(f"jobs={len(run_manifest['jobs'])} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
