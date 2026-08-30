#!/usr/bin/env python3
"""Funções compartilhadas pelo contrato e pelo harness de FC.

O módulo usa apenas a biblioteca padrão para que a validação continue
reprodutível antes de instalar dependências de treinamento.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


CANONICAL_KEYS = {"action", "tool", "arguments"}


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_registry(path: str | Path) -> dict[str, dict[str, Any]]:
    document = load_json(path)
    tools = document.get("tools", []) if isinstance(document, dict) else []
    registry: dict[str, dict[str, Any]] = {}
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            raise ValueError("catálogo contém uma ferramenta sem nome válido")
        name = tool["name"]
        if name in registry:
            raise ValueError(f"ferramenta duplicada no catálogo: {name}")
        registry[name] = tool
    if not registry:
        raise ValueError("catálogo sem ferramentas")
    return registry


def _check_value(value: Any, specification: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    expected = specification.get("type")
    if expected == "boolean" and not isinstance(value, bool):
        errors.append(f"{path}: esperado boolean, recebido {type(value).__name__}")
        return errors
    if expected == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
        errors.append(f"{path}: esperado integer, recebido {type(value).__name__}")
        return errors
    if expected == "string" and not isinstance(value, str):
        errors.append(f"{path}: esperado string, recebido {type(value).__name__}")
        return errors
    if expected == "object" and not isinstance(value, dict):
        errors.append(f"{path}: esperado object, recebido {type(value).__name__}")
        return errors

    if "enum" in specification and value not in specification["enum"]:
        errors.append(f"{path}: valor fora do enum {specification['enum']!r}")
    if isinstance(value, int) and not isinstance(value, bool):
        if "minimum" in specification and value < specification["minimum"]:
            errors.append(f"{path}: valor menor que {specification['minimum']}")
        if "maximum" in specification and value > specification["maximum"]:
            errors.append(f"{path}: valor maior que {specification['maximum']}")
    if isinstance(value, str) and "maxLength" in specification:
        if len(value) > specification["maxLength"]:
            errors.append(f"{path}: texto excede maxLength={specification['maxLength']}")
    return errors


def validate_target(target: Any, registry: dict[str, dict[str, Any]]) -> list[str]:
    """Valida a saída canônica e o schema específico da ferramenta."""

    if not isinstance(target, dict):
        return ["target: esperado objeto"]
    errors: list[str] = []
    missing = CANONICAL_KEYS - set(target)
    extra = set(target) - CANONICAL_KEYS
    errors.extend(f"target: campo obrigatório ausente: {key}" for key in sorted(missing))
    errors.extend(f"target: campo não permitido: {key}" for key in sorted(extra))
    if errors:
        return errors

    action = target["action"]
    tool_name = target["tool"]
    arguments = target["arguments"]
    if action not in {"call", "abstain"}:
        errors.append("target.action: esperado call ou abstain")
        return errors

    if action == "abstain":
        if tool_name is not None:
            errors.append("target.tool: deve ser null quando action=abstain")
        if arguments != {}:
            errors.append("target.arguments: deve ser {} quando action=abstain")
        return errors

    if not isinstance(tool_name, str) or not tool_name:
        errors.append("target.tool: esperado nome de ferramenta não vazio")
        return errors
    if tool_name not in registry:
        errors.append(f"target.tool: ferramenta desconhecida: {tool_name}")
        return errors
    if not isinstance(arguments, dict):
        errors.append("target.arguments: esperado objeto")
        return errors

    argument_schema = registry[tool_name].get("arguments", {})
    properties = argument_schema.get("properties", {})
    required = set(argument_schema.get("required", []))
    actual = set(arguments)
    errors.extend(
        f"target.arguments: argumento obrigatório ausente: {key}"
        for key in sorted(required - actual)
    )
    if argument_schema.get("additionalProperties") is False:
        errors.extend(
            f"target.arguments: argumento não permitido: {key}"
            for key in sorted(actual - set(properties))
        )
    for key, value in arguments.items():
        if key in properties:
            errors.extend(_check_value(value, properties[key], f"target.arguments.{key}"))
    return errors


def canonical_target(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": target.get("action"),
        "tool": target.get("tool"),
        "arguments": target.get("arguments", {}),
    }


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_json_object(raw: str) -> tuple[dict[str, Any] | None, bool, str | None]:
    """Extrai o primeiro objeto JSON válido de uma resposta textual."""

    if not isinstance(raw, str):
        return None, False, "saída bruta não é string"
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            return candidate, True, None
    return None, False, "nenhum objeto JSON foi encontrado"


def parse_qwen_tool_markup(raw: str) -> dict[str, Any] | None:
    """Converte o formato funcional observado no tokenizer do Qwen.

    Exemplo aceito:
    <function=wifi_set_state><parameter=enabled>True</parameter></function>
    """

    if not isinstance(raw, str):
        return None
    function_match = re.search(r"<function=([^>]+)>", raw)
    if function_match is None:
        return None
    function_name = function_match.group(1).strip()
    arguments: dict[str, Any] = {}
    for parameter_match in re.finditer(
        r"<parameter=([^>]+)>(.*?)</parameter>", raw, flags=re.DOTALL
    ):
        key = parameter_match.group(1).strip()
        arguments[key] = _parse_scalar(parameter_match.group(2))
    return {"action": "call", "tool": function_name, "arguments": arguments}


def parse_prediction_record(record: Any) -> tuple[dict[str, Any] | None, bool, str | None]:
    """Retorna (predição, é_json, erro) a partir de uma linha de predições."""

    if not isinstance(record, dict):
        return None, False, "linha de predição não é objeto"
    if isinstance(record.get("prediction"), dict):
        return record["prediction"], True, None
    if isinstance(record.get("raw"), dict):
        return record["raw"], True, None
    raw = record.get("raw", record.get("prediction"))
    if isinstance(raw, str):
        candidate, is_json, error = parse_json_object(raw)
        if candidate is not None:
            return candidate, is_json, None
        candidate = parse_qwen_tool_markup(raw)
        if candidate is not None:
            return candidate, False, None
        return None, False, error
    if {"action", "tool", "arguments"}.issubset(record):
        return canonical_target(record), True, None
    return None, False, "registro precisa de prediction ou raw"

