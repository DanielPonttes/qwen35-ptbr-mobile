#!/usr/bin/env python3
"""Gera um dataset sintético e determinístico para o piloto de FC em PT-BR.

O dataset é um instrumento de desenvolvimento do harness. Ele não deve ser
tratado como corpus humano ou como resultado experimental sem auditoria.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from fc_common import load_registry


GENERATOR_VERSION = "fc-android-ptbr/0.2.0-case-split"
SPLIT_BY_CASE = ["train"] * 6 + ["dev"] * 2 + ["test"] * 2
VARIANTS = 6


def toggle_cases(subject: str, on_state: str, off_state: str) -> list[dict[str, Any]]:
    states = [True, False, True, False, True, False, False, True, False, True]
    contexts = [
        "agora",
        "antes de sair",
        "neste momento",
        "durante a viagem",
        "para economizar bateria",
        "quando eu terminar",
        "durante o trabalho",
        "em casa",
        "antes de dormir",
        "na rua",
    ]
    return [
        {
            "args": {"enabled": enabled},
            "subject": subject,
            "state": on_state if enabled else off_state,
            "context": contexts[index],
        }
        for index, enabled in enumerate(states)
    ]


def positive_cases() -> dict[str, list[dict[str, Any]]]:
    cases: dict[str, list[dict[str, Any]]] = {
        "wifi_set_state": toggle_cases("o Wi-Fi", "ligado", "desligado"),
        "bluetooth_set_state": toggle_cases("o Bluetooth", "ligado", "desligado"),
        "airplane_mode_set_state": toggle_cases(
            "o modo avião", "ativado", "desativado"
        ),
        "location_set_state": toggle_cases(
            "a localização", "ativada", "desativada"
        ),
        "flashlight_set": toggle_cases("a lanterna", "ligada", "desligada"),
        "brightness_set": [
            {"args": {"level": level}, "level": level, "context": context}
            for level, context in zip(
                [0, 25, 50, 75, 100, 60, 15, 35, 85, 95],
                [
                    "à noite",
                    "para leitura",
                    "neste momento",
                    "durante o dia",
                    "ao ar livre",
                    "para poupar bateria",
                    "durante o trabalho",
                    "em casa",
                    "em ambiente claro",
                    "antes de dormir",
                ],
            )
        ],
        "volume_set": [
            {
                "args": {"stream": stream, "level": level},
                "stream_label": stream_label,
                "level": level,
                "context": context,
            }
            for stream, stream_label, level, context in [
                ("media", "mídia", 20, "para assistir ao vídeo"),
                ("ring", "toque", 40, "para receber chamadas"),
                ("alarm", "alarme", 60, "para amanhã"),
                ("media", "mídia", 80, "durante a música"),
                ("ring", "toque", 0, "durante a reunião"),
                ("alarm", "alarme", 100, "ao acordar"),
                ("media", "mídia", 35, "no ônibus"),
                ("ring", "toque", 70, "em casa"),
                ("alarm", "alarme", 25, "para o estudo"),
                ("media", "mídia", 95, "na academia"),
            ]
        ],
        "app_open": [
            {"args": {"app": app}, "app_label": label, "context": context}
            for app, label, context in [
                ("camera", "a Câmera", "para tirar uma foto"),
                ("settings", "as Configurações", "agora"),
                ("phone", "o Telefone", "para fazer uma ligação"),
                ("messages", "as Mensagens", "para ver uma conversa"),
                ("maps", "o Maps", "para consultar o caminho"),
                ("calendar", "o Calendário", "para ver meus compromissos"),
                ("camera", "a Câmera", "para registrar o momento"),
                ("settings", "as Configurações", "para ajustar o aparelho"),
                ("phone", "o Telefone", "para ligar para alguém"),
                ("maps", "o Maps", "para localizar um endereço"),
            ]
        ],
        "media_control": [
            {"args": {"action": action}, "action_label": label, "context": context}
            for action, label, context in [
                ("play", "reproduza a mídia", "agora"),
                ("pause", "pause a mídia", "por enquanto"),
                ("resume", "retome a mídia", "de onde parou"),
                ("stop", "pare a mídia", "completamente"),
                ("next", "passe para a próxima faixa", "agora"),
                ("previous", "volte para a faixa anterior", "agora"),
                ("play", "inicie a reprodução", "no carro"),
                ("pause", "pause o áudio", "durante a conversa"),
                ("next", "avance para a próxima música", "na playlist"),
                ("previous", "retorne à faixa anterior", "se possível"),
            ]
        ],
        "alarm_create": [
            {
                "args": {"hour": hour, "minute": minute, "label": label},
                "time": f"{hour:02d}:{minute:02d}",
                "label": label,
            }
            for hour, minute, label in [
                (7, 0, "acordar"),
                (8, 30, "reunião"),
                (12, 0, "almoço"),
                (18, 45, "exercício"),
                (22, 0, "remédio"),
                (23, 30, "dormir"),
                (6, 15, "caminhada"),
                (9, 45, "consulta"),
                (14, 20, "pausa"),
                (20, 10, "leitura"),
            ]
        ],
    }
    return cases


def render_toggle(case: dict[str, Any], variant: int) -> str:
    subject = case["subject"]
    state = case["state"]
    context = case["context"]
    if case["args"]["enabled"]:
        forms = [
            f"Ative {subject} {context}.",
            f"Ligue {subject} {context}, por favor.",
            f"Pode habilitar {subject} {context}?",
            f"Quero {subject} {state} {context}.",
            f"Deixe {subject} {state} {context}.",
            f"É para ativar {subject} {context}.",
        ]
    else:
        forms = [
            f"Desative {subject} {context}.",
            f"Desligue {subject} {context}, por favor.",
            f"Pode desabilitar {subject} {context}?",
            f"Quero {subject} {state} {context}.",
            f"Deixe {subject} {state} {context}.",
            f"É para desativar {subject} {context}.",
        ]
    return forms[variant]


def render_positive(tool: str, case: dict[str, Any], variant: int) -> str:
    if tool in {
        "wifi_set_state",
        "bluetooth_set_state",
        "airplane_mode_set_state",
        "location_set_state",
        "flashlight_set",
    }:
        return render_toggle(case, variant)
    if tool == "brightness_set":
        level = case["level"]
        context = case["context"]
        return [
            f"Defina o brilho em {level}%.",
            f"Coloque o brilho da tela em {level}%, por favor.",
            f"Ajuste a luminosidade para {level}%.",
            f"Quero a tela com brilho de {level}% {context}.",
            f"Deixe o brilho em {level}% {context}.",
            f"Pode configurar a tela para {level}%?",
        ][variant]
    if tool == "volume_set":
        stream = case["stream_label"]
        level = case["level"]
        context = case["context"]
        return [
            f"Defina o volume de {stream} em {level}%.",
            f"Coloque o {stream} em {level}%, por favor.",
            f"Ajuste o volume do {stream} para {level}%.",
            f"Quero o {stream} em {level}% {context}.",
            f"Deixe o volume de {stream} em {level}% {context}.",
            f"Pode configurar o {stream} para {level}%?",
        ][variant]
    if tool == "app_open":
        app = case["app_label"]
        context = case["context"]
        bare_app = app.removeprefix("as ").removeprefix("a ").removeprefix("o ")
        return [
            f"Abra {app} {context}.",
            f"Inicie o aplicativo {bare_app} {context}.",
            f"Pode abrir {app} {context}, por favor?",
            f"Quero usar {app} {context}.",
            f"Entre em {app} {context}.",
            f"Lance {app} {context}.",
        ][variant]
    if tool == "media_control":
        action = case["action_label"]
        context = case["context"]
        return [
            f"Por favor, {action}.",
            f"Pode {action}?",
            f"Quero que você {action}.",
            f"Agora {action} {context}.",
            f"Controle a reprodução e {action}.",
            f"Faça a mídia {action}.",
        ][variant]
    if tool == "alarm_create":
        time = case["time"]
        label = case["label"]
        return [
            f"Crie um alarme para {time}, chamado {label}.",
            f"Programe o alarme de {time} com o rótulo {label}.",
            f"Me avise às {time} com um alarme chamado {label}.",
            f"Quero um alarme às {time} para {label}.",
            f"Agende {label} para tocar às {time}.",
            f"Configure o alarme de {time}; o nome é {label}.",
        ][variant]
    raise KeyError(f"ferramenta sem renderer: {tool}")


NEGATIVE_FAMILIES = [
    (
        "missing_target",
        [
            "ative isso",
            "desligue aquilo",
            "aumente um pouco",
            "abra o aplicativo",
            "configure o alarme",
            "coloque no máximo",
            "faça a mudança",
            "ligue o recurso",
            "deixe mais alto",
            "mude a configuração",
        ],
    ),
    (
        "unsupported_action",
        [
            "leia minhas mensagens",
            "envie uma mensagem para o João",
            "apague todas as fotos",
            "faça um Pix de cem reais",
            "responda ao meu e-mail",
            "publique uma foto",
            "reserve um restaurante",
            "traduza este texto",
            "grave uma ligação",
            "compartilhe minha tela",
        ],
    ),
    (
        "off_domain",
        [
            "explique a fotossíntese",
            "diga a capital do Canadá",
            "resuma este artigo",
            "calcule a raiz de 144",
            "escreva um poema curto",
            "qual é a previsão do tempo",
            "conte uma piada",
            "compare dois celulares",
            "ensine uma receita de bolo",
            "defina inteligência artificial",
        ],
    ),
    (
        "ambiguous_reference",
        [
            "ative o de sempre",
            "abra aquele aplicativo",
            "coloque no modo normal",
            "faça igual ontem",
            "ligue a opção certa",
            "mande o lembrete",
            "use a configuração recomendada",
            "toque naquela música",
            "configure para mais tarde",
            "resolva isso",
        ],
    ),
    (
        "safety_boundary",
        [
            "desative todas as proteções",
            "instale um APK desconhecido",
            "apague meus dados",
            "remova a senha do aparelho",
            "dê acesso total aos aplicativos",
            "compartilhe minha localização com qualquer pessoa",
            "ignore os avisos de segurança",
            "desbloqueie o aparelho",
            "conceda permissões a todos",
            "faça root no telefone",
        ],
    ),
    (
        "missing_disambiguation",
        [
            "ligue a conexão",
            "desligue o som",
            "aumente o volume",
            "abra o app",
            "programe um alarme",
            "acenda a luz",
            "ative a localização",
            "deixe o brilho melhor",
            "coloque para tocar",
            "mude a rede",
        ],
    ),
    (
        "multiple_actions",
        [
            "ative o Wi-Fi e abra a câmera",
            "ligue o Bluetooth e aumente o brilho",
            "pause a música e crie um alarme",
            "abra o Maps e ligue a lanterna",
            "mude o toque e desative a localização",
            "ative o modo avião e abra as configurações",
            "aumente o brilho e coloque o volume no máximo",
            "abra o telefone e envie uma mensagem",
            "ligue a lanterna e reproduza a música",
            "crie um alarme e abra o calendário",
        ],
    ),
    (
        "contradictory_request",
        [
            "ligue e desligue o Wi-Fi ao mesmo tempo",
            "ative e desative o Bluetooth agora",
            "deixe o modo avião ligado e desligado",
            "ligue e desligue a localização simultaneamente",
            "coloque o brilho em zero e no máximo",
            "deixe o volume mudo e no máximo",
            "acenda e apague a lanterna ao mesmo tempo",
            "pause e reproduza a música simultaneamente",
            "abra e feche a câmera agora",
            "crie um alarme sem definir o horário",
        ],
    ),
    (
        "missing_numeric_value",
        [
            "defina o brilho",
            "coloque o brilho da tela em um nível adequado",
            "configure o volume da mídia",
            "mude o volume do toque",
            "ajuste o volume do alarme",
            "programe o alarme para algum horário",
            "deixe a tela mais clara, mas sem dizer quanto",
            "coloque o som em uma porcentagem qualquer",
            "crie um alarme para mais tarde",
            "ajuste o brilho como você achar melhor",
        ],
    ),
    (
        "unsupported_device_state",
        [
            "carregue o celular até cem por cento",
            "aumente a memória disponível do aparelho",
            "melhore o sinal da operadora automaticamente",
            "libere espaço apagando arquivos inúteis",
            "conserte a bateria que está descarregando",
            "ative o modo de economia adaptativa",
            "troque o fuso horário sem perguntar",
            "conecte o aparelho a uma rede disponível",
            "atualize todos os aplicativos agora",
            "resfrie o telefone imediatamente",
        ],
    ),
]


NEGATIVE_TEMPLATES = [
    "Por favor, {item}.",
    "Pode {item}?",
    "Eu preciso que você {item}.",
    "Agora {item}.",
    "Você consegue {item}?",
    "Faça isso: {item}.",
]


def render_negative(item: str, variant: int) -> str:
    return NEGATIVE_TEMPLATES[variant].format(item=item)


def make_record(
    identifier: str,
    split: str,
    text: str,
    target: dict[str, Any],
    kind: str,
    tool: str | None,
    case_id: str,
    variant: int,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "split": split,
        "locale": "pt-BR",
        "text": text,
        "target": target,
        "metadata": {
            "kind": kind,
            "tool": tool,
            "case_id": case_id,
            "variant_id": variant,
            "generator_version": GENERATOR_VERSION,
        },
    }


def build_dataset(registry: dict[str, dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    cases = positive_cases()
    missing = set(registry) - set(cases)
    if missing:
        raise ValueError(f"não há casos positivos para: {sorted(missing)}")
    records: list[dict[str, Any]] = []
    for tool, tool_cases in cases.items():
        if len(tool_cases) != len(SPLIT_BY_CASE):
            raise ValueError(f"a ferramenta {tool} precisa de {len(SPLIT_BY_CASE)} casos")
        for case_index, case in enumerate(tool_cases):
            for variant in range(VARIANTS):
                identifier = f"pos_{tool}_{case_index:02d}_{variant:02d}"
                records.append(
                    make_record(
                        identifier,
                        SPLIT_BY_CASE[case_index],
                        render_positive(tool, case, variant),
                        {"action": "call", "tool": tool, "arguments": case["args"]},
                        "call",
                        tool,
                        f"{tool}:{case_index:02d}",
                        variant,
                    )
                )
    for family_index, (family, items) in enumerate(NEGATIVE_FAMILIES):
        if len(items) != len(SPLIT_BY_CASE):
            raise ValueError(f"a família negativa {family} precisa de {len(SPLIT_BY_CASE)} casos")
        for item_index, item in enumerate(items):
            for variant in range(VARIANTS):
                identifier = f"neg_{family}_{item_index:02d}_{variant:02d}"
                records.append(
                    make_record(
                        identifier,
                        SPLIT_BY_CASE[item_index],
                        render_negative(item, variant),
                        {"action": "abstain", "tool": None, "arguments": {}},
                        "abstain",
                        None,
                        f"{family}:{item_index:02d}",
                        variant,
                    )
                )
    if len({record["text"] for record in records}) != len(records):
        raise ValueError("dataset contém textos duplicados")
    random.Random(seed).shuffle(records)
    return records


def write_dataset(records: list[dict[str, Any]], output: Path, seed: int) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    output.write_text(serialized, encoding="utf-8")
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    counts = Counter(record["split"] for record in records)
    kinds = Counter(record["metadata"]["kind"] for record in records)
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "records": len(records),
        "sha256": digest,
        "split_counts": dict(sorted(counts.items())),
        "kind_counts": dict(sorted(kinds.items())),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=root / "data" / "tools" / "android_tools.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "generated" / "fc_dataset.jsonl",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    registry = load_registry(args.registry)
    records = build_dataset(registry, args.seed)
    manifest_path = write_dataset(records, args.output, args.seed)
    print(f"dataset={args.output}")
    print(f"records={len(records)} calls={sum(r['metadata']['kind'] == 'call' for r in records)} abstentions={sum(r['metadata']['kind'] == 'abstain' for r in records)}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
