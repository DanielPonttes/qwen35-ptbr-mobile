# FSC Ultra-small Command-Routing Benchmark

**Status:** benchmark principal do artigo ERAMIA-RS 2026, fase experimental em 30 de agosto de 2026.

Este repositório contém um benchmark reprodutível de modelos de linguagem ultrapequenos (corte estrito: no máximo 3.000.000.000 de parâmetros) para transformar transcrições humanas em inglês em objetos JSON canônicos de comando. A entrada é texto de transcrição; não há ASR, execução de ações, APK ou medição no celular nesta fase.

## Pergunta experimental

Com o mesmo contrato, prompt, decodificação e hardware, como modelos ultrapequenos de famílias diferentes se comportam em roteamento estruturado e abstention sobre comandos humanos? O benchmark compara cinco checkpoints:

| Modelo | Parâmetros medidos | Família |
|---|---:|---|
| Qwen2.5-0.5B-Instruct | 494.032.768 | Qwen2 |
| Qwen3.5-0.8B | 873.438.784 | Qwen3.5 |
| TinyLlama-1.1B-Chat-v1.0 | 1.100.048.384 | Llama |
| SmolLM2-1.7B-Instruct | 1.711.376.384 | SmolLM |
| Qwen3.5-2B | 2.274.069.824 | Qwen3.5 |

Os pesos são baixados para o cache privado do Neuromancer e não são versionados.

## Dados e leakage

O corpus humano é o Fluent Speech Commands (FSC), usado somente por suas transcrições. A derivação aceita quatro combinações nativas (`media_control` play/pause e `volume_adjust` up/down) e transforma as demais em abstention de política explicitamente derivada. O contrato está em `data/tools/fsc_command_benchmark.json`.

Há dois protocolos balanceados, cada um com 7.478 chamadas e 7.478 abstentions:

- `official`: split oficial disjunto por falante, usado para medir transferência de falante e revelar sobreposição lexical;
- `phrase_disjoint`: grupos de transcrição normalizada não atravessam train/dev/test, usado como controle principal de leakage lexical.

O avaliador reporta métricas por item e métricas cluster-aware por `template_id`, incluindo bootstrap por grupo. Os controles `always_abstain` e `lexical` são obrigatórios.

## Reprodução no Neuromancer

```bash
cd /home/daniel/qwen35-ptbr-mobile
PY='/home/daniel/Área de trabalho/swarm-emotions-tag/python-ml/.venv/bin/python'

# validar os datasets privados derivados
$PY scripts/validate_fc_dataset.py data/external/fluent_speech_commands_command_benchmark.jsonl \
  --registry data/tools/fsc_command_benchmark.json --expected-locale en-US \
  --allow-text-duplicates --group-field speaker_id --expected-total 14956
$PY scripts/validate_fc_dataset.py data/external/fluent_speech_commands_command_benchmark_phrase_disjoint.jsonl \
  --registry data/tools/fsc_command_benchmark.json --expected-locale en-US \
  --allow-text-duplicates --group-field template_id --expected-total 14956

# executar a matriz zero-shot na RTX 5090; --skip-existing permite retomar
HF_HUB_OFFLINE=1 $PY scripts/run_ultrasmall_benchmark.py --skip-training --skip-existing
```

O runner registra comandos em `logs/ultrasmall/` e a matriz em `results/fsc_ultrasmall_benchmark_manifest.json`. Métricas agregadas podem ser versionadas; predições por exemplo, adapters, transcrições derivadas e pesos permanecem ignorados pelo Git.

## Estrutura relevante

```text
benchmarks/ultrasmall_models.json       matriz, cap e revisões dos checkpoints
data/tools/fsc_command_benchmark.json   contrato genérico de duas operações
scripts/prepare_fsc_android_fc.py       derivação determinística do FSC
scripts/run_fc_baselines.py             controles always-abstain e lexical
scripts/run_qwen_fc_baseline.py         inferência comum aos checkpoints
scripts/fc_eval.py                      métricas, grupos e intervalos
scripts/run_ultrasmall_benchmark.py     runner da matriz
paper/paper.tex                         manuscrito SBC
```

O nome histórico de `prepare_fsc_android_fc.py` é mantido por compatibilidade; o contrato e o artigo atuais não fazem alegação Android.

## Governança e escopo

O arquivo bruto FSC fica fora do repositório, com hashes no manifesto privado. A licença do FSC é registrada como CC BY-NC-ND 4.0 para pesquisa acadêmica; o projeto não redistribui transcrições, áudio ou predições por exemplo. O diretório `app/`, os binários ARM e `scripts/benchmark.sh` pertencem ao protótipo móvel legado e não fazem parte da evidência deste artigo.
