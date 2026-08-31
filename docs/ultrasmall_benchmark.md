# Ultra-small benchmark: protocolo congelado e resultados

**Data:** 31 de agosto de 2026
**Status:** matriz zero-shot concluída no Neuromancer, 10/10 jobs sem falha
**GPU:** NVIDIA GeForce RTX 5090, 32 GB

## Escopo

O artigo atual mede modelos com no máximo 3.000.000.000 de parâmetros em roteamento estruturado de transcrições humanas em inglês para JSON. Não mede Android, PT-BR, ASR, áudio, APK, execução, bateria, térmica ou celular.

## Matriz

| Nome | Checkpoint | Parâmetros medidos | Família |
|---|---|---:|---|
| `qwen25_0_5b` | `Qwen/Qwen2.5-0.5B-Instruct` | 494.032.768 | Qwen2 |
| `qwen35_0_8b` | `Qwen/Qwen3.5-0.8B` | 873.438.784 | Qwen3.5 |
| `tinyllama_1_1b` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.100.048.384 | Llama |
| `smollm2_1_7b` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.711.376.384 | SmolLM |
| `qwen35_2b` | `Qwen/Qwen3.5-2B` | 2.274.069.824 | Qwen3.5 |

As contagens são a soma das formas dos tensores safetensors locais. `Qwen/Qwen2.5-3B-Instruct` foi excluído porque mede 3.085.938.688 parâmetros; `Qwen/Qwen3.5-4B-Base` mede 4.659.865.088.

## Dados e controles de leakage

O FSC contém 30.043 gravações humanas, 97 falantes e 31 intents nativos. A derivação balanceada tem 14.956 linhas, com 7.478 chamadas e 7.478 abstentions. Quatro combinações nativas viram chamadas; todo o restante vira uma abstention de política definida pelo experimento.

| Protocolo | Train | Dev | Test | Controle | Templates train/dev/test |
|---|---:|---:|---:|---|---|
| `official` | 11.442 | 1.580 | 1.934 | falante disjunto | 248 / 245 / 247 |
| `phrase_disjoint` | 10.458 | 2.202 | 2.296 | template disjunto | 178 / 32 / 38 |

O oficial tem interseções de template 245 (train-dev), 247 (train-test) e 244 (dev-test), embora os falantes sejam disjuntos. O phrase-disjoint tem zero interseções de template; seus 38 clusters de teste produzem incerteza larga e são reportados com cluster bootstrap.

## Resultados principais

Todos os valores abaixo são do teste phrase-disjoint, salvo indicação. `Valid` significa que a resposta satisfaz o contrato, não apenas que contém algum objeto JSON.

| Sistema | Parseável | Valid | Exact | Abstain F1 | Exact macro por template |
|---|---:|---:|---:|---:|---:|
| Always abstain | 100,00% | 100,00% | 50,00% | 66,67% | 76,32% |
| Lexical control | 100,00% | 100,00% | **77,40%** | 81,56% | **89,47%** |
| Qwen2.5-0.5B | 70,17% | 0,00% | 0,00% | -- | 0,00% |
| Qwen3.5-0.8B | 100,00% | 60,15% | 43,60% | 69,92% | 50,00% |
| TinyLlama-1.1B | 43,42% | 0,00% | 0,00% | -- | 0,00% |
| SmolLM2-1.7B | 96,47% | 0,00% | 0,00% | -- | 0,00% |
| Qwen3.5-2B | 100,00% | 100,00% | 50,00% | 66,67% | 76,32% |

No oficial, o exact match foi: Qwen2.5-0.5B 0,00%; Qwen3.5-0.8B 54,65%; TinyLlama 0,00%; SmolLM2 0,00%; Qwen3.5-2B 50,98%; controle lexical 77,92%; always abstain 50,00%.

## Interpretação congelada

O controle lexical supera todos os modelos zero-shot nos dois protocolos. O Qwen3.5-2B obtém validade de contrato perfeita, mas exatidão de 50%, compatível com abstention quase universal. O Qwen3.5-0.8B é o único modelo com roteamento de chamadas substancial, mas cai de 54,65% para 43,60% quando templates repetidos são retirados. Os modelos externos frequentemente produzem texto ou JSON plausível fora do esquema; esses erros permanecem no denominador.

## Reprodução

```bash
cd /home/daniel/qwen35-ptbr-mobile
PY='/home/daniel/Área de trabalho/swarm-emotions-tag/python-ml/.venv/bin/python'
HF_HUB_OFFLINE=1 $PY scripts/run_ultrasmall_benchmark.py --skip-training --skip-existing
$PY scripts/summarize_ultrasmall_results.py --output results/fsc_ultrasmall_summary.json
```

O runner registra os comandos em `logs/ultrasmall/`. Os pesos ficam no cache privado Hugging Face; datasets derivados, predições e adapters são server-local e ignorados pelo Git. Apenas manifestos, código, logs agregados e métricas são candidatos a commit.
