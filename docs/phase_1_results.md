# Fase 1 — Resultados do piloto corrigido no Neuromancer

> **Nota de escopo:** os resultados abaixo pertencem ao piloto sintético PT-BR
> e não são a avaliação humana principal. A rodada humana em inglês baseada no
> Fluent Speech Commands está documentada separadamente em
> docs/phase_1_english_dataset.md; seus resultados serão registrados em um
> relatório próprio.

**Data de fechamento:** 2026-08-30
**GPU:** NVIDIA GeForce RTX 5090 (32.607 MiB)
**Modelo base:** Qwen3.5-2B, snapshot local `15852e8c16360a2fea060d615a32b45270f8a8fc`
**Dataset:** `data/generated/fc_dataset.jsonl`, SHA-256 `43e88020821b46cb741367bfcfda8eac5ccb1cef57d0e8b053ec02c7ebfacd1b`
**Aparelho Android:** não utilizado nesta fase.

Este documento registra um piloto de engenharia corrigido após a revisão adversarial. Ele ainda não é o resultado final do artigo.

## 1. Correção metodológica

A versão anterior usava a mesma identidade de caso em treino, desenvolvimento e teste, mudando apenas a formulação textual. Seus resultados foram preservados como histórico, mas não são usados como evidência de generalização:

- dataset e manifesto antigos: `data/archive/fc_dataset_variant_holdout_v1/`;
- predições e métricas antigas: `results/archive/variant_holdout_v1/`.

A versão `fc-android-ptbr/0.2.0-case-split` contém 1.200 registros, 200 `case_id` únicos e seis formulações por caso. Cada caso aparece em um único split: seis casos por ferramenta/família em treino, dois em desenvolvimento e dois em teste.

| Split | Registros | Chamadas | Abstenções |
|---|---:|---:|---:|
| train | 720 | 360 | 360 |
| dev | 240 | 120 | 120 |
| test | 240 | 120 | 120 |
| **total** | **1.200** | **600** | **600** |

A validação encontrou zero `case_id` atravessando splits e zero texto duplicado. As seis formas superficiais são reutilizadas entre casos; portanto, este é um holdout por caso/valores, não um holdout completo de templates ou de famílias semânticas. Essa limitação permanece explícita.

## 2. Comparação no teste por casos inéditos

Todos os sistemas receberam o mesmo prompt canônico, catálogo e 240 exemplos de teste. B0 é o Qwen base; as demais linhas são três treinamentos LoRA independentes, cada um com a seed indicada.

| Sistema | JSON válido | Canônico | Exact match | Ação | Ferramenta | Argumentos | F1 abst. |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 Qwen base | 100,00% | 87,50% | 71,67% | 71,67% | 62,50% | 62,50% | 82,55% |
| LoRA seed 20260830 | 100,00% | 100,00% | 99,17% | 100,00% | 100,00% | 98,33% | 100,00% |
| LoRA seed 20260831 | 100,00% | 100,00% | 99,17% | 99,58% | 99,17% | 98,33% | 99,59% |
| LoRA seed 20260832 | 100,00% | 99,58% | 97,08% | 97,50% | 95,00% | 94,17% | 97,96% |

IC95% de Wilson para exact match: B0 `[65,66%, 76,99%]`; seeds 20260830 e 20260831 `[97,01%, 99,77%]`; seed 20260832 `[94,10%, 98,58%]`. No teste pareado, o McNemar exato bilateral para exact match foi `p=2,71e-20` (66 pares ganhos e 0 perdidos) nas duas primeiras seeds e `p=6,80e-16` (64 ganhos e 3 perdas) na terceira.

As médias descritivas entre as três seeds LoRA foram: exact match 98,47%, ação 99,03%, seleção de ferramenta 98,06%, argumentos 96,94% e F1 de abstenção 99,18%. Elas resumem a variabilidade observada; não substituem um experimento principal com dados humanos e splits semânticos congelados.

### Latência de geração no 5090

Os valores abaixo são por exemplo, sem carregamento do modelo, e não representam latência no telefone.

| Sistema | Média | Mediana | p95 |
|---|---:|---:|---:|
| B0 Qwen base | 435,590 ms | 451,596 ms | 585,323 ms |
| LoRA seed 20260830 | 408,351 ms | 392,229 ms | 575,424 ms |
| LoRA seed 20260831 | 403,926 ms | 387,422 ms | 511,257 ms |
| LoRA seed 20260832 | 402,768 ms | 385,234 ms | 564,003 ms |

## 3. Treinamento LoRA

Cada seed usou dois epochs, batch físico 2, acumulação 8, 90 passos do otimizador, comprimento máximo 2.048, `r=16`, `alpha=32`, dropout 0,05 e taxa `2e-4`. Foram 720 exemplos de treino e 240 de desenvolvimento.

| Seed | Train loss epoch 1 | Dev loss epoch 1 | Train loss epoch 2 | Dev loss epoch 2 | Tempo |
|---:|---:|---:|---:|---:|---:|
| 20260830 | 0,042124 | 0,015946 | 0,003452 | 0,027487 | 321,688 s |
| 20260831 | 0,041768 | 0,016048 | 0,000110 | 0,003563 | 322,429 s |
| 20260832 | 0,037355 | 0,020100 | 0,001892 | 0,013435 | 322,036 s |

Foram treináveis 23.340.032 de 2.236.581.696 parâmetros (1,0436%). O runtime foi PyTorch `2.10.0+cu128`, Transformers `5.3.0` e PEFT `0.20.0`. Os manifestos completos estão em `results/qwen35_2b_lora_fc_seed*/training_manifest.json` e os pesos permanecem ignorados pelo Git.

O seed 20260830 foi usado como exemplar para o smoke de integração GGUF; isso não deve ser interpretado como seleção baseada no teste para o artigo. A análise principal deve pré-especificar como as seeds serão agregadas.

## 4. Interpretação e limites

O piloto sustenta uma conclusão restrita: neste corpus sintético, com holdout por caso e contrato controlado, o LoRA melhora substancialmente a saída estruturada do mesmo Qwen3.5-2B, e o efeito aparece nas três seeds. Ele não demonstra compreensão de comandos naturais, fala, dialetos, ruído, OOD humano, segurança operacional ou execução Android.

Ainda faltam conjunto humano independente, famílias semânticas disjuntas, casos composicionais e ruidosos, ablação de quantidade de dados, baselines externos, grid de quantização e benchmark físico. Nenhuma métrica de RAM, energia, temperatura, bateria ou latência fim a fim foi coletada no A54.

## 5. Exportação e smoke test no `llama.cpp`

O adapter da seed 20260830 foi exportado para `results/qwen35_2b_lora_fc_seed20260830.gguf` em F16.

- tamanho: 33.664.736 bytes;
- SHA-256: `c9b497df1bb35c98d2125664abb01f243622412dddf38875fb2fde84a4e93a29`;
- base: `models/qwen35-2b-q4_k_m.gguf`;
- `llama.cpp`: commit `a66d50588`;
- GPU: RTX 5090.

`scripts/run_llama_fc_smoke.py` carregou base Q4 e adapter juntos, com schema JSON, e validou:

| Caso | Saída | Validação |
|---|---|---|
| `pos_bluetooth_set_state_08_00` | `bluetooth_set_state(enabled=false)` | JSON e contrato válidos |
| `neg_missing_target_08_00` | `abstain` | JSON e contrato válidos |

O wrapper de conversão usa uma visão temporária do `config.json` para preencher `text_config.architectures`; o cache do modelo não foi alterado. O smoke confirma a cadeia HF → LoRA → GGUF → `llama.cpp` → JSON, não a execução no aparelho.

## 6. Base Android sem aparelho

Foi instalado no servidor um SDK local em `/home/daniel/android-sdk`, com plataforma Android 35, Build-Tools 35 e Platform-Tools. `cd app && ./gradlew test assembleDebug --no-daemon --console=plain` terminou com `BUILD SUCCESSFUL`.

O APK compilado é a aplicação legada de chat. Ainda não há parser FC Kotlin, camada de segurança, allowlist executável ou dispatcher integrado ao app. Por isso o A54 permaneceu apenas carregando e não foi conectado novamente.

## 7. Artefatos rastreáveis

- dataset e manifesto: `data/generated/fc_dataset.jsonl` e `.manifest.json`;
- baseline: `results/qwen35_2b_base_canonical_fc_case_test.{predictions,metrics}.json*`;
- LoRA: `results/qwen35_2b_lora_fc_seed2026083{0,1,2}_test.{predictions,metrics}.json*`;
- comparações: `results/qwen35_2b_lora_fc_seed202608{30,31,32}_mcnemar.json`;
- adapter GGUF e smoke: `results/qwen35_2b_lora_fc_seed20260830.gguf` e `results/qwen35_2b_lora_fc_seed20260830_llama_smoke*.json`;
- scripts: `generate_fc_dataset.py`, `validate_fc_dataset.py`, `fc_eval.py`, `compare_fc_predictions.py`, `convert_qwen35_lora_gguf.py` e `run_llama_fc_smoke.py`;
- correção metodológica: `docs/phase_1_correction_log.md`.
