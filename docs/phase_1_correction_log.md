# Registro de correções — Fase 1

**Data:** 2026-08-30
**Servidor:** Neuromancer (`neuromancer.inf.ufpel.edu.br:50000`)
**Projeto:** `/home/daniel/qwen35-ptbr-mobile`
**Branch:** `codex/phase1-fc`
**GPU utilizada:** NVIDIA GeForce RTX 5090, 32.607 MiB

## 1. Substituição do revisor

O modelo solicitado foi localizado no inventário do Neuromancer como `opencode-go/glm-5.3-flash`. Foram feitas duas tentativas:

1. revisão do dossiê factual com modo de planejamento e esforço alto;
2. teste mínimo solicitando apenas `PONG`.

Ambas excederam o timeout sem produzir resposta. Portanto, não existe parecer do GLM para transcrever. O resultado foi documentado em `docs/phase_1_adversarial_review.md`; as decisões foram tomadas a partir dos pareceres Gemini/Grok disponíveis e de verificações reprodutíveis no repositório.

## 2. Correções executadas

### Dataset e leakage

- A versão anterior foi movida com `git mv` para `data/archive/fc_dataset_variant_holdout_v1/`.
- A nova versão usa `fc-android-ptbr/0.2.0-case-split` e seed `20260830`.
- Resultado: 1.200 registros, 600 chamadas, 600 abstenções, 200 `case_id` únicos.
- Splits: train 720, dev 240, test 240; cada split contém metade chamadas e metade abstenções.
- Auditoria: zero casos cruzam splits e zero textos são duplicados.
- Limite: as seis formulações superficiais reaparecem em casos diferentes; o holdout não é completo por template/família semântica.

### Estatística

- `scripts/fc_eval.py`: IC95% Wilson para proporções e bootstrap determinístico de F1 de abstenção.
- `scripts/compare_fc_predictions.py`: alinhamento por `id`, IC95% e McNemar exato bilateral.
- p-values muito pequenos são preservados em notação científica, não convertidos em `0.0` por arredondamento.

### Reexecução no RTX 5090

- B0 Qwen3.5-2B sem adapter: exact match 71,67% no teste corrigido.
- LoRA seed 20260830: 99,17%; McNemar `p=2,71e-20`.
- LoRA seed 20260831: 99,17%; McNemar `p=2,71e-20`.
- LoRA seed 20260832: 97,08%; McNemar `p=6,80e-16`.
- Todas as seeds foram treinadas por dois epochs, com 90 passos de otimização, e avaliadas nos mesmos 240 exemplos.

Os números detalhados, intervalos, latências e perdas estão em `docs/phase_1_results.md` e nos JSON em `results/`.

### Exportação e integração de servidor

- Adapter representativo: `results/qwen35_2b_lora_fc_seed20260830.gguf`.
- Tamanho: 33.664.736 bytes.
- SHA-256: `c9b497df1bb35c98d2125664abb01f243622412dddf38875fb2fde84a4e93a29`.
- Base: `models/qwen35-2b-q4_k_m.gguf`.
- `llama.cpp`: commit `a66d50588`.
- Smoke call e smoke abstain passaram com JSON e contrato válidos.

## 3. Verificações realizadas

```text
python3 -m py_compile scripts/*.py                         OK
python3 -m unittest discover -s tests -p 'test_*.py'       7 testes OK
generate_fc_dataset.py + validate_fc_dataset.py             OK
treinamento LoRA seeds 20260830/31/32                      OK
avaliação HF B0 + três adapters                            OK
três comparações McNemar                                   OK
conversão GGUF F16                                         OK
dois smoke tests llama.cpp                                 OK
```

## 4. O que permanece deliberadamente pendente

- parser/validador/dispatcher FC dentro do APK Android;
- dataset humano e teste OOD/noisy/compositional;
- divisão final por famílias semânticas;
- comparação externa, ablação de dados e grid Q8/Q4/Q3;
- medições no A54 de TTFT, latência fim a fim, RAM, energia, temperatura e bateria.

Esses itens não foram simulados nem inferidos a partir da RTX 5090. O celular permaneceu carregando e não foi necessário para esta correção.

## 5. Rastreabilidade Git

O branch `codex/phase1-fc` continua local no Neuromancer, sem push e sem alteração do `master`. Adapters, GGUFs, builds e resultados grandes permanecem ignorados pelo Git; datasets, scripts, testes, manifestos, métricas, comparações e documentação são os artefatos destinados ao commit.
