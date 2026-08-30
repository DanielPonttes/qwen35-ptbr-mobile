# Fase 1 — Resultados do piloto no Neuromancer

**Data:** 2026-08-30  
**GPU:** NVIDIA GeForce RTX 5090 (32.607 MiB)  
**Modelo base:** Qwen3.5-2B do snapshot local `15852e8c16360a2fea060d615a32b45270f8a8fc`  
**Dataset:** `data/generated/fc_dataset.jsonl`, SHA-256 `3fe6a38e9232b744817ae9987682f1631465f99ec5ba30b04402d804380bc98f`  
**Aparelho Android:** não utilizado nesta fase.

## 1. Comparação pareada

Os dois modelos abaixo usam o mesmo prompt canônico, o mesmo catálogo, a mesma seed do dataset e os mesmos 120 exemplos de teste (60 chamadas e 60 abstenções). O primeiro é o Qwen base sem adapter; o segundo é o mesmo checkpoint com o adapter LoRA treinado no 5090.

| Métrica | Qwen base | Qwen + LoRA |
|---|---:|---:|
| JSON válido | 100,00% | 100,00% |
| Saída canônica válida | 85,00% | 100,00% |
| Exact match | 64,17% | 99,17% |
| Acurácia de ação | 65,00% | 99,17% |
| Seleção de ferramenta | 61,67% | 100,00% |
| Argumentos exatos | 61,67% | 100,00% |
| F1 de abstenção | 74,07% | 99,16% |
| Latência média no 5090 | 439,125 ms | 407,741 ms |
| Latência mediana no 5090 | 455,209 ms | 387,781 ms |
| Latência p95 no 5090 | 568,986 ms | 552,391 ms |

As métricas foram calculadas por `scripts/fc_eval.py`. A latência foi medida por exemplo durante a geração, sem incluir carregamento do modelo, e não representa execução Android.

## 2. Configuração LoRA

- épocas: 2;
- batch físico: 2;
- acumulação de gradiente: 8;
- passos do otimizador: 60;
- comprimento máximo: 2.048 tokens;
- `r=16`, `alpha=32`, dropout 0,05;
- taxa de aprendizado: `2e-4`;
- parâmetros treináveis: 23.340.032 de 2.236.581.696 (1,0436%);
- PyTorch `2.10.0+cu128`, Transformers `5.3.0`, PEFT `0.20.0`.

Perdas registradas:

| Época | Train loss | Dev loss |
|---:|---:|---:|
| 1 | 0,051189 | 0,001158 |
| 2 | 0,005472 | 0,002338 |

O adapter salvo em `results/qwen35_2b_lora_fc/` tem aproximadamente 109 MB e não é incluído no commit Git; o manifesto de treinamento registra o caminho, a configuração e o hash do dataset.

## 3. Interpretação e limites

O resultado mostra que o caminho técnico — saída canônica, validação estrita e LoRA no 5090 — funciona. Não demonstra ainda que o modelo generaliza para comandos naturais reais, fala, variações regionais ou APIs Android executáveis.

O teste reserva a sexta formulação textual para avaliação, mas continua sendo sintético e pequeno. O ganho alto é esperado em uma tarefa com vocabulário controlado e não deve ser usado como resultado final do artigo. Antes do manuscrito, é necessário adicionar um conjunto humano/independente, casos de composição, ruído linguístico e testes de abstenção não templated.

Nenhuma métrica de latência, memória, energia ou temperatura do telefone foi coletada. Essa etapa depende do aparelho carregado e da implementação Android do despachante.

## 4. Artefatos

- base: `results/qwen35_2b_base_canonical_fc_test.predictions.jsonl` e `.metrics.json`;
- adapter: `results/qwen35_2b_lora_fc_test.predictions.jsonl` e `.metrics.json`;
- manifesto do treino: `results/qwen35_2b_lora_fc/training_manifest.json`;
- manifesto do dataset: `data/generated/fc_dataset.manifest.json`;
- resultados da divisão anterior, arquivados em `results/archive/case_split/`.
