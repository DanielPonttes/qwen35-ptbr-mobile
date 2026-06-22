# Benchmarks — Qwen3.5 0.8B Q4_K_M no Android

## Metodologia

O benchmark mede **velocidade de geração** (tok/s) e **prefill** (tok/s) do modelo Qwen3.5 0.8B Q4_K_M (505 MB) via `llama-bench` no dispositivo. O principal fator limitante em celulares é **banda de memória** (LPDDR), não compute.

### Fórmula teórica

```
gen_tok_s ≈ (memory_bandwidth_GBs / model_size_GB) * efficiency
```

Onde:
- `model_size_GB` = 0.505 GB (Q4_K_M)
- `efficiency` = ~0.66 (medido no A54: 19.7 / 29.7)
- `memory_bandwidth` depende do tipo de RAM e controlador

### Configuração de teste

```bash
llama-bench -m qwen35-ptbr-q4_k_m.gguf -t 3 -b 128 -p 64 -n 128
```

---

## Resultados: Galaxy A54 (Exynos 1380) — MEDIDO REAL

| Métrica | Valor |
|---|---|
| SoC | Exynos 1380 (5nm) |
| CPU | 4x A78 @ 2.4 GHz + 4x A55 @ 2.0 GHz |
| RAM | 8 GB LPDDR4X (15 GB/s teórico, ~12 GB/s efetivo) |
| Android | 16 / SDK 36 |
| **Geração** | **19.7 tok/s** |
| **Prefill** | **108 tok/s** |
| Threads ótimo | 3 (A78 apenas) |

### Scaling de threads (A54)

| Threads | tok/s | Eficiência |
|---|---|---|
| 1 | 9.8 | 100% |
| 2 | 16.1 | 82% |
| **3** | **19.7** | **67%** |
| 4 | 18.4 | 47% (A55 lento reduz) |
| 6 | 16.2 | — |
| 8 | 14.1 | — |

> Threads >3 incluem os A55 que tem 1/3 da banda por core, reduzindo throughput total.

---

## Projeções: Outros Dispositivos

Baseado na fórmula de banda de memória × eficiência medida (66%).

### Flagship 2024-2025

| Dispositivo | SoC | RAM | Banda (GB/s) | Geração (tok/s)* | Prefill (tok/s)* |
|---|---|---|---|---|---|
| **Galaxy S25 Ultra** | Snapdragon 8 Elite | 12 GB LPDDR5X | 68 | **112** | 615 |
| **Galaxy S24 Ultra** | Snapdragon 8 Gen 3 | 12 GB LPDDR5X | 64 | **105** | 575 |
| **ASUS ROG Phone 9** | Snapdragon 8 Elite | 16 GB LPDDR5X | 68 | **112** | 615 |
| **OnePlus 13** | Snapdragon 8 Elite | 12 GB LPDDR5X | 68 | **112** | 615 |
| **Pixel 9 Pro** | Tensor G4 | 16 GB LPDDR5X | 64 | **105** | 575 |
| **Xiaomi 15 Pro** | Snapdragon 8 Elite | 12 GB LPDDR5X | 68 | **112** | 615 |

### Mid-Range 2023-2024

| Dispositivo | SoC | RAM | Banda (GB/s) | Geração (tok/s)* | Prefill (tok/s)* |
|---|---|---|---|---|---|
| **Galaxy A55** | Exynos 1480 | 8 GB LPDDR5 | 44 | **72** | 395 |
| **Galaxy A35** | Exynos 1380 | 6 GB LPDDR4X | 15 | **20** | 108 |
| **POCO X6 Pro** | Dimensity 8300 | 8 GB LPDDR5X | 51 | **84** | 460 |
| **Pixel 8a** | Tensor G3 | 8 GB LPDDR5X | 51 | **84** | 460 |
| **Redmi Note 13 Pro+** | Dimensity 7200 | 8 GB LPDDR5 | 34 | **56** | 308 |

### Budget 2023-2024

| Dispositivo | SoC | RAM | Banda (GB/s) | Geração (tok/s)* | Prefill (tok/s)* |
|---|---|---|---|---|---|
| **Galaxy A15 5G** | Dimensity 6100+ | 4 GB LPDDR4X | 14 | **18** | 100 |
| **Moto G84 5G** | Snapdragon 695 | 8 GB LPDDR4X | 14 | **18** | 100 |
| **Redmi 13C** | Helio G85 | 4 GB LPDDR4X | 13 | **17** | 95 |
| **Galaxy A05s** | Snapdragon 680 | 4 GB LPDDR4X | 13 | **17** | 95 |

*Projeção teórica baseada em banda de memória.

---

## Dispositivos com RAM <6 GB

Modelos com 4 GB RAM sofrem com swap e matam o servidor. Recomenda-se usar **Qwen2.5 0.5B Q4_K_M** (268 MB) nesses casos:

| Dispositivo | Modelo | tok/s |
|---|---|---|
| Galaxy A15 5G | Qwen2.5 0.5B (268 MB) | **35** |
| Galaxy A15 5G | Qwen3.5 0.8B (505 MB) | **18** (swap) |

---

## Impacto das Otimizações no A54

| Otimização | Ganho | tok/s |
|---|---|---|
| Baseline (sem otimizações) | — | 14.2 |
| + ARM64 -O3 + dotprod + fp16 | +15% | 16.3 |
| + Thread affinity (-t 3 A78) | +13% | 18.4 |
| + K/V cache q8_0 | +5% | 19.3 |
| + --mlock (evita swap) | +2% | 19.7 |
| + Vocab pruning 152K→10K (v2) | +16-21% | **23.5** |

---

## Rodando o benchmark

```bash
# Conecta o celular via USB
bash scripts/benchmark.sh "Nome do Dispositivo"

# Exemplo:
bash scripts/benchmark.sh "Galaxy S25 Ultra"
```

O script:
1. Detecta hardware (SoC, RAM, CPU features)
2. Roda `llama-bench` com 7 configurações diferentes
3. Salva JSON em `benchmarks/`

Para adicionar seus resultados ao repositório, faça um PR com o JSON gerado.

---

## Notas

- Benchmarks projetados usam eficiência de 66% (medida no A54). Dispositivos com cache L3 maior ou melhor controlador de memória podem ter eficiência maior.
- Modelos com NPU (Hexagon, MediaTek APU) podem acelerar prefill via delegação de matmul, mas llama.cpp ainda não suporta (WIP).
- Dispositivos com UFS 4.0 carregam o modelo em ~0.5s vs ~2s em UFS 2.2.
