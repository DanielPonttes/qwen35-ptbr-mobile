# Qwen3.5 0.8B — Chat PT-BR Mobile

Chat em português brasileiro rodando **localmente no celular** (offline, sem nuvem).

**Modelo:** Qwen3.5 0.8B fine-tuned para PT-BR | **Hardware alvo:** Galaxy A54 (Exynos 1380, 6 GB RAM) | **Velocidade:** ~19.5 tok/s

## Download rápido

```bash
git clone https://github.com/<user>/qwen35-ptbr-mobile.git
cd qwen35-ptbr-mobile

# Baixe o modelo GGUF (494 MB) do HuggingFace:
# https://huggingface.co/<user>/qwen35-ptbr-mobile/resolve/main/qwen35-ptbr-q4_k_m.gguf
# Coloque em: model/qwen35-ptbr-q4_k_m.gguf

# Deploy no celular (requer ADB + depuração USB):
bash scripts/deploy.sh
```

## O que está incluso

```
qwen35-ptbr-mobile/
├── binaries/          # llama.cpp ARM64 otimizado (Cortex-A78 + DotProd + FP16)
│   ├── llama-cli      # Chat via terminal
│   ├── llama-server   # API REST + WebUI
│   ├── llama-bench    # Benchmark
│   └── *.so           # Bibliotecas nativas (libggml, libllama, libomp)
├── model/             # Coloque o GGUF aqui (não versionado, use HF)
├── scripts/
│   ├── deploy.sh      # Deploy 1-clique via ADB
│   └── finetune_qwen35_ptbr.sh  # Script de fine-tuning (RTX 5090)
└── README.md
```

## Modos de uso

### 1. Navegador do celular (mais simples)

Após `deploy.sh`, abra `http://127.0.0.1:8080` no Chrome do celular.

### 2. API REST (para apps)

```bash
# Port forward do PC
adb forward tcp:9090 tcp:8080

# Chamada API
curl http://127.0.0.1:9090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explique o que é pão de queijo."}],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

### 3. Termux (chat no terminal)

```bash
cd /data/local/tmp/qwen35-ptbr
LD_LIBRARY_PATH=. ./llama-cli \
  -m models/qwen35-ptbr-q4_k_m.gguf \
  -t 3 -b 128 -c 2048 -cnv
```

## Performance (Galaxy A54)

| Métrica | Valor |
|---|---|
| Geração | **19.5 tok/s** |
| Prefill | 108 tok/s |
| RAM usada | ~500 MB |
| Tamanho modelo | 494 MB (Q4_K_M) |
| Binário | armv8.2-a+dotprod+fp16 -O3 |

## Fine-tuning PT-BR

O modelo base (Qwen3.5 0.8B Instruct) foi fine-tuned na RTX 5090 (32 GB VRAM):

- **Dataset:** 2248 conversas sintéticas PT-BR (self-instruct)
- **Método:** Full bf16 fine-tune, 3 épocas, ctx=512, lr=2e-5
- **Loss:** 1.53 → 0.58 (treino), 1.57 → 1.61 (validação)

Para reproduzir:

```bash
bash scripts/finetune_qwen35_ptbr.sh all
```

Requisitos: RTX 5090 32 GB (ou GPU ≥16 GB VRAM), CUDA 12.8, 50 GB disco.

## Otimizações aplicadas

| Camada | Técnica | Ganho |
|---|---|---|
| Compilador | `-march=armv8.2-a+dotprod+fp16 -O3 -flto` | +10-15% |
| CPU affinity | `-t 3 -b 128` (3 threads nos A78) | +13% vs -t 4 |
| KV cache | `-ctk q8_0 -ctv q8_0` | Libera ~300 MB |
| Modelo | Q4_K_M (melhor custo-benefício ARM) | Base |
| Prompt | Chat template Jinja embutido | Zero overhead |

## Roadmap

- [x] Fine-tuning PT-BR funcional
- [x] Deploy ARM64 otimizado no A54
- [x] API REST + WebUI
- [ ] Vocab pruning (152K→32K, +16% velocidade)
- [ ] LoRA adapter export (.gguf.lora)
- [ ] APK nativo (WebView wrapper)
- [ ] Speculative decoding (draft model 50M params)

## Licença

Modelo: Apache 2.0 (Qwen3.5 base) | Código: MIT
