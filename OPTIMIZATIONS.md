# Resultados das Otimizações

## 1. ✅ Thinking Mode Fix

**Problema:** Template GGUF do Qwen3.5 força `<think>\n\n</think>\n\n` vazio mesmo com `reasoning: disabled`, desperdiçando ~20% dos tokens gerados.

**Solução:** Patch binário no GGUF removendo o bloco `else` vazio do template Jinja.

**Arquivo:** `model/qwen35-ptbr-q4_k_m-nothink.gguf` (mesmo tamanho, template corrigido)

**Template modificado:**
```jinja
# Antes (bug):
{%- if enable_thinking is defined and enable_thinking is true %}
    {{- '<think>\n' }}
{%- else %}
    {{- '<think>\n\n</think>\n\n' }}  ← EMPTY THINK BLOCK
{%- endif %}

# Depois (corrigido):
{%- if enable_thinking is defined and enable_thinking is true %}
    {{- '<think>\n' }}
{%- endif %}
```

**Ganho:** ~20% menos tokens por resposta (sem o bloco `<think>` vazio forçado).

---

## 2. ❌ IQ2_XXS / IQ2_XS — Rejeitado

**Hipótese:** Quantização mais agressiva (IQ2_XXS = 336 MB, IQ2_XS = 348 MB) poderia dar mais velocidade no A54 sacrificando qualidade.

**Resultados (perplexidade no dataset PT-BR):**

| Quant | Tamanho | BPW | PPL |
|---|---|---|---|
| Q4_K_M | 505 MB | 4.65 | **9.4** |
| IQ2_XS | 348 MB | 3.75 | **34.8** |
| IQ2_XXS | 336 MB | 3.63 | **65.6** |

**Conclusão:** Para modelos <1B parâmetros, abaixo de 4 BPW a qualidade colapsa. IQ2 só é viável em modelos ≥7B onde a redundância de parâmetros absorve a quantização.

---

## 3. 🔄 Vocab Pruning (152K → 32K) — Em Progresso

**Objetivo:** Reduzir vocabulário de 152K para 32K tokens (OBS: Qwen3.5-0.8B tem 152064 tokens, nao 248K), diminuindo embedding de 254M para 32M parâmetros (-29.4% do modelo total).

**Progresso:**
- ✅ Token frequencies analisadas no dataset PT-BR (8.5K tokens únicos)
- ✅ Mapping old->new criado (32K tokens mantidos)
- ✅ Modelo podado: 752M → 531M parâmetros (embeddings reduzidas)
- ✅ Tokenizer BPE 10K treinado, embeddings alinhados (61% match)
- ✅ Fine-tune de adaptação (3 epochs): eval_loss 2.8, PPL 50 vs 9.4 baseline
- ❌ Qualidade INADEQUADA para produção (PPL 5x pior que baseline)
- ❌ Dataset de 2K exemplos insuficiente para adaptar tokenizer novo

**Próximos passos para v2:**
1. Treinar tokenizer BPE novo (32K) no corpus PT-BR com `tokenizers` library
2. Alinhar embeddings do modelo podado com novo tokenizer via "embedding initialization"
3. Fine-tune de adaptação por 3-5 épocas
4. Converter para GGUF e quantizar Q4_K_M

**Ganho estimado:** +16-21% velocidade (→23 tok/s no A54), modelo ~380 MB.

---

## 4. ✅ K/V Cache Quantization

**Otimização:** `-ctk q8_0 -ctv q8_0` no llama-server

**Efeito:** Cache de atenção quantizado em 8-bit, liberando ~300 MB de RAM. Com `--mlock` para evitar swap.

**Aplicado em:** `scripts/deploy.sh`

---

## Resumo

| Otimização | Status | Ganho |
|---|---|---|
| Thinking fix | ✅ Pronto | -20% tokens |
| IQ2_XXS | ❌ Rejeitado | Qualidade colapsa |
| Vocab pruning | ❌ Adiado v2 | PPL 50, precisa 10K+ exemplos |
| K/V cache q8_0 | ✅ Pronto | -300 MB RAM |
| ARM64 -O3 + DotProd | ✅ Pronto | +10-15% |

## 5. ✅ Codex Review Fixes (Jun 2026)

Correções aplicadas após revisão crítica:

| Fix | Arquivo | Descrição |
|---|---|---|
| Server readiness | `MainActivity.kt` | Health check com exponential backoff (500ms→5s, 30 tentativas) |
| Server readiness | `LlamaServerService.kt` | Broadcast de status (loading/ready/error) para Activity |
| Wakelock | `LlamaServerService.kt` | `PARTIAL_WAKE_LOCK` (24h) para servidor sobreviver com tela bloqueada |
| Battery exemption | `MainActivity.kt` | `REQUEST_IGNORE_BATTERY_OPTIMIZATIONS` flow |
| Model download | `MainActivity.kt` | `DownloadManager` integrado (HuggingFace → Download/ dir) |
| Context truncation | `ChatScreen.kt` | `prepareContext()`: mantém últimos ~1500 tokens, descarta resto |
| Session cache | `ChatScreen.kt` | `id_session` enviado na API para reuso de KV cache no servidor |
| Home press fix | `MainActivity.kt` | Servidor NÃO para no `onStop()`, só com botão explícito "Parar servidor" |
| Auto-restart | `LlamaServerService.kt` | Watchdog com exponential backoff (1s→2s→4s→...max 30s) |
| Double-start guard | `LlamaServerService.kt` | `serverReady` flag previne porta duplicada no `START_STICKY` |
| deploy.sh fixes | `scripts/deploy.sh` | Shebang `#!/bin/bash`, `SCRIPT_DIR` resolve paths, `$MODEL_DIR` criado, health via `/proc/net/tcp` |
| CPU affinity (-t 3) | ✅ Pronto | +13% |
