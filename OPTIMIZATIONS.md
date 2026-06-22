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

**Objetivo:** Reduzir vocabulário de 248K para 32K tokens, diminuindo embedding de 254M para 32M parâmetros (-29.4% do modelo total).

**Progresso:**
- ✅ Token frequencies analisadas no dataset PT-BR (8.5K tokens únicos)
- ✅ Mapping old→new criado (32K tokens mantidos)
- ✅ Modelo podado: 752M → 531M parâmetros (embeddings reduzidas)
- ❌ Tokenizer BPE quebrado — podar merges BPE é complexo:
  - 943 tokens inalcançáveis (não decomponíveis pelos merges restantes)
  - 68 bytes faltando (0x00-0x13 e alguns bytes altos)
  - BPE tokenizer requer reconstrução completa

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
| Vocab pruning | 🔄 Adiado v2 | +16-21% velocidade |
| K/V cache q8_0 | ✅ Pronto | -300 MB RAM |
| ARM64 -O3 + DotProd | ✅ Pronto | +10-15% |
| CPU affinity (-t 3) | ✅ Pronto | +13% |
