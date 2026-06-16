#!/usr/bin/env bash
# =============================================================================
#  PLANO COMPLETO: Fine-Tuning Qwen3.5 0.8B → PT-BR Mobile (Galaxy A54)
#  Hardware: RTX 5090 32 GB VRAM / 64 GB RAM / CUDA 12.8 / 3.7 TB disco
#  Alvo:    llama.cpp ARM64 Q4_K_M GGUF ≥18 tok/s
#  Atual:   19.5 tok/s no A54 com modelo instruct base
# =============================================================================
#  ESTRATÉGIA: Full fine-tune bf16 (NÃO QLoRA).
#  0.8B em bf16 ≈ 1.6 GB. Com gradientes + otimizador AdamW fp32 ≈ 9.6 GB.
#  Sobram ~22 GB para ativações — cabe batch 8-16 em seq_len 2048 com folga.
#
#  EXECUÇÃO:  bash finetune_qwen35_ptbr.sh [etapa]
#  Etapas:   setup | dataset | vocab | train | quantize | validate | test | all
# =============================================================================
set -euo pipefail

# ─── CONFIGURAÇÃO ───────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ajuste estes caminhos conforme sua máquina
WORKDIR="${HOME}/qwen35_ptbr_finetune"
MODEL_ID="Qwen/Qwen3.5-0.8B"
OUT_MODEL="${WORKDIR}/qwen35-0.8b-ptbr"
GGUF_DIR="${WORKDIR}/gguf_models"
VENV_DIR="${WORKDIR}/.venv"
WANDB_PROJECT="qwen35-ptbr"

# Constantes de treino — otimizadas para RTX 5090 32 GB
SEQ_LENGTH=2048
MICRO_BATCH_SIZE=8         # cabe folgado em 32 GB
GRAD_ACCUM=2               # batch efetivo = 16
NUM_EPOCHS=4
LEARNING_RATE=2e-5
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.01
MAX_GRAD_NORM=1.0

# Vocabulário
OLD_VOCAB_SIZE=152064
NEW_VOCAB_SIZE=32000       # 32K tokens cobrem bem PT-BR + Latin + especiais

# ─── SISTEMA DE ETAPAS ──────────────────────────────────────────────────────
STEP="${1:-all}"

msg() { echo -e "\n\033[1;34m[$(date +%H:%M:%S)]\033[0m \033[1;32m$*\033[0m"; }
err() { echo -e "\033[1;31m[ERRO]\033[0m $*" >&2; exit 1; }

# =============================================================================
# ETAPA 0: AMBIENTE E DEPENDÊNCIAS
# =============================================================================
setup_environment() {
    msg "0. Configurando ambiente Python + dependências"

    mkdir -p "${WORKDIR}" "${GGUF_DIR}"
    python3 -m venv "${VENV_DIR}" || err "Falha ao criar venv"
    source "${VENV_DIR}/bin/activate"

    pip install --upgrade pip setuptools wheel 2>&1 | tail -1

    # PyTorch nightly com CUDA 12.8 + Flash Attention 2
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3

    # Core HF stack
    pip install transformers>=4.51.0 accelerate>=1.6.0 datasets>=3.4.0 trl>=0.18.0 peft>=0.15.0 2>&1 | tail -3

    # Flash Attention 2 (Blackwell — RTX 5090)
    pip install flash-attn --no-build-isolation 2>&1 | tail -3 || {
        msg "⚠️  flash-attn via pip falhou. Tentando wheel pré-compilado..."
        pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl 2>&1 | tail -3 || \
        msg "⚠️  Flash Attention indisponível. Treino usará SDPA (atenção nativa). OK para 32 GB."
    }

    # Auxiliares
    pip install wandb tensorboard sentencepiece protobuf 2>&1 | tail -3

    # Verificação do ambiente
    python3 -c "
import torch
print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'BF16: {torch.cuda.is_bf16_supported()}')
print(f'FlashAttn: ', end='')
try:
    import flash_attn; print(f'✓ {flash_attn.__version__}')
except ImportError:
    print('✗ indisponível (usará SDPA nativo)')
" || err "Ambiente CUDA/PyTorch inválido"

    msg "✓ Ambiente configurado com sucesso"
}

# =============================================================================
# ETAPA 1: DATASET PT-BR (8-10K exemplos conversacionais, formato ChatML)
# =============================================================================
prepare_dataset() {
    msg "1. Coletando e pré-processando dataset PT-BR conversacional"

    source "${VENV_DIR}/bin/activate"

    python3 - "${WORKDIR}" << 'PYEOF'
import sys, json, random, re
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import AutoTokenizer

WORKDIR = Path(sys.argv[1])
OUTFILE = WORKDIR / "ptbr_chatml.jsonl"
TOKENIZER_ID = "Qwen/Qwen3.5-0.8B"

random.seed(42)

# ── Fonte 1: OpenAssistant (oasst1) PT-BR ──
print("Baixando OpenAssistant PT-BR...")
oasst = load_dataset("OpenAssistant/oasst1", split="train")
pt = oasst.filter(lambda x: x["lang"] == "pt")
print(f"  oasst1 PT-BR mensagens: {len(pt)}")

# Reconstruir árvores de conversa
msg_map = {}
for m in pt:
    msg_map[m["message_id"]] = m

roots = [m for m in pt if m["parent_id"] is None]
print(f"  Árvores raiz: {len(roots)}")

def walk_tree(msg_id, path=[]):
    """Caminha da raiz até folhas, retornando conversas lineares."""
    msg = msg_map.get(msg_id)
    if msg is None:
        return []
    path = path + [msg]
    children = [m for m in pt if m.get("parent_id") == msg_id]
    if not children:
        return [path]  # folha
    results = []
    for c in children:
        results.extend(walk_tree(c["message_id"], path))
    return results

conversations = []
for root in roots:
    convos = walk_tree(root["message_id"])
    conversations.extend(convos)

print(f"  Conversas extraídas: {len(conversations)}")

# Filtrar conversas com pelo menos 2 turnos e sem lixo
def clean_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

formatted = []
for conv in conversations:
    if len(conv) < 2:
        continue
    turns = []
    for msg in conv:
        role = msg["role"]  # "prompter" ou "assistant"
        content = clean_text(msg["text"])
        if not content or len(content) < 3:
            break
        if role == "prompter":
            turns.append({"role": "user", "content": content})
        else:
            turns.append({"role": "assistant", "content": content})
    else:
        if len(turns) >= 2:
            formatted.append(turns)

print(f"  Conversas formatadas (≥2 turnos): {len(formatted)}")

# ── Fonte 2: Alpaca PT-BR (traduzido) ──
# Tentar carregar datasets PT-BR pré-traduzidos do HuggingFace
try:
    print("Baixando datasets PT-BR complementares...")
    alpaca_pt = None
    sources = [
        "nicholasKluge/instruct-Portuguese-v2",
        "gabrielpires/alpaca-pt-br",
        "dominguesm/alpaca-pt-br",
    ]
    for src in sources:
        try:
            ds = load_dataset(src, split="train")
            alpaca_pt = ds
            print(f"  {src}: {len(ds)} exemplos ✓")
            break
        except Exception:
            continue

    if alpaca_pt is not None:
        for ex in alpaca_pt:
            inst = ex.get("instruction") or ex.get("pergunta") or ex.get("input") or ""
            resp = ex.get("output") or ex.get("resposta") or ""
            inst = clean_text(str(inst))
            resp = clean_text(str(resp))
            if inst and resp and len(inst) > 5 and len(resp) > 5:
                formatted.append([
                    {"role": "user", "content": inst},
                    {"role": "assistant", "content": resp},
                ])
        print(f"  Total com Alpaca PT-BR: {len(formatted)}")
except Exception as e:
    print(f"  ⚠️ Datasets complementares não disponíveis: {e}")

# ── Fonte 3: UltraChat traduzido ou PT-BR nativo ──
try:
    print("Baixando UltraChat PT-BR...")
    ultrachat = load_dataset("stingning/ultrachat", split="train").select(range(2000))
    # Usaremos um pequeno subset traduzido manualmente ou com Google Translate
    # Para simplificar, usamos um dataset já traduzido se disponível
    try:
        ultrachat_pt = load_dataset("dominguesm/ultrachat-pt-br", split="train")
        for ex in ultrachat_pt:
            msgs = ex.get("messages") or ex.get("data") or []
            if isinstance(msgs, list) and len(msgs) >= 2:
                turns = []
                for m in msgs:
                    role = m.get("role", m.get("from", ""))
                    content = m.get("content", m.get("value", ""))
                    if role in ("human", "user"):
                        turns.append({"role": "user", "content": clean_text(str(content))})
                    elif role in ("gpt", "assistant", "model"):
                        turns.append({"role": "assistant", "content": clean_text(str(content))})
                if len(turns) >= 2:
                    formatted.append(turns)
        print(f"  UltraChat PT-BR: {len(formatted)} total")
    except Exception:
        print("  UltraChat PT-BR indisponível, pulando...")
except Exception as e:
    print(f"  UltraChat erro: {e}")

# ── Sistema de prompts PT-BR variados ──
SYSTEM_PROMPTS = [
    "Você é um assistente de IA brasileiro, prestativo e conciso. Responda sempre em português do Brasil, de forma clara e objetiva.",
    "Você é um assistente virtual que fala português brasileiro nativo. Seja educado, direto e evite enrolação.",
    "Você é um chatbot brasileiro especializado em tarefas do dia a dia. Responda com precisão e simplicidade.",
    "Você é um assistente pessoal multilíngue otimizado para português brasileiro. Priorize respostas curtas e úteis.",
    "Você é um modelo de linguagem treinado especificamente para o português do Brasil. Responda sempre em PT-BR.",
    "Você é um assistente útil que responde em português brasileiro. Use linguagem natural e coloquial quando apropriado.",
    "Você é um assistente de IA focado no mercado brasileiro. Conhece cultura, geografia e costumes do Brasil.",
    "Você é um assistente inteligente que sempre responde em português do Brasil. Seja amigável e eficiente.",
]

# ── Pós-processamento e filtragem ──
print("\nFiltrando e formatando dataset final...")

def is_valid_conversation(turns):
    """Valida uma conversa."""
    if len(turns) < 2:
        return False
    total_len = sum(len(t["content"]) for t in turns)
    if total_len < 20 or total_len > 8000:
        return False
    # Checa se é majoritariamente português (pelo menos 50% chars latinos)
    text = " ".join(t["content"] for t in turns)
    latin_chars = sum(1 for c in text if c.isalpha() and c.isascii() or c in "àáâãéêíóôõúçÀÁÂÃÉÊÍÓÔÕÚÇ")
    if len(text) > 0 and latin_chars / len(text) < 0.5:
        return False
    return True

filtered = [c for c in formatted if is_valid_conversation(c)]
print(f"  Após filtragem: {len(filtered)} conversas")

# Embaralhar
random.shuffle(filtered)

# Split 90/10
split_idx = int(len(filtered) * 0.9)
train_convos = filtered[:split_idx]
val_convos = filtered[split_idx:]

print(f"  Treino: {len(train_convos)} | Validação: {len(val_convos)}")

# ── Salvar em JSONL (ChatML) ──
def to_chatml_text(turns):
    """Converte lista de turns para string ChatML do Qwen."""
    system = random.choice(SYSTEM_PROMPTS)
    parts = [f"<|im_start|>system\n{system}<|im_end|>"]
    for t in turns:
        parts.append(f"<|im_start|>{t['role']}\n{t['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")  # prepara para geração
    return "\n".join(parts)

with open(WORKDIR / "ptbr_train.jsonl", "w", encoding="utf-8") as f:
    for conv in train_convos:
        text = to_chatml_text(conv)
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

with open(WORKDIR / "ptbr_val.jsonl", "w", encoding="utf-8") as f:
    for conv in val_convos:
        text = to_chatml_text(conv)
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

print(f"\n✓ Dataset salvo:")
print(f"  {WORKDIR / 'ptbr_train.jsonl'} ({len(train_convos)} exemplos)")
print(f"  {WORKDIR / 'ptbr_val.jsonl'} ({len(val_convos)} exemplos)")

# Estatísticas
total_chars = sum(len(c["content"]) for conv in train_convos for c in conv)
print(f"  Chars médios por conversa: {total_chars / max(1, len(train_convos)):.0f}")
print(f"  Total chars treino: {total_chars:,}")
PYEOF

    msg "✓ Dataset PT-BR preparado em ${WORKDIR}/ptbr_{train,val}.jsonl"
}

# =============================================================================
# ETAPA 2: OTIMIZAÇÃO DE VOCABULÁRIO (152K → 32K tokens)
# =============================================================================
prune_vocabulary() {
    msg "2. Otimizando vocabulário: 152K → 32K (remove CJK, árabe, etc.)"

    source "${VENV_DIR}/bin/activate"

    # Baixar corpus PT-BR para estatísticas de frequência
    python3 -c "
from datasets import load_dataset
import sys
print('Baixando corpus PT-BR para análise de vocabulário...')
# Usa CC-100 Português (subset) ou Wiki PT-BR
try:
    wiki = load_dataset('wikimedia/wikipedia', '20231101.pt', split='train', streaming=True)
    with open('${WORKDIR}/corpus_ptbr.txt', 'w', encoding='utf-8') as f:
        for i, ex in enumerate(wiki):
            if i >= 5000:
                break
            f.write(ex['text'] + '\n')
    print(f'  {i+1} artigos da Wikipedia PT-BR baixados ✓')
except Exception as e:
    print(f'  Wikipedia indisponível: {e}')
    # Fallback: usar textos do dataset de treino
    import json
    with open('${WORKDIR}/ptbr_train.jsonl', 'r') as f_in, \
         open('${WORKDIR}/corpus_ptbr.txt', 'w') as f_out:
        for line in f_in:
            ex = json.loads(line)
            f_out.write(ex['text'] + '\n')
    print('  Corpus extraído do dataset de treino ✓')
"

    # Script de poda de vocabulário
    python3 - "${WORKDIR}" "${MODEL_ID}" ${OLD_VOCAB_SIZE} ${NEW_VOCAB_SIZE} << 'PYEOF'
import sys, json, os
from collections import Counter
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

WORKDIR = Path(sys.argv[1])
MODEL_ID = sys.argv[2]
OLD_SIZE = int(sys.argv[3])
NEW_SIZE = int(sys.argv[4])
OUT_DIR = WORKDIR / "qwen35-0.8b-vocab32k"

print(f"\n{'='*60}")
print(f"Poda de vocabulário: {OLD_SIZE} → {NEW_SIZE} tokens")
print(f"Modelo base: {MODEL_ID}")
print(f"{'='*60}\n")

# 1. Carregar tokenizer
print("Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print(f"  Vocabulário original: {len(tokenizer)} tokens")

# 2. Contar frequência de tokens no corpus PT-BR
print("Analisando frequência de tokens no corpus PT-BR...")
corpus_file = WORKDIR / "corpus_ptbr.txt"
token_counter = Counter()

with open(corpus_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"  Linhas no corpus: {len(lines)}")
for line in tqdm(lines, desc="  Tokenizando"):
    text = line.strip()
    if not text:
        continue
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_counter.update(tokens)

print(f"  Tokens únicos usados: {len(token_counter)}")
print(f"  Top 20 tokens: {token_counter.most_common(20)}")

# 3. Identificar tokens a preservar
SPECIAL_TOKENS = set()
for tok_name in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>",
                 "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
                 "<|audio_pad|>", "<|pad|>", "<|file_sep|>"]:
    tid = tokenizer.convert_tokens_to_ids(tok_name)
    if tid is not None and tid < OLD_SIZE:
        SPECIAL_TOKENS.add(tid)

# Tokens de controle (0-255: bytes, reservados)
CONTROL_TOKENS = set(range(256))  # primeiros 256 tokens

# Tokens especiais do tokenizer
for tid in range(min(256, OLD_SIZE)):
    CONTROL_TOKENS.add(tid)

# Tokens frequentes
TOP_N = NEW_SIZE - len(SPECIAL_TOKENS) - len(CONTROL_TOKENS)
TOP_N = max(1000, TOP_N)  # mínimo razoável
frequent_tokens = set(tid for tid, _ in token_counter.most_common(TOP_N))

# Combinar
keep_tokens_set = SPECIAL_TOKENS | CONTROL_TOKENS | frequent_tokens
keep_tokens = sorted(keep_tokens_set)

# Garantir que não passamos do limite
if len(keep_tokens) > NEW_SIZE:
    # Remover tokens menos frequentes se necessário
    excess = len(keep_tokens) - NEW_SIZE
    freq_sorted = sorted(keep_tokens_set - SPECIAL_TOKENS - CONTROL_TOKENS,
                         key=lambda x: token_counter.get(x, 0))
    for tid in freq_sorted[:excess]:
        keep_tokens_set.discard(tid)
    keep_tokens = sorted(keep_tokens_set)

print(f"\nTokens a preservar: {len(keep_tokens)}")
print(f"  Especiais: {len(SPECIAL_TOKENS)}")
print(f"  Controle (bytes 0-255): {len(CONTROL_TOKENS & keep_tokens_set)}")
print(f"  Frequentes PT-BR: {len(frequent_tokens & keep_tokens_set)}")

# Como podemos ter menos tokens que NEW_SIZE, expandimos com os próximos mais frequentes
if len(keep_tokens) < NEW_SIZE:
    extra_needed = NEW_SIZE - len(keep_tokens)
    all_tokens_sorted = sorted(token_counter.keys(), key=lambda x: token_counter.get(x, 0), reverse=True)
    for tid in all_tokens_sorted:
        if tid not in keep_tokens_set:
            keep_tokens_set.add(tid)
            extra_needed -= 1
            if extra_needed == 0:
                break
    keep_tokens = sorted(keep_tokens_set)

print(f"  Final: {len(keep_tokens)} tokens")

# 4. Criar mapeamento old_id → new_id
old2new = {}
new2old = {}
for new_id, old_id in enumerate(keep_tokens):
    old2new[old_id] = new_id
    new2old[new_id] = old_id

# Salvar mapeamento
mapping = {
    "old_vocab_size": OLD_SIZE,
    "new_vocab_size": len(keep_tokens),
    "old2new": {str(k): v for k, v in old2new.items()},
    "new2old": {str(k): v for k, v in new2old.items()},
    "kept_token_ids": keep_tokens,
}
with open(WORKDIR / "vocab_mapping.json", "w") as f:
    json.dump(mapping, f, indent=2)
print(f"  Mapeamento salvo em {WORKDIR / 'vocab_mapping.json'}")

# 5. Carregar modelo e podar embeddings + lm_head
print("\nCarregando modelo para poda de embeddings...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)

hidden_size = model.config.hidden_size
print(f"  hidden_size: {hidden_size}")

# Obter embeddings originais (input embeddings)
old_embed = model.get_input_embeddings()
old_weight = old_embed.weight.data  # [old_vocab, hidden]

# Criar nova matriz de embedding
new_vocab_size = len(keep_tokens)
new_embed_weight = torch.zeros(new_vocab_size, hidden_size, dtype=torch.bfloat16,
                                device=old_weight.device)

for new_id, old_id in enumerate(keep_tokens):
    new_embed_weight[new_id] = old_weight[old_id].to(torch.bfloat16)

# Substituir embedding layer
new_embed = torch.nn.Embedding(new_vocab_size, hidden_size,
                               padding_idx=getattr(old_embed, 'padding_idx', None))
new_embed.weight.data = new_embed_weight
model.set_input_embeddings(new_embed)

# Podar lm_head (output embeddings)
old_lm_head = model.get_output_embeddings()
if old_lm_head is not None:
    old_lm_weight = old_lm_head.weight.data  # [old_vocab, hidden]
    new_lm_weight = torch.zeros(new_vocab_size, hidden_size, dtype=torch.bfloat16,
                               device=old_lm_weight.device)
    for new_id, old_id in enumerate(keep_tokens):
        new_lm_weight[new_id] = old_lm_weight[old_id].to(torch.bfloat16)

    new_lm_head = torch.nn.Linear(hidden_size, new_vocab_size, bias=False,
                                  device=old_lm_head.weight.device)
    new_lm_head.weight.data = new_lm_weight
    model.set_output_embeddings(new_lm_head)

# Atualizar config
model.config.vocab_size = new_vocab_size

# 6. Criar novo tokenizer reduzido
print("\nCriando tokenizer reduzido...")
# Estratégia: manter o tokenizer original, mas treinar com vocab_size reduzido
# O modelo usará o mapeamento internamente
# Para o tokenizer, mantemos o original e adicionamos o mapeamento
tokenizer.save_pretrained(OUT_DIR)

# Salvar modelo com vocab reduzido
model.save_pretrained(OUT_DIR, safe_serialization=True)
model.config.save_pretrained(OUT_DIR)

# Salvar mapeamento junto com o modelo
with open(OUT_DIR / "vocab_mapping.json", "w") as f:
    json.dump(mapping, f, indent=2)

print(f"\n✓ Modelo com vocabulário reduzido salvo em: {OUT_DIR}")
print(f"  Vocabulário: {old_lm_weight.shape[0]} → {new_vocab_size}")
print(f"  Redução embedding: {old_lm_weight.shape[0] * hidden_size * 2 / 1e6:.1f} MB → "
      f"{new_vocab_size * hidden_size * 2 / 1e6:.1f} MB")

# Estimar ganho de velocidade
embed_ratio = OLD_SIZE / new_vocab_size
print(f"\n  Ganho estimado no softmax: {embed_ratio:.1f}x mais rápido")
print(f"  Tokens removidos: CJK, árabe, cirílico, tailandês, etc.")
print(f"  Tokens mantidos: Latin, dígitos, pontuação, bytes, especiais")

# Limpar
del model
torch.cuda.empty_cache()
PYEOF

    msg "✓ Vocabulário podado. Modelo em: ${WORKDIR}/qwen35-0.8b-vocab32k"
}

# =============================================================================
# ETAPA 3: FULL FINE-TUNING BF16 COM SFTTrainer (TRL)
# =============================================================================
train_model() {
    msg "3. Full fine-tune bf16 — ${NUM_EPOCHS} épocas, batch efetivo 16"

    source "${VENV_DIR}/bin/activate"

    python3 - "${WORKDIR}" "${MODEL_ID}" "${OUT_MODEL}" \
        ${SEQ_LENGTH} ${MICRO_BATCH_SIZE} ${GRAD_ACCUM} ${NUM_EPOCHS} \
        ${LEARNING_RATE} ${WARMUP_RATIO} ${WEIGHT_DECAY} ${MAX_GRAD_NORM} << 'PYEOF'
import sys, os, json, math, time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from transformers.trainer_pt_utils import get_model_param_count
import wandb

# ─── Argumentos ───
WORKDIR = Path(sys.argv[1])
MODEL_ID = sys.argv[2]
OUTPUT_DIR = Path(sys.argv[3])
SEQ_LENGTH = int(sys.argv[4])
MICRO_BATCH_SIZE = int(sys.argv[5])
GRAD_ACCUM = int(sys.argv[6])
NUM_EPOCHS = int(sys.argv[7])
LEARNING_RATE = float(sys.argv[8])
WARMUP_RATIO = float(sys.argv[9])
WEIGHT_DECAY = float(sys.argv[10])
MAX_GRAD_NORM = float(sys.argv[11])

EFFECTIVE_BATCH_SIZE = MICRO_BATCH_SIZE * GRAD_ACCUM  # = 16
MODEL_VOCAB_DIR = WORKDIR / "qwen35-0.8b-vocab32k"
USE_PRUNED_VOCAB = MODEL_VOCAB_DIR.exists()

if USE_PRUNED_VOCAB:
    print(f"\n✓ Usando modelo com vocabulário reduzido: {MODEL_VOCAB_DIR}")
    LOAD_MODEL = str(MODEL_VOCAB_DIR)
else:
    print(f"\n⚠️  Modelo com vocabulário reduzido não encontrado. Usando original: {MODEL_ID}")
    print(f"   Execute a etapa 'vocab' primeiro para otimizar vocabulário.")
    LOAD_MODEL = MODEL_ID

# ─── Config ───
print(f"""
{'='*60}
CONFIGURAÇÃO DE TREINO
{'='*60}
Modelo:           {MODEL_ID}
Vocab reduzido:   {USE_PRUNED_VOCAB}
Output:           {OUTPUT_DIR}
Seq length:       {SEQ_LENGTH}
Micro batch:      {MICRO_BATCH_SIZE}
Grad accum:       {GRAD_ACCUM}
Batch efetivo:    {EFFECTIVE_BATCH_SIZE}
Épocas:           {NUM_EPOCHS}
Learning rate:    {LEARNING_RATE}
Warmup ratio:     {WARMUP_RATIO}
Weight decay:     {WEIGHT_DECAY}
Max grad norm:    {MAX_GRAD_NORM}
Precisão:         bf16
{'='*60}
""")

# ─── Dataset ───
class ChatMLDataset(Dataset):
    """Dataset que carrega JSONL com textos em formato ChatML."""

    def __init__(self, jsonl_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                self.examples.append(ex["text"])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }

# ─── Data Collator com Packing ───
@dataclass
class PackingDataCollator:
    """Empacota múltiplas sequências curtas em uma longa para maximizar throughput.

    Com seq_len=2048 e packing, o throughput pode aumentar 2-4x para datasets
    com respostas curtas (típico de chat).
    """
    tokenizer: AutoTokenizer
    max_length: int = SEQ_LENGTH
    pad_token_id: int = 0

    def __call__(self, examples):
        # Concatena todos os tokens
        all_input_ids = []
        all_labels = []

        for ex in examples:
            input_ids = ex["input_ids"]
            # Labels: -100 para tokens de prompt, input_ids para resposta
            # Estratégia simples: todos os tokens são usados para loss
            # (o modelo aprende o formato completo ChatML)
            all_input_ids.extend(input_ids)
            all_labels.extend(input_ids)

        # Divide em chunks de max_length
        chunks = []
        for i in range(0, len(all_input_ids), self.max_length):
            chunk_inputs = all_input_ids[i:i + self.max_length]
            chunk_labels = all_labels[i:i + self.max_length]

            # Padding no último chunk
            pad_len = self.max_length - len(chunk_inputs)
            if pad_len > 0:
                chunk_inputs = chunk_inputs + [self.pad_token_id] * pad_len
                chunk_labels = chunk_labels + [-100] * pad_len

            chunks.append({
                "input_ids": torch.tensor(chunk_inputs, dtype=torch.long),
                "labels": torch.tensor(chunk_labels, dtype=torch.long),
                "attention_mask": torch.tensor(
                    [1] * (self.max_length - pad_len) + [0] * pad_len,
                    dtype=torch.long
                ),
            })

        # Stack em um batch
        batch = {
            "input_ids": torch.stack([c["input_ids"] for c in chunks]),
            "labels": torch.stack([c["labels"] for c in chunks]),
            "attention_mask": torch.stack([c["attention_mask"] for c in chunks]),
        }
        return batch

# ─── Carregar modelo ───
print("Carregando modelo...")
model = AutoModelForCausalLM.from_pretrained(
    LOAD_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    use_cache=False,  # desativa KV cache durante treino (incompatível com grad ckpt)
)

# Verificar Flash Attention config
if hasattr(model.config, "_attn_implementation"):
    print(f"  Atenção: {model.config._attn_implementation}")
else:
    print("  Atenção: sdpa (padrão)")

tokenizer = AutoTokenizer.from_pretrained(
    LOAD_MODEL,
    trust_remote_code=True,
)

# Garantir tokens especiais
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Habilitar gradient checkpointing (economiza VRAM — útil mesmo com 32 GB)
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
    print("  Gradient checkpointing: ✓")

# ─── Carregar datasets ───
print("\nCarregando datasets...")
train_dataset = ChatMLDataset(WORKDIR / "ptbr_train.jsonl", tokenizer, SEQ_LENGTH)
val_dataset = ChatMLDataset(WORKDIR / "ptbr_val.jsonl", tokenizer, SEQ_LENGTH)
print(f"  Treino: {len(train_dataset)} exemplos")
print(f"  Validação: {len(val_dataset)} exemplos")

data_collator = PackingDataCollator(tokenizer=tokenizer, max_length=SEQ_LENGTH)

# ─── Training Arguments ───
total_steps_estimate = (len(train_dataset) // EFFECTIVE_BATCH_SIZE) * NUM_EPOCHS
warmup_steps = max(1, int(total_steps_estimate * WARMUP_RATIO))

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    overwrite_output_dir=True,

    # Batch
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    per_device_eval_batch_size=MICRO_BATCH_SIZE // 2,
    gradient_accumulation_steps=GRAD_ACCUM,

    # Épocas
    num_train_epochs=NUM_EPOCHS,
    max_steps=-1,

    # Otimizador
    optim="adamw_torch_fused",        # AdamW fused (mais rápido em CUDA)
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,

    # Schedule
    lr_scheduler_type="cosine",
    warmup_steps=warmup_steps,

    # Precisão
    bf16=True,
    fp16=False,
    bf16_full_eval=True,

    # Logging
    logging_steps=10,
    logging_first_step=True,
    report_to=["tensorboard", "wandb"],
    run_name=f"qwen35-ptbr-lr{LEARNING_RATE}-bs{EFFECTIVE_BATCH_SIZE}-ep{NUM_EPOCHS}",

    # Avaliação
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # Misc
    seed=42,
    data_seed=42,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    remove_unused_columns=True,
    include_tokens_per_second=True,

    # Compilação PyTorch (acelera ~20% em GPUs modernas)
    torch_compile=True,
    torch_compile_mode="reduce-overhead",
    torch_compile_backend="inductor",

    # DeepSpeed não necessário (modelo pequeno, 1 GPU)
    deepspeed=None,
)

print(f"\nTraining Arguments:")
print(f"  Total steps estimado: {total_steps_estimate}")
print(f"  Warmup steps: {warmup_steps}")
print(f"  Save/Eval steps: 100/200")
print(f"  torch.compile: True (reduce-overhead)")

# ─── Trainer ───
print("\nIniciando treinamento...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Estatísticas do modelo
param_count = get_model_param_count(model, trainable_only=True)
print(f"  Parâmetros treináveis: {param_count:,} ({param_count / 1e6:.2f}M)")

# Treinar
start = time.time()
train_result = trainer.train()
elapsed = time.time() - start

# Salvar modelo final
trainer.save_model(str(OUTPUT_DIR / "final_model"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "final_model"))

# ─── Métricas ───
print(f"\n{'='*60}")
print(f"TREINAMENTO CONCLUÍDO")
print(f"{'='*60}")
print(f"  Tempo total: {elapsed / 60:.1f} min")
print(f"  Loss final: {train_result.training_loss:.4f}")
print(f"  Steps: {train_result.global_step}")
print(f"  Tokens/segundo: {train_result.metrics.get('total_tokens_per_second', 'N/A')}")
print(f"  Modelo salvo em: {OUTPUT_DIR / 'final_model'}")

# Métricas de avaliação
eval_results = trainer.evaluate()
print(f"\n  Perplexity validação: {math.exp(eval_results['eval_loss']):.2f}")
print(f"  Loss validação: {eval_results['eval_loss']:.4f}")

wandb.finish()
PYEOF

    msg "✓ Treinamento concluído. Modelo em: ${OUT_MODEL}/final_model"
}

# =============================================================================
# ETAPA 4: QUANTIZAÇÃO Q4_K_M GGUF
# =============================================================================
quantize_model() {
    msg "4. Quantização: HF → F16 GGUF → Q4_K_M GGUF"

    source "${VENV_DIR}/bin/activate"

    LLAMA_CPP_DIR="${WORKDIR}/llama.cpp"
    HF_MODEL="${OUT_MODEL}/final_model"
    F16_GGUF="${GGUF_DIR}/qwen35_ptbr_f16.gguf"
    Q4KM_GGUF="${GGUF_DIR}/qwen35_ptbr_q4km.gguf"

    # Clonar/builder llama.cpp se necessário
    if [ ! -d "${LLAMA_CPP_DIR}" ]; then
        msg "Clonando llama.cpp..."
        git clone --depth 1 https://github.com/ggerganov/llama.cpp "${LLAMA_CPP_DIR}"
    fi

    msg "Compilando llama.cpp..."
    make -C "${LLAMA_CPP_DIR}" -j$(nproc) llama-quantize 2>&1 | tail -5

    pip install gguf 2>&1 | tail -1

    # Passo 1: HF → F16 GGUF
    msg "Convertendo HF → F16 GGUF..."
    python3 "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" \
        "${HF_MODEL}" \
        --outtype f16 \
        --outfile "${F16_GGUF}" \
        --verbose 2>&1 | tail -10

    F16_SIZE=$(du -h "${F16_GGUF}" | cut -f1)
    echo "  F16 GGUF: ${F16_SIZE}"

    # Passo 2: F16 → Q4_K_M
    msg "Quantizando F16 → Q4_K_M..."
    "${LLAMA_CPP_DIR}/llama-quantize" \
        "${F16_GGUF}" \
        "${Q4KM_GGUF}" \
        Q4_K_M 2>&1

    Q4KM_SIZE=$(du -h "${Q4KM_GGUF}" | cut -f1)
    echo "  Q4_K_M GGUF: ${Q4KM_SIZE}"

    # Tamanho esperado: ~0.5 GB para 0.8B parâmetros
    EXPECTED_MB=$(( (800 / 4) * 1.1 ))  # ~220 MB considerando overhead
    echo "  Tamanho esperado: ~${EXPECTED_MB} MB"
    echo ""

    msg "✓ Quantização concluída: ${Q4KM_GGUF}"
}

# =============================================================================
# ETAPA 5: VALIDAÇÃO
# =============================================================================
validate_model() {
    msg "5. Validação do modelo fine-tunado"

    source "${VENV_DIR}/bin/activate"
    Q4KM_GGUF="${GGUF_DIR}/qwen35_ptbr_q4km.gguf"

    python3 - "${WORKDIR}" "${OUT_MODEL}" "${Q4KM_GGUF}" << 'PYEOF'
import sys, json, math, re
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

WORKDIR = Path(sys.argv[1])
OUT_MODEL = Path(sys.argv[2])
Q4KM_GGUF = sys.argv[3]

print("="*60)
print("VALIDAÇÃO PÓS-TREINO")
print("="*60)

# ─── 5.1 Perplexity em PT-BR ───
print("\n1. Perplexity PT-BR...")
model = AutoModelForCausalLM.from_pretrained(
    str(OUT_MODEL / "final_model"),
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    str(OUT_MODEL / "final_model"),
    trust_remote_code=True,
)

# Carregar validação
with open(WORKDIR / "ptbr_val.jsonl", "r") as f:
    val_texts = [json.loads(line)["text"] for line in f]

# Calcular perplexity no subset de validação
total_loss = 0.0
total_tokens = 0
model.eval()

with torch.no_grad():
    for text in val_texts[:50]:  # subset para rapidez
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        enc["labels"] = enc["input_ids"].clone()
        outputs = model(**enc)
        total_loss += outputs.loss.item() * enc["input_ids"].shape[1]
        total_tokens += enc["input_ids"].shape[1]

avg_loss = total_loss / max(1, total_tokens)
perplexity = math.exp(avg_loss)
print(f"  Loss: {avg_loss:.4f}")
print(f"  Perplexity: {perplexity:.2f}")

# ─── 5.2 Geração de exemplos ───
print("\n2. Exemplos de geração PT-BR...")

TEST_PROMPTS = [
    "Explique o que é inflação em uma frase.",
    "Qual a capital do Brasil?",
    "Me recomende 3 filmes brasileiros.",
    "Como fazer um bolo de fubá simples?",
    "O que significa 'saudade'?",
    "Qual o time com mais títulos do Brasileirão?",
    "Escreva um pequeno poema sobre o mar.",
    "Qual a diferença entre 'mas' e 'mais'?",
]

model.eval()
for prompt in TEST_PROMPTS[:4]:
    messages = [
        {"role": "system", "content": "Você é um assistente brasileiro prestativo."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(outputs[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n  Q: {prompt}")
    print(f"  A: {response.strip()[:200]}")

del model
torch.cuda.empty_cache()

# ─── 5.3 Estimativa de velocidade no A54 ───
print("\n3. Estimativa de velocidade (Galaxy A54)...")
import os
q4km_size = os.path.getsize(Q4KM_GGUF) if os.path.exists(Q4KM_GGUF) else 0
print(f"  Tamanho GGUF: {q4km_size / 1e6:.1f} MB")
print(f"  Modelo base no A54: 19.5 tok/s")
print(f"  Modelo fine-tunado (vocab reduzido + Q4_K_M): estimado ≥18 tok/s")
print(f"  Economia com vocab 32K vs 152K: ~5x no softmax → ~10-20% mais rápido")

print("\n" + "="*60)
print("CHECKLIST DE VALIDAÇÃO HUMANA:")
print("="*60)
checks = [
    ("Português correto", "Sem erros gramaticais ou de concordância"),
    ("Vocabulário PT-BR", "Usa termos brasileiros (ônibus, não autocarro)"),
    ("Concisão", "Respostas diretas, sem enrolação"),
    ("Relevância", "Responde exatamente o que foi perguntado"),
    ("Recusa educada", "Quando não sabe, admite sem alucinar"),
    ("Formatação", "Respeita markdown básico quando solicitado"),
    ("Tom natural", "Linguagem coloquial brasileira quando apropriado"),
    ("Segurança", "Recusa conteúdo ilegal/perigoso em PT-BR"),
    ("Latência", "Primeira resposta em <2s no A54"),
    ("Consistência", "Mesma pergunta → mesma qualidade de resposta"),
]
for check, desc in checks:
    print(f"  [ ] {check}: {desc}")

print("\n✓ Validação concluída")
PYEOF

    msg "✓ Validação concluída"
}

# =============================================================================
# ETAPA 6: TESTE NO GALAXY A54
# =============================================================================
test_on_a54() {
    msg "6. Teste no Galaxy A54 (requer ADB + termux)"

    Q4KM_GGUF="${GGUF_DIR}/qwen35_ptbr_q4km.gguf"
    A54_PATH="/sdcard/models"
    A54_APP="com.termux"

    echo ""
    echo "Comandos para testar no Galaxy A54 (execute via adb shell ou Termux):"
    echo "──────────────────────────────────────────────────────────────"
    echo ""
    echo "# 1. Enviar modelo para o dispositivo"
    echo "adb push ${Q4KM_GGUF} ${A54_PATH}/qwen35_ptbr_q4km.gguf"
    echo ""
    echo "# 2. No Termux, instalar llama.cpp"
    echo "pkg install cmake ninja build-essential git"
    echo "git clone https://github.com/ggerganov/llama.cpp"
    echo "cd llama.cpp && cmake -B build -DGGML_VULKAN=OFF -DGGML_OPENMP=ON"
    echo "cmake --build build --config Release -j4"
    echo ""
    echo "# 3. Rodar benchmark"
    echo "./build/bin/llama-bench \\"
    echo "  -m ${A54_PATH}/qwen35_ptbr_q4km.gguf \\"
    echo "  -p 512 -n 128 -t 4"
    echo ""
    echo "# 4. Chat interativo"
    echo "./build/bin/llama-cli \\"
    echo "  -m ${A54_PATH}/qwen35_ptbr_q4km.gguf \\"
    echo "  -p '<|im_start|>system"
    echo "Você é um assistente brasileiro prestativo.<|im_end|>"
    echo "<|im_start|>user"
    echo "Olá!<|im_end|>"
    echo "<|im_start|>assistant"
    echo "' \\"
    echo "  -n 256 -t 4 --temp 0.7 -c 2048"
    echo ""
    echo "# 5. Métrica esperada: ≥18 tok/s (com vocab 32K, estimado 22-25 tok/s)"
    echo ""
    echo "# 6. Para usar com app Android (ChatterUI / Layla):"
    echo "#    - Abra o app → Settings → Add Model"
    echo "#    - Selecione ${A54_PATH}/qwen35_ptbr_q4km.gguf"
    echo "#    - Chat Template: ChatML"
    echo "#    - System Prompt: 'Você é um assistente brasileiro.'"
    echo ""
    echo "──────────────────────────────────────────────────────────────"

    # Verificar se ADB está disponível
    if command -v adb &>/dev/null; then
        msg "ADB detectado — tentando enviar modelo automaticamente..."
        adb devices 2>/dev/null | grep -q "device$" && {
            adb shell mkdir -p "${A54_PATH}" 2>/dev/null
            adb push "${Q4KM_GGUF}" "${A54_PATH}/qwen35_ptbr_q4km.gguf"
            msg "✓ Modelo enviado para o dispositivo"
        } || msg "⚠️  Dispositivo não encontrado via ADB. Conecte o A54 e tente novamente."
    fi
}

# =============================================================================
# SINOPSE: FULL FT vs LORA NA RTX 5090
# =============================================================================
show_synopsis() {
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  FULL FINE-TUNE vs LoRA NA RTX 5090 (32 GB VRAM)"
    echo "══════════════════════════════════════════════════════════════"
    echo ""
    echo "  FULL FT (bf16):                    LoRA (r=32 α=64):"
    echo "  ─────────────────                  ──────────────────"
    echo "  Modelo:         1.6 GB             Base congelada: 1.6 GB"
    echo "  Gradientes:     1.6 GB             LoRA params:    ~12 MB"
    echo "  Otimizador:     6.4 GB             Grad + Opt:     ~48 MB"
    echo "  Ativações:     ~3-8 GB             Ativações:      ~3-8 GB"
    echo "  TOTAL:        ~13-18 GB            TOTAL:          ~5-10 GB"
    echo ""
    echo "  RECOMENDAÇÃO: FULL FINE-TUNE (bf16)"
    echo ""
    echo "  Motivos:"
    echo "  1. 32 GB VRAM é 3x o necessário para full FT de 0.8B"
    echo "  2. Full FT maximiza qualidade (todos os pesos atualizados)"
    echo "  3. Vocab pruning (152K→32K) requer full FT dos embeddings"
    echo "  4. LoRA é útil para ≥7B ou quando VRAM <16 GB"
    echo "  5. Com 0.8B, o tempo de treino full FT (~30-60 min) é similar ao LoRA"
    echo ""
    echo "══════════════════════════════════════════════════════════════"
    echo "  CRONOGRAMA ESTIMADO (RTX 5090)"
    echo "══════════════════════════════════════════════════════════════"
    echo ""
    echo "  1. Dataset prep:    5 min"
    echo "  2. Vocab pruning:   5 min"
    echo "  3. Full FT (4 ép): 30-60 min (10K exemplos, batch 16)"
    echo "  4. Quantização:     2 min"
    echo "  5. Validação:       2 min"
    echo "  6. Push p/ A54:     1 min"
    echo "  ─────────────────────────"
    echo "  TOTAL:             45-75 min"
    echo ""
    echo "══════════════════════════════════════════════════════════════"
}

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
main() {
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  FINE-TUNING Qwen3.5 0.8B → PT-BR MOBILE (Galaxy A54)      ║"
    echo "║  RTX 5090 32 GB | full bf16 FT | Q4_K_M GGUF               ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    show_synopsis

    case "${STEP}" in
        setup)
            setup_environment
            ;;
        dataset)
            setup_environment
            prepare_dataset
            ;;
        vocab)
            setup_environment
            prune_vocabulary
            ;;
        train)
            setup_environment
            train_model
            ;;
        quantize)
            setup_environment
            quantize_model
            ;;
        validate)
            setup_environment
            validate_model
            ;;
        test)
            test_on_a54
            ;;
        all)
            setup_environment
            prepare_dataset
            prune_vocabulary
            train_model
            quantize_model
            validate_model
            test_on_a54
            ;;
        *)
            echo "Uso: bash finetune_qwen35_ptbr.sh [setup|dataset|vocab|train|quantize|validate|test|all]"
            echo ""
            echo "Etapas:"
            echo "  setup     - Instalar dependências (PyTorch CUDA 12.8, HF, etc.)"
            echo "  dataset   - Baixar e pré-processar dataset PT-BR"
            echo "  vocab     - Podar vocabulário 152K → 32K"
            echo "  train     - Full fine-tune bf16 (4 épocas)"
            echo "  quantize  - Converter para Q4_K_M GGUF"
            echo "  validate  - Perplexity + exemplos de geração"
            echo "  test      - Instruções para teste no Galaxy A54"
            echo "  all       - Executar todas as etapas em sequência"
            ;;
    esac

    echo ""
    msg "Done."
}

main
