#!/bin/bash
# Qwen3.5 0.8B PT-BR — Deploy rápido no Android via ADB
#
# Uso (no PC, com celular conectado via USB):
#   bash deploy.sh
#
# Requisitos:
#   - ADB instalado e celular com depuração USB ativada
#   - Pelo menos 4 GB livres no armazenamento
#   - Android 14+ (testado no Galaxy A54, Android 16)
#
# O script:
#   1. Empurra os binários ARM64 otimizados (llama.cpp + libs)
#   2. Empurra o modelo Qwen3.5 0.8B Q4_K_M fine-tuned PT-BR (505 MB)
#   3. Inicia o llama-server na porta 8080
#   4. Mostra instruções para acessar o chat

set -e

# Resolve script directory for safe relative paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo " Qwen3.5 0.8B PT-BR — Deploy Mobile"
echo " Galaxy A54 / Android 14+"
echo "============================================"

# Verifica ADB
if ! command -v adb >/dev/null 2>&1; then
    echo "ERRO: adb não encontrado. Instale o Android Platform Tools."
    echo "  curl -O https://dl.google.com/android/repository/platform-tools-latest-linux.zip"
    exit 1
fi

# Verifica dispositivo
DEVICE=$(adb devices | tail -n +2 | head -1 | awk '{print $2}')
if [ "$DEVICE" != "device" ]; then
    echo "ERRO: Nenhum dispositivo conectado ou não autorizado."
    echo "  Verifique: adb devices"
    echo "  Aceite o pop-up 'Permitir depuração USB' no celular."
    exit 1
fi
echo "✓ Dispositivo conectado"

# Diretórios no dispositivo
TARGET=/data/local/tmp/qwen35-ptbr
MODEL_DIR=$TARGET/models

# Passo 1: Empurrar binários
echo ""
echo "[1/4] Empurrando binários ARM64 otimizados..."
adb shell "mkdir -p $TARGET"
adb push "$PROJ_DIR/binaries/" "$TARGET/" 1>&2
adb shell "chmod 755 $TARGET/llama-cli $TARGET/llama-server $TARGET/llama-bench"
echo "✓ Binários instalados ($(adb shell "ls $TARGET/*.so 2>/dev/null | wc -l") .so + 3 executáveis)"

# Passo 2: Modelo
echo ""
echo "[2/4] Verificando modelo..."
adb shell "mkdir -p $MODEL_DIR"
if [ -f "$PROJ_DIR/model/qwen35-ptbr-q4_k_m.gguf" ]; then
    echo "  Empurrando modelo (505 MB, ~15s)..."
    adb push "$PROJ_DIR/model/qwen35-ptbr-q4_k_m.gguf" "$MODEL_DIR/" 2>&1 | tail -1
    echo "✓ Modelo instalado"
else
    echo "  ⚠ Modelo não encontrado em model/qwen35-ptbr-q4_k_m.gguf"
    echo "  Baixe de: https://huggingface.co/DanielPonttes/qwen35-ptbr-mobile"
    exit 1
fi

# Passo 3: Matar servidor antigo e iniciar novo
echo ""
echo "[3/4] Iniciando servidor..."
adb shell "pkill -f llama-server" 2>/dev/null || true
sleep 2

# Iniciar servidor com configuração otimizada para A54
# -t 3: threads nos A78 (evita A55 lentos)
# -b 128: batch pequeno para baixa latência
# -c 2048: contexto de 2K tokens
# -ctk q8_0 -ctv q8_0: KV cache quantizado (libera ~300 MB)
# --mlock: trava modelo em RAM (evita swap)
adb shell "cd $TARGET && LD_LIBRARY_PATH=. nohup ./llama-server \
    -m $MODEL_DIR/qwen35-ptbr-q4_k_m.gguf \
    -t 3 -b 128 -c 2048 \
    -ctk q8_0 -ctv q8_0 \
    --mlock \
    --host 127.0.0.1 --port 8080 \
    > /dev/null 2>&1 &"

echo "✓ Servidor iniciado (carregando modelo...)"

# Verificar se iniciou (espera até 30s pelo modelo de 505 MB)
echo "  Aguardando servidor ficar pronto..."
for i in $(seq 1 30); do
    sleep 2
    if adb shell "grep -q ':081F ' /proc/net/tcp" 2>/dev/null; then
        echo "✓ Servidor ouvindo na porta 8080"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠ Timeout: servidor pode ainda estar carregando. Verifique manualmente."
    fi
done

# Passo 4: Instruções finais
echo ""
echo "[4/4] Pronto!"
echo ""
echo "============================================"
echo " COMO USAR"
echo "============================================"
echo ""
echo "  Opção 1 — Navegador do celular:"
echo "    Abra: http://127.0.0.1:8080"
echo ""
echo "  Opção 2 — PC (via ADB port forward):"
echo "    adb forward tcp:9090 tcp:8080"
echo "    Abra: http://127.0.0.1:9090"
echo ""
echo "  Opção 3 — API (curl):"
echo "    curl http://127.0.0.1:9090/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Olá!\"}],\"max_tokens\":64}'"
echo ""
echo "  Opção 4 — Termux (chat via terminal):"
echo "    cd $TARGET && LD_LIBRARY_PATH=. ./llama-cli \\"
echo "      -m $MODEL_DIR/qwen35-ptbr-q4_k_m.gguf \\"
echo "      -t 3 -b 128 -c 2048 -cnv"
echo ""
echo "  Benchmark:"
echo "    cd $TARGET && LD_LIBRARY_PATH=. ./llama-bench \\"
echo "      -m $MODEL_DIR/qwen35-ptbr-q4_k_m.gguf -t 3 -b 128 -p 64 -n 128"
echo ""
echo "============================================"
echo "  Performance esperada (Galaxy A54):"
echo "    Geração: ~19-20 tok/s"
echo "    Prefill: ~100-110 tok/s"
echo "    RAM:     ~500 MB"
echo "============================================"
