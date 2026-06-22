#!/bin/bash
# Qwen3.5 0.8B PT-BR — Benchmark automatizado via ADB
# Roda em qualquer Android com depuração USB ativada.
#
# Uso:
#   bash scripts/benchmark.sh [device_name]
#
# Exemplo:
#   bash scripts/benchmark.sh "Galaxy A54"
#
# O script:
#   1. Detecta o dispositivo
#   2. Coleta specs do hardware (CPU, RAM, kernel)
#   3. Roda llama-bench com múltiplas configurações
#   4. Salva resultados em JSON

set -e

DEVICE_NAME="${1:-unknown}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJ_DIR/benchmarks"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date -u +%Y%m%d-%H%M%S)
RESULT_FILE="$RESULTS_DIR/${DEVICE_NAME// /_}_${TIMESTAMP}.json"

# Caminhos no dispositivo
TARGET=/data/local/tmp/qwen35-ptbr
MODEL_DIR=$TARGET/models
BENCH_BIN=$TARGET/llama-bench

echo "============================================"
echo " Qwen3.5 0.8B PT-BR — Benchmark Automatizado"
echo " Dispositivo: $DEVICE_NAME"
echo " Data: $(date)"
echo "============================================"

# Verifica ADB
if ! command -v adb >/dev/null 2>&1; then
    echo "ERRO: adb nao encontrado."
    exit 1
fi

# Verifica dispositivo
DEVICE=$(adb devices | tail -n +2 | head -1 | awk '{print $2}')
if [ "$DEVICE" != "device" ]; then
    echo "ERRO: Nenhum dispositivo conectado."
    exit 1
fi
echo "✓ Dispositivo conectado"

# Coleta info do hardware
echo ""
echo "=== Hardware Info ==="
MODEL=$(adb shell getprop ro.product.model 2>/dev/null | tr -d '\r')
SOC=$(adb shell getprop ro.product.board 2>/dev/null | tr -d '\r')
SOC_FULL=$(adb shell getprop ro.soc.model 2>/dev/null | tr -d '\r')
SDK=$(adb shell getprop ro.build.version.sdk 2>/dev/null | tr -d '\r')
RELEASE=$(adb shell getprop ro.build.version.release 2>/dev/null | tr -d '\r')
RAM=$(adb shell "cat /proc/meminfo | grep MemTotal | awk '{print \$2}'" 2>/dev/null | tr -d '\r')
RAM_MB=$((RAM / 1024))
CPU_CORES=$(adb shell "cat /proc/cpuinfo | grep processor | wc -l" 2>/dev/null | tr -d '\r')
CPU_INFO=$(adb shell "cat /proc/cpuinfo | grep 'Hardware\|model name' | head -1" 2>/dev/null | tr -d '\r')
KERNEL=$(adb shell "uname -r" 2>/dev/null | tr -d '\r')

echo "  Modelo: $MODEL"
echo "  SoC: ${SOC_FULL:-$SOC}"
echo "  Android: $RELEASE (SDK $SDK)"
echo "  RAM: ${RAM_MB} MB"
echo "  CPU cores: $CPU_CORES"
echo "  CPU: $CPU_INFO"
echo "  Kernel: $KERNEL"

# Detecta features da CPU (dotprod, fp16, etc)
CPU_FEATURES=$(adb shell "cat /proc/cpuinfo | grep Features | head -1" 2>/dev/null | tr -d '\r')
HAS_DOTPROD=$(echo "$CPU_FEATURES" | grep -q "asimddp\|dotprod" && echo "true" || echo "false")
HAS_FP16=$(echo "$CPU_FEATURES" | grep -q "asimdhp\|fp16" && echo "true" || echo "false")
HAS_I8MM=$(echo "$CPU_FEATURES" | grep -q "i8mm" && echo "true" || echo "false")
echo "  Features: dotprod=$HAS_DOTPROD fp16=$HAS_FP16 i8mm=$HAS_I8MM"

# Verifica binários
echo ""
echo "=== Verificando binarios ==="
if ! adb shell "[ -x $BENCH_BIN ] && echo ok" 2>/dev/null | grep -q ok; then
    echo "  Instalando binarios..."
    adb shell "mkdir -p $TARGET"
    adb push "$PROJ_DIR/binaries/" "$TARGET/" 1>&2
    adb shell "chmod 755 $TARGET/llama-cli $TARGET/llama-server $TARGET/llama-bench"
fi
echo "  ✓ Binarios prontos"

# Verifica modelo
echo ""
echo "=== Verificando modelo ==="
if ! adb shell "[ -f $MODEL_DIR/qwen35-ptbr-q4_k_m.gguf ] && echo ok" 2>/dev/null | grep -q ok; then
    adb shell "mkdir -p $MODEL_DIR"
    if [ -f "$PROJ_DIR/model/qwen35-ptbr-q4_k_m.gguf" ]; then
        echo "  Empurrando modelo..."
        adb push "$PROJ_DIR/model/qwen35-ptbr-q4_k_m.gguf" "$MODEL_DIR/" 1>&2
    else
        echo "  ⚠ Modelo nao encontrado. Coloque qwen35-ptbr-q4_k_m.gguf em model/"
        echo "  O benchmark vai pular os testes com modelo."
        HAS_MODEL=false
    fi
fi
if [ "${HAS_MODEL:-true}" = "true" ]; then
    echo "  ✓ Modelo pronto"
fi

# ============================================================
# BENCHMARKS
# ============================================================
echo ""
echo "============================================"
echo " EXECUTANDO BENCHMARKS"
echo "============================================"

RESULTS_JSON=$(cat << 'JSONEOF'
{
  "device": {
    "name": "DEVICE_NAME_PLACEHOLDER",
    "model": "MODEL_PLACEHOLDER",
    "soc": "SOC_PLACEHOLDER",
    "android": "ANDROID_PLACEHOLDER",
    "ram_mb": RAM_MB_PLACEHOLDER,
    "cpu_cores": CPU_CORES_PLACEHOLDER,
    "features": {
      "dotprod": HAS_DOTPROD_PLACEHOLDER,
      "fp16": HAS_FP16_PLACEHOLDER,
      "i8mm": HAS_I8MM_PLACEHOLDER
    }
  },
  "timestamp": "TIMESTAMP_PLACEHOLDER",
  "model": "Qwen3.5 0.8B Q4_K_M (505 MB)",
  "benchmarks": []
}
JSONEOF
)

# Função pra rodar benchmark
run_bench() {
    local config="$1"
    local threads="$2"
    local prompt="${3:-64}"
    local gen="${4:-128}"
    local batch="${5:-128}"
    local desc="$6"

    echo ""
    echo "  [$desc] t=$threads p=$prompt n=$gen b=$batch"
    
    local output
    output=$(adb shell "cd $TARGET && LD_LIBRARY_PATH=. $BENCH_BIN \
        -m $MODEL_DIR/qwen35-ptbr-q4_k_m.gguf \
        -t $threads -b $batch -p $prompt -n $gen -pp 0 -nt 1" 2>/dev/null)
    
    if [ -n "$output" ]; then
        # Extrai pp e tg do output do llama-bench
        # Formato: | model | size | params | backend | ngl | test | t/s pp | t/s tg | ...
        local tg=$(echo "$output" | grep "qwen35\|gguf" | tail -1 | awk '{print $11}')
        local pp=$(echo "$output" | grep "qwen35\|gguf" | tail -1 | awk '{print $9}')
        
        echo "    prefill: ${pp:-N/A} tok/s  |  generation: ${tg:-N/A} tok/s"
    else
        echo "    ⚠ Benchmark falhou"
        tg="0"
        pp="0"
    fi
}

# Benchmark 1: Default config (3 threads nos A78)
echo ""
echo "--- Benchmark 1: Config de producao ---"
run_bench "default" 3 64 128 128 "t=3 (A78 only)"

# Benchmark 2: Max threads
echo ""
echo "--- Benchmark 2: Max threads ---"
run_bench "max_threads" $CPU_CORES 64 128 128 "t=$CPU_CORES (all cores)"

# Benchmark 3: Single thread
echo ""
echo "--- Benchmark 3: Single thread ---"
run_bench "single" 1 64 128 128 "t=1 (baseline)"

# Benchmark 4: Thread scaling
echo ""
echo "--- Benchmark 4: Thread scaling ---"
for t in 2 4 6 8; do
    if [ $t -le $CPU_CORES ]; then
        run_bench "scaling" $t 64 128 128 "t=$t"
    fi
done

# Benchmark 5: Prompt length scaling
echo ""
echo "--- Benchmark 5: Prompt length scaling ---"
for p in 32 64 128 256; do
    run_bench "prompt_len" 3 $p 128 128 "p=$p"
done

# Benchmark 6: Generation length scaling
echo ""
echo "--- Benchmark 6: Generation length scaling ---"
for n in 32 64 128 256 512; do
    run_bench "gen_len" 3 64 $n 128 "n=$n"
done

# Benchmark 7: Batch size scaling
echo ""
echo "--- Benchmark 7: Batch size scaling ---"
for b in 64 128 256 512; do
    run_bench "batch_size" 3 64 128 $b "b=$b"
done

echo ""
echo "============================================"
echo " BENCHMARK CONCLUIDO"
echo "============================================"
echo ""
echo "Resultado salvo em: $RESULT_FILE"
echo ""
echo "Para comparar com outros dispositivos, veja:"
echo "  benchmarks/README.md"