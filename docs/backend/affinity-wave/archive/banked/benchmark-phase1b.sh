#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/arian/llama.cpp-qwen36
RESULTS="$ROOT/results/qwen36-35b-moe-pp-20260721/affinitywave-20260722"
WATCH_TARGET=/home/arian/.xsession-errors
VARIANT=${1:-native}
WIRE=f32
M64_SPLIT=2

case "$VARIANT" in
    native)
        LAYOUT=native
        DOWN_CACHE=0
        ROUTE_PATTERN=uniform
        ;;
    t64)
        LAYOUT=t64k32
        DOWN_CACHE=0
        ROUTE_PATTERN=uniform
        ;;
    t64-bf16)
        LAYOUT=t64k32
        DOWN_CACHE=0
        ROUTE_PATTERN=uniform
        WIRE=bf16
        ;;
    t64-split1)
        LAYOUT=t64k32
        DOWN_CACHE=0
        ROUTE_PATTERN=uniform
        M64_SPLIT=1
        ;;
    t64-cache)
        LAYOUT=t64k32
        DOWN_CACHE=4
        ROUTE_PATTERN=uniform
        ;;
    t64-tail)
        LAYOUT=t64k32
        DOWN_CACHE=0
        ROUTE_PATTERN=tail-mix
        ;;
    t64-cache-tail)
        LAYOUT=t64k32
        DOWN_CACHE=4
        ROUTE_PATTERN=tail-mix
        ;;
    *)
        echo "unknown variant: $VARIANT" >&2
        exit 2
        ;;
esac

OUT="$RESULTS/phase1b-$VARIANT.log"

watch_xsession() {
    while true; do
        if [[ -f "$WATCH_TARGET" ]] && (( $(stat -c%s "$WATCH_TARGET") > 100000000 )); then
            truncate -s 0 "$WATCH_TARGET"
        fi
        sleep 5
    done
}

watch_xsession &
WATCHDOG_PID=$!
cleanup() {
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if pgrep -f 'bin/llama-[a-z-]*|n[s]ys' >/dev/null; then
    echo "another llama or nsys process is active" >&2
    exit 3
fi

STATUS=0
{
    echo "RUN START $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "variant=$VARIANT layout=$LAYOUT down_cache=$DOWN_CACHE route_pattern=$ROUTE_PATTERN wire=$WIRE m64_split=$M64_SPLIT"
    env -i \
        HOME=/home/arian \
        PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin \
        LD_LIBRARY_PATH="$ROOT/build-p100-affinitywave/bin:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu" \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        NCCL_IB_DISABLE=1 \
        NCCL_CUMEM_ENABLE=0 \
        NCCL_ALGO=Ring \
        GGML_CUDA_P2P=1 \
        GGML_CUDA_AFFINITY_WAVE=1 \
        GGML_CUDA_AW_MAP="$RESULTS/placement-hot16.json" \
        GGML_CUDA_AW_WIRE="$WIRE" \
        GGML_CUDA_AW_CHECK=1 \
        GGML_CUDA_AW_Q8_LAYOUT="$LAYOUT" \
        GGML_CUDA_AW_DOWN_CACHE="$DOWN_CACHE" \
        GGML_CUDA_AW_M64_SPLIT="$M64_SPLIT" \
        GGML_CUDA_AW_HOME=layer \
        timeout --signal=TERM --kill-after=10s 300 \
        taskset --cpu-list 0-11 \
        "$ROOT/build-p100-affinitywave/bin/llama-affinity-wave-bench" \
        --tokens-per-cell 2048 --repeats 12 --route-pattern "$ROUTE_PATTERN" || STATUS=$?
    echo "RUN END $(date -u '+%Y-%m-%d %H:%M:%S UTC') status=$STATUS"
} > "$OUT" 2>&1

cat "$OUT"
exit "$STATUS"
