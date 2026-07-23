#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/arian/llama.cpp-qwen36
OUT="$ROOT/results/qwen36-35b-moe-pp-20260721/affinitywave-20260722/phase1-service-f32-next.log"
WATCH_TARGET=/home/arian/.xsession-errors

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

{
    echo "RUN START $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "git 05dcabf9a + uncommitted TAG_AFFINITY_WAVE Phase-1 prototype"
    echo "command: env -i <CUDA/NCCL/AffinityWave env> taskset --cpu-list 0-11 llama-affinity-wave-bench --tokens-per-cell 2048 --repeats 12"
    STATUS=0
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
        GGML_CUDA_AW_MAP="$ROOT/results/qwen36-35b-moe-pp-20260721/affinitywave-20260722/placement-hot16.json" \
        GGML_CUDA_AW_WIRE=f32 \
        GGML_CUDA_AW_CHECK=1 \
        timeout --signal=TERM --kill-after=10s 300 \
        taskset --cpu-list 0-11 \
        "$ROOT/build-p100-affinitywave/bin/llama-affinity-wave-bench" \
        --tokens-per-cell 2048 --repeats 12 || STATUS=$?
    echo "RUN END $(date -u '+%Y-%m-%d %H:%M:%S UTC') status=$STATUS"
} > "$OUT" 2>&1
exit "$STATUS"
