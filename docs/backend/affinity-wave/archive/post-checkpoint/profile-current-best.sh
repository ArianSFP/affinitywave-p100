#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/arian/llama.cpp-qwen36
BUILD="$ROOT/build-p100-affinitywave-experimental"
BIN="$BUILD/bin/llama-bench"
MODEL=/home/arian/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf
BANKED="$ROOT/results/qwen36-35b-moe-pp-20260721/affinitywave-20260722"
RESULTS="$ROOT/results/qwen36-35b-moe-pp-20260721/affinitywave-experimental-20260723"
MAP="$BANKED/placement-hot16.json"
EPLB="$BANKED/placement-primary.eplb"
XS=/home/arian/.xsession-errors
P=${1:-8128}
TAG=${2:-current-best-p8128}

watch_xsession() {
    while true; do
        xs=$(stat -c%s "$XS" 2>/dev/null || echo 0)
        avail=$(df --output=avail / | tail -1 | tr -d ' ')
        if [ "${xs:-0}" -gt 52428800 ] || [ "${avail:-999999999}" -lt 5242880 ]; then
            echo "WATCHDOG TRIP $(date -u '+%H:%M:%S UTC')"
            : > "$XS" 2>/dev/null
            pkill -9 -f 'build-p100-affinitywave-experimental/bin/llama-' || true
            break
        fi
        sleep 2
    done
}

watch_xsession &
WATCHDOG_PID=$!
trap 'kill "$WATCHDOG_PID" 2>/dev/null || true' EXIT INT TERM

echo "PROFILE START tag=$TAG p=$P $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
env \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NCCL_IB_DISABLE=1 \
    NCCL_CUMEM_ENABLE=0 \
    NCCL_ALGO=Ring \
    GGML_CUDA_P2P=1 \
    GGML_CUDA_GRAPHS_PRE_AMPERE=1 \
    GGML_CUDA_GRAPHS_SPLIT_BUFFER=1 \
    GGML_CUDA_Q8_1_DEDUP=1 \
    GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1 \
    GGML_CUDA_MOE_CSORT=1 \
    GGML_CUDA_MOE_EP=1 \
    GGML_CUDA_MOE_CSORT_REUSE=1 \
    GGML_CUDA_MOE_F16GATHER=1 \
    GGML_CUDA_MOE_PINNED=1 \
    GGML_CUDA_MOE_GROUPED=1 \
    GGML_CUDA_MOE_PLAN=1 \
    GGML_CUDA_MOE_EPLB_MAP="$EPLB" \
    GGML_META_SUBMIT_THREADS=1 \
    GGML_CUDA_AFFINITY_WAVE=1 \
    GGML_CUDA_AW_MAP="$MAP" \
    GGML_CUDA_AW_WIRE=bf16 \
    GGML_CUDA_AW_CHECK=1 \
    GGML_CUDA_AW_Q8_LAYOUT=t64k32 \
    GGML_CUDA_AW_DOWN_CACHE=0 \
    GGML_CUDA_AW_M64_SPLIT=2 \
    GGML_CUDA_AW_M64_ALL=0 \
    GGML_CUDA_AW_HOME=layer \
    GGML_CUDA_AW_DECODE=t64 \
    GGML_CUDA_AW_WAVE_DRY=0 \
    GGML_CUDA_AW_WAVE_DENSE=1 \
    GGML_CUDA_AW_WAVE_TOKEN_SPLIT=1 \
    GGML_CUDA_AW_WAVE_TOKEN_PLAN=0 \
    GGML_CUDA_AW_WAVE_DENSE_BENCH=service \
    GGML_CUDA_AW_LANE_STAGGER=1 \
    GGML_CUDA_AW_PRECAPTURE=1 \
    GGML_CUDA_AW_SHARED_SERVICE=0 \
    GGML_CUDA_AW_LANE_BALANCE=0 \
    GGML_CUDA_AW_DEBUG_NODE_SYNC=0 \
    GGML_CUDA_AW_GROUP_CELLS=1 \
    GGML_CUDA_AW_GDN_CHUNKED=2 \
    GGML_CUDA_AW_GROUP_PATTERN=1111 \
    GGML_CUDA_AW_GDN_WARPS=4 \
    GGML_CUDA_AW_FUSED_GATE_UP=0 \
    GGML_CUDA_AW_CORRIDOR_ASYNC=0 \
    GGML_CUDA_AW_CORRIDOR_EARLY=all \
    GGML_CUDA_AW_CORRIDOR_STATE_SPLIT=1 \
    GGML_CUDA_AW_WAVE_TOKENS="$P" \
    LD_LIBRARY_PATH="$BUILD/bin:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu" \
    timeout --signal=TERM --kill-after=10s 900 \
    nsys profile --trace=cuda --sample=none --cpuctxsw=none --delay=22 --duration=15 \
        --force-overwrite=true -o "$RESULTS/$TAG" \
        taskset --cpu-list 0-11 "$BIN" --model "$MODEL" -ngl 99 -sm tensor -fa 1 \
        -b "$P" -ub "$P" -mmp 0 -p "$P" -n 1 -r 1 -o json \
        > "$RESULTS/$TAG.json" 2> "$RESULTS/$TAG.err"
status=$?
echo "PROFILE END tag=$TAG status=$status $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
tail -80 "$RESULTS/$TAG.err"
exit "$status"
