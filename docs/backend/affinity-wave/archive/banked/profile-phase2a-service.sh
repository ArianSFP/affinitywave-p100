#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/arian/llama.cpp-qwen36
RESULTS="$ROOT/results/qwen36-35b-moe-pp-20260721/affinitywave-20260722"
BIN="$ROOT/build-p100-affinitywave/bin/llama-bench"
MODEL=/home/arian/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf
XS=/home/arian/.xsession-errors
P=${1:-1024}

watch_xsession() {
    while true; do
        xs=$(stat -c%s "$XS" 2>/dev/null || echo 0)
        avail=$(df --output=avail / | tail -1 | tr -d ' ')
        if [ "${xs:-0}" -gt 52428800 ] || [ "${avail:-999999999}" -lt 5242880 ]; then
            : > "$XS" 2>/dev/null
            pkill -9 -f 'build-p100-affinitywave/bin/llama-' || true
            break
        fi
        sleep 2
    done
}

watch_xsession &
WATCHDOG_PID=$!
trap 'kill "$WATCHDOG_PID" 2>/dev/null || true' EXIT INT TERM

env \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NCCL_IB_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_ALGO=Ring \
    GGML_CUDA_P2P=1 GGML_CUDA_GRAPHS_PRE_AMPERE=1 GGML_CUDA_GRAPHS_SPLIT_BUFFER=1 \
    GGML_CUDA_Q8_1_DEDUP=1 GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1 \
    GGML_CUDA_MOE_CSORT=1 GGML_CUDA_MOE_EP=1 GGML_CUDA_MOE_CSORT_REUSE=1 \
    GGML_CUDA_MOE_F16GATHER=1 GGML_CUDA_MOE_PINNED=1 GGML_CUDA_MOE_GROUPED=1 \
    GGML_CUDA_MOE_PLAN=1 GGML_CUDA_MOE_EPLB_MAP="$RESULTS/placement-primary.eplb" \
    GGML_META_SUBMIT_THREADS=1 \
    GGML_CUDA_AFFINITY_WAVE=1 GGML_CUDA_AW_MAP="$RESULTS/placement-hot16.json" \
    GGML_CUDA_AW_WIRE="${3:-f32}" GGML_CUDA_AW_CHECK=1 GGML_CUDA_AW_Q8_LAYOUT=t64k32 \
    GGML_CUDA_AW_DOWN_CACHE=0 GGML_CUDA_AW_M64_SPLIT=2 GGML_CUDA_AW_HOME=layer \
    GGML_CUDA_AW_DECODE=t64 GGML_CUDA_AW_WAVE_DRY=1 GGML_CUDA_AW_WAVE_DENSE=1 \
    GGML_CUDA_AW_WAVE_TOKEN_SPLIT=1 GGML_CUDA_AW_WAVE_DENSE_BENCH=service \
    GGML_CUDA_AW_PRECAPTURE=1 GGML_CUDA_AW_GROUP_CELLS=2 \
    GGML_CUDA_AW_GROUP_PATTERN="${4:-}" \
    GGML_CUDA_AW_GDN_CHUNKED="${2:-0}" GGML_CUDA_AW_GDN_WARPS="${5:-8}" \
    GGML_CUDA_AW_WAVE_TOKENS="$P" \
    nsys profile --trace=cuda --sample=none --cpuctxsw=none --delay=22 --duration=15 \
        --force-overwrite=true -o "$RESULTS/phase2a-service-p${P}-gdn${2:-0}w${5:-8}-${3:-f32}-gp${4:-none}" \
        taskset --cpu-list 0-11 "$BIN" --model "$MODEL" -ngl 99 -sm tensor -fa 1 \
        -b "$P" -ub "$P" -mmp 0 -p "$P" -n 0 -r 1 -o json
