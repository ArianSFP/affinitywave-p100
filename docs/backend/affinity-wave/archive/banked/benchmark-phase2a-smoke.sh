#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/arian/llama.cpp-qwen36
BIN="$ROOT/build-p100-affinitywave/bin/llama-bench"
MODEL=/home/arian/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf
RESULTS="$ROOT/results/qwen36-35b-moe-pp-20260721/affinitywave-20260722"
MAP="$RESULTS/placement-hot16.json"
EPLB="$RESULTS/placement-primary.eplb"
XS=/home/arian/.xsession-errors
SMOKE_P=${8:-512}
SMOKE_N=${9:-32}
DENSE_BENCH=${10:-0}

watch_xsession() {
    while true; do
        xs=$(stat -c%s "$XS" 2>/dev/null || echo 0)
        avail=$(df --output=avail / | tail -1 | tr -d ' ')
        if [ "${xs:-0}" -gt 52428800 ] || [ "${avail:-999999999}" -lt 5242880 ]; then
            echo "WATCHDOG TRIP $(date -u '+%H:%M:%S UTC')"
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

BASE_ENV=(
    CUDA_VISIBLE_DEVICES=0,1,2,3
    NCCL_IB_DISABLE=1
    NCCL_CUMEM_ENABLE=0
    NCCL_ALGO=Ring
    GGML_CUDA_P2P=1
    GGML_CUDA_GRAPHS_PRE_AMPERE=1
    GGML_CUDA_GRAPHS_SPLIT_BUFFER=1
    GGML_CUDA_Q8_1_DEDUP=1
    GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1
    GGML_CUDA_MOE_CSORT=1
    GGML_CUDA_MOE_EP=1
    GGML_CUDA_MOE_CSORT_REUSE=1
    GGML_CUDA_MOE_F16GATHER=1
    GGML_CUDA_MOE_PINNED=1
    GGML_CUDA_MOE_GROUPED=1
    GGML_CUDA_MOE_PLAN=1
    GGML_CUDA_MOE_EPLB_MAP="$EPLB"
    GGML_META_SUBMIT_THREADS=1
)

run_arm() {
    arm=$1
    shift
    out="$RESULTS/phase2a-smoke-$arm.json"
    err="$RESULTS/phase2a-smoke-$arm.err"
    echo "RUN START arm=$arm $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    env -i "${BASE_ENV[@]}" "$@" HOME=/home/arian PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin \
        LD_LIBRARY_PATH="$ROOT/build-p100-affinitywave/bin:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu" \
        timeout --signal=TERM --kill-after=10s 900 taskset --cpu-list 0-11 "$BIN" \
        --model "$MODEL" -ngl 99 -sm tensor -fa 1 -b "$SMOKE_P" -ub "$SMOKE_P" -mmp 0 \
        -p "$SMOKE_P" -n "$SMOKE_N" -r 1 -o json > "$out" 2> "$err"
    status=$?
    echo "RUN END arm=$arm status=$status $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    if [ "$status" -eq 0 ]; then
        grep -E '"test"|"avg_ts"' "$out" | paste - -
    else
        tail -80 "$err"
    fi
    return "$status"
}

if [ "${1:-both}" != t64 ]; then
    run_arm native || exit $?
fi
run_arm t64 \
    GGML_CUDA_AFFINITY_WAVE=1 \
    GGML_CUDA_AW_MAP="$MAP" \
    GGML_CUDA_AW_WIRE="${17:-${AW_WIRE:-f32}}" \
    GGML_CUDA_AW_CHECK=1 \
    GGML_CUDA_AW_Q8_LAYOUT=t64k32 \
    GGML_CUDA_AW_DOWN_CACHE=0 \
    GGML_CUDA_AW_M64_SPLIT=2 \
    GGML_CUDA_AW_HOME=layer \
    GGML_CUDA_AW_DECODE="${2:-${AW_DECODE:-t64}}" \
    GGML_CUDA_AW_WAVE_INSPECT="${3:-0}" \
    GGML_CUDA_AW_WAVE_DRY="${4:-0}" \
    GGML_CUDA_AW_WAVE_DENSE="${5:-0}" \
    GGML_CUDA_AW_WAVE_TOKEN_SPLIT="${6:-0}" \
    GGML_CUDA_AW_WAVE_TOKEN_PLAN="${7:-0}" \
    GGML_CUDA_AW_WAVE_DENSE_BENCH="$DENSE_BENCH" \
    GGML_CUDA_AW_LANE_STAGGER="${11:-${AW_LANE_STAGGER:-1}}" \
    GGML_CUDA_AW_PRECAPTURE="${12:-${AW_PRECAPTURE:-0}}" \
    GGML_CUDA_AW_SHARED_SERVICE="${13:-${AW_SHARED_SERVICE:-0}}" \
    GGML_CUDA_AW_LANE_BALANCE="${14:-${AW_LANE_BALANCE:-0}}" \
    CUDA_LAUNCH_BLOCKING="${15:-${AW_CUDA_LAUNCH_BLOCKING:-0}}" \
    GGML_CUDA_AW_DEBUG_NODE_SYNC="${16:-${AW_DEBUG_NODE_SYNC:-0}}" \
    GGML_CUDA_AW_GROUP_CELLS="${18:-${AW_GROUP_CELLS:-2}}" \
    GGML_CUDA_AW_GDN_CHUNKED="${19:-${AW_GDN_CHUNKED:-0}}" \
    GGML_CUDA_AW_GROUP_PATTERN="${20:-${AW_GROUP_PATTERN:-}}" \
    GGML_CUDA_AW_GDN_WARPS="${21:-${AW_GDN_WARPS:-8}}" \
    GGML_CUDA_AW_FUSED_GATE_UP="${22:-${AW_FUSED_GATE_UP:-0}}" \
    GGML_CUDA_AW_WAVE_TOKENS="$SMOKE_P"
