#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/arian/llama.cpp-q36-moe/.worktrees/affinitywave-openai-sass
BUILD=/home/arian/llama.cpp-q36-moe/.worktrees/build-p100-openai-sass
RESULTS="$ROOT/results/qwen36-35b-moe-pp-20260721/perf-recovery-20260723"
REFERENCE=/home/arian/llama.cpp-qwen36/results/qwen36-35b-moe-pp-20260721/affinitywave-20260722
MODEL=/home/arian/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf
XS=/home/arian/.xsession-errors

TAG=${1:?tag}
Q8_KERNEL=${2:-interleave}
WIRE=${3:-f32}
NCCL_SUM=${4:-1}
WAVE_OUTPUT=${5:-0}
PRECAPTURE=${6:-0}
TOKENS=${7:-8128}
NCCL_ORDER_SUM=${8:-0}
PRECAPTURE_OUTPUT=${9:-0}

mkdir -p "$RESULTS"
exec 9>/tmp/affinitywave-4gpu.lock
if ! flock -n 9; then
    echo "another AffinityWave job holds /tmp/affinitywave-4gpu.lock" >&2
    exit 75
fi

watch_xsession() {
    while true; do
        xs=$(stat -c%s "$XS" 2>/dev/null || echo 0)
        avail=$(df --output=avail / | tail -1 | tr -d ' ')
        if [ "${xs:-0}" -gt 52428800 ] || [ "${avail:-999999999}" -lt 5242880 ]; then
            echo "WATCHDOG TRIP $(date -u '+%H:%M:%S UTC')" >&2
            : > "$XS" 2>/dev/null
            pkill -9 -f "$BUILD/bin/llama-" || true
            break
        fi
        sleep 2
    done
}

watch_xsession 9>&- &
WATCHDOG_PID=$!
cleanup() {
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ENV_ARGS=(
    HOME=/home/arian
    PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin
    LD_LIBRARY_PATH="$BUILD/bin:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu"
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
    GGML_CUDA_MOE_EPLB_MAP="$REFERENCE/placement-primary.eplb"
    GGML_META_SUBMIT_THREADS=1
    GGML_CUDA_AFFINITY_WAVE=1
    GGML_CUDA_AW_MAP="$REFERENCE/placement-hot16.json"
    GGML_CUDA_AW_WIRE="$WIRE"
    GGML_CUDA_AW_PARTIAL=bf16
    GGML_CUDA_AW_NCCL_SUM="$NCCL_SUM"
    GGML_CUDA_AW_NCCL_ORDER_SUM="$NCCL_ORDER_SUM"
    GGML_CUDA_AW_CHECK=1
    GGML_CUDA_AW_Q8_LAYOUT=t64k32
    GGML_CUDA_AW_Q8_KERNEL="$Q8_KERNEL"
    GGML_CUDA_AW_DOWN_CACHE=0
    GGML_CUDA_AW_M64_SPLIT=2
    GGML_CUDA_AW_HOME=layer
    GGML_CUDA_AW_DECODE=t64
    GGML_CUDA_AW_WAVE_INSPECT=0
    GGML_CUDA_AW_WAVE_DRY=0
    GGML_CUDA_AW_WAVE_DENSE=1
    GGML_CUDA_AW_WAVE_TOKEN_SPLIT=1
    GGML_CUDA_AW_WAVE_TOKEN_PLAN=0
    GGML_CUDA_AW_WAVE_DENSE_BENCH=service
    GGML_CUDA_AW_WAVE_OUTPUT="$WAVE_OUTPUT"
    GGML_CUDA_AW_LANE_STAGGER=1
    GGML_CUDA_AW_CORRIDOR_EARLY=all
    GGML_CUDA_AW_CORRIDOR_STATE_SPLIT=1
    GGML_CUDA_AW_PRECAPTURE="$PRECAPTURE"
    GGML_CUDA_AW_PRECAPTURE_OUTPUT="$PRECAPTURE_OUTPUT"
    GGML_CUDA_AW_SHARED_SERVICE=0
    GGML_CUDA_AW_LANE_BALANCE=0
    GGML_CUDA_AW_DEBUG_NODE_SYNC=0
    GGML_CUDA_AW_DEBUG_LIVE_SYNC=0
    GGML_CUDA_AW_GROUP_CELLS=1
    GGML_CUDA_AW_GROUP_PATTERN=1111
    GGML_CUDA_AW_GDN_CHUNKED=2
    GGML_CUDA_AW_GDN_WARPS=4
    GGML_CUDA_AW_FUSED_GATE_UP=0
    GGML_CUDA_AW_WAVE_TOKENS="$TOKENS"
)

OUT="$RESULTS/$TAG.out"
ERR="$RESULTS/$TAG.err"
REPORT="$RESULTS/$TAG"

echo "RUN START tag=$TAG q8=$Q8_KERNEL wire=$WIRE nccl=$NCCL_SUM output=$WAVE_OUTPUT precapture=$PRECAPTURE tokens=$TOKENS $(date -u '+%Y-%m-%d %H:%M:%S UTC')" > "$OUT"
env "${ENV_ARGS[@]}" \
    timeout --signal=TERM --kill-after=10s 900 \
    nsys profile \
        --trace=cuda,nvtx \
        --sample=none \
        --cpuctxsw=none \
        --cuda-graph-trace=node \
        --force-overwrite=true \
        --output="$REPORT" \
        /bin/bash -c 'wave_output=$1; shift; "$@"; status=$?; if [ "$status" -eq 1 ] && [ "$wave_output" -eq 0 ]; then exit 0; fi; exit "$status"' \
            recovery-trace "$WAVE_OUTPUT" \
            taskset --cpu-list 0-11 "$BUILD/bin/llama-bench" \
                --model "$MODEL" -ngl 99 -sm tensor -fa 1 \
                -b "$TOKENS" -ub "$TOKENS" -mmp 0 --no-warmup \
                -p "$TOKENS" -n 0 -r 1 -o json \
            >> "$OUT" 2> "$ERR"
status=$?
echo "RUN END tag=$TAG status=$status $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$OUT"

if [ "$status" -eq 0 ] && [ -s "$REPORT.nsys-rep" ]; then
    nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
        --format csv --force-export=true "$REPORT.nsys-rep" \
        > "$RESULTS/$TAG.stats.csv"
else
    tail -100 "$ERR"
fi
exit "$status"
