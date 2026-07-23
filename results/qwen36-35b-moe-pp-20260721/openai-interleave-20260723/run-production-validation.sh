#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/arian/llama.cpp-q36-moe/.worktrees/affinitywave-openai-sass
BUILD=/home/arian/llama.cpp-q36-moe/.worktrees/build-p100-openai-sass
RESULTS="$ROOT/results/qwen36-35b-moe-pp-20260721/openai-interleave-20260723"
REFERENCE=/home/arian/llama.cpp-qwen36/results/qwen36-35b-moe-pp-20260721/affinitywave-20260722
MODEL=/home/arian/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf
CORPUS=/home/arian/llama.cpp-q36-decodeopt/wikitext-2-raw/wiki.test.raw
XS=/home/arian/.xsession-errors

MODE=${1:?base, affinity, or bench}
TOKENS=${2:-512}
Q8_KERNEL=${3:-interleave}
LAUNCH_BLOCKING=${4:-0}
DEBUG_LIVE_SYNC=${5:-0}
DEBUG_NODE_SYNC=${6:-0}
WIRE=${7:-f32}
VERIFY_LOCAL=${8:-0}
DEBUG_ROUTES=${9:-0}
FULL_ALLOC=${10:-0}
PARTIAL=${11:-bf16}
GDN_CHUNKED=${12:-2}
SYNC_DIAGONAL=${13:-0}
ACTIVATION_DUMP=${14:-}
ACTIVATION_FILTER=${15:-}
DEBUG_CORRIDOR=${16:-0}
ACTIVATION_ALL_DEVICES=${17:-0}
UBATCH=${18:-$TOKENS}
BASE_FILE=${19:-"$RESULTS/logits-base-c$TOKENS.bin"}
NCCL_SUM=${20:-1}
SERVICE_DUMP=${21:-}
SERVICE_DUMP_LAYER=${22:-0}
GROUP_PATTERN=${23:-1111}
GROUP_CELLS=${24:-1}
WAVE_OUTPUT=${25:-1}
PRECAPTURE=${26:-0}
NCCL_ORDER_SUM=${27:-0}
PRECAPTURE_OUTPUT=${28:-0}
NO_WARMUP=${29:-1}
CORRIDOR_EARLY=${31:-0}
CORRIDOR_STATE_SPLIT=${32:-0}
OUT="$RESULTS/production-$MODE-c$TOKENS.out"
ERR="$RESULTS/production-$MODE-c$TOKENS.err"

if [ -n "$ACTIVATION_DUMP" ]; then
    mkdir -p "$ACTIVATION_DUMP"
fi
if [ -n "$SERVICE_DUMP" ]; then
    mkdir -p "$SERVICE_DUMP"
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

if pgrep -f 'bin/llama-[a-z-]*|n[s]ys' >/dev/null; then
    echo "another llama or nsys process is active" >&2
    exit 3
fi

watch_xsession &
WATCHDOG_PID=$!
cleanup() {
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

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
    GGML_CUDA_MOE_EPLB_MAP="$REFERENCE/placement-primary.eplb"
    GGML_META_SUBMIT_THREADS=1
    GGML_CUDA_ACTIVATION_DUMP="$ACTIVATION_DUMP"
    GGML_CUDA_ACTIVATION_FILTER="$ACTIVATION_FILTER"
    GGML_CUDA_ACTIVATION_ALL_DEVICES="$ACTIVATION_ALL_DEVICES"
)

AW_ENV=(
    GGML_CUDA_AFFINITY_WAVE=1
    GGML_CUDA_AW_MAP="$REFERENCE/placement-hot16.json"
    GGML_CUDA_AW_WIRE="$WIRE"
    GGML_CUDA_AW_PARTIAL="$PARTIAL"
    GGML_CUDA_AW_NCCL_SUM="$NCCL_SUM"
    GGML_CUDA_AW_NCCL_ORDER_SUM="$NCCL_ORDER_SUM"
    GGML_CUDA_AW_SERVICE_DUMP="$SERVICE_DUMP"
    GGML_CUDA_AW_SERVICE_DUMP_LAYER="$SERVICE_DUMP_LAYER"
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
    GGML_CUDA_AW_CORRIDOR_EARLY="$CORRIDOR_EARLY"
    GGML_CUDA_AW_CORRIDOR_STATE_SPLIT="$CORRIDOR_STATE_SPLIT"
    GGML_CUDA_AW_PRECAPTURE="$PRECAPTURE"
    GGML_CUDA_AW_PRECAPTURE_OUTPUT="$PRECAPTURE_OUTPUT"
    GGML_CUDA_AW_SHARED_SERVICE=0
    GGML_CUDA_AW_LANE_BALANCE=0
    CUDA_LAUNCH_BLOCKING="$LAUNCH_BLOCKING"
    GGML_CUDA_AW_DEBUG_NODE_SYNC="$DEBUG_NODE_SYNC"
    GGML_CUDA_AW_SYNC_DIAGONAL="$SYNC_DIAGONAL"
    GGML_CUDA_AW_DEBUG_LIVE_SYNC="$DEBUG_LIVE_SYNC"
    GGML_CUDA_AW_VERIFY_LOCAL="$VERIFY_LOCAL"
    GGML_CUDA_AW_DEBUG_ROUTES="$DEBUG_ROUTES"
    GGML_CUDA_AW_FULL_ALLOC="$FULL_ALLOC"
    GGML_CUDA_AW_GROUP_CELLS="$GROUP_CELLS"
    GGML_CUDA_AW_GDN_CHUNKED="$GDN_CHUNKED"
    GGML_CUDA_AW_DEBUG_CORRIDOR="$DEBUG_CORRIDOR"
    GGML_CUDA_AW_GROUP_PATTERN="$GROUP_PATTERN"
    GGML_CUDA_AW_GDN_WARPS=4
    GGML_CUDA_AW_FUSED_GATE_UP=0
    GGML_CUDA_AW_WAVE_TOKENS="$TOKENS"
)

COMMON_ARGS=(
    --model "$MODEL"
    -ngl 99
    -sm tensor
    -dev CUDA0,CUDA1,CUDA2,CUDA3
    -fa on
    -b "$TOKENS"
    -ub "$UBATCH"
    --no-mmap
    --no-warmup
)

BENCH_ARGS=(
    --model "$MODEL"
    -ngl 99
    -sm tensor
    -fa 1
    -b "$TOKENS"
    -ub "$UBATCH"
    -mmp 0
)
if [ "$NO_WARMUP" -ne 0 ]; then
    BENCH_ARGS+=(--no-warmup)
fi

STATUS=0
echo "RUN START mode=$MODE tokens=$TOKENS q8_kernel=$Q8_KERNEL wire=$WIRE $(date -u '+%Y-%m-%d %H:%M:%S UTC')" > "$OUT"
case "$MODE" in
    base)
        env -i "${BASE_ENV[@]}" \
            HOME=/home/arian \
            PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin \
            LD_LIBRARY_PATH="$BUILD/bin:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu" \
            timeout --signal=TERM --kill-after=10s 900 taskset --cpu-list 0-11 \
            "$BUILD/bin/llama-perplexity" "${COMMON_ARGS[@]}" \
            -f "$CORPUS" -c "$TOKENS" --chunks 1 --save-all-logits "$BASE_FILE" \
            >> "$OUT" 2> "$ERR" || STATUS=$?
        ;;
    compare)
        test -s "$BASE_FILE" || {
            echo "missing base logits: $BASE_FILE" >&2
            exit 4
        }
        env -i "${BASE_ENV[@]}" \
            HOME=/home/arian \
            PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin \
            LD_LIBRARY_PATH="$BUILD/bin:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu" \
            timeout --signal=TERM --kill-after=10s 900 taskset --cpu-list 0-11 \
            "$BUILD/bin/llama-perplexity" "${COMMON_ARGS[@]}" \
            -f "$CORPUS" -c "$TOKENS" --chunks 1 \
            --kl-divergence-base "$BASE_FILE" --kl-divergence \
            >> "$OUT" 2> "$ERR" || STATUS=$?
        ;;
    affinity)
        test -s "$BASE_FILE" || {
            echo "missing base logits: $BASE_FILE" >&2
            exit 4
        }
        env -i "${BASE_ENV[@]}" "${AW_ENV[@]}" \
            HOME=/home/arian \
            PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin \
            LD_LIBRARY_PATH="$BUILD/bin:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu" \
            timeout --signal=TERM --kill-after=10s 900 taskset --cpu-list 0-11 \
            "$BUILD/bin/llama-perplexity" "${COMMON_ARGS[@]}" \
            -f "$CORPUS" -c "$TOKENS" --chunks 1 \
            --kl-divergence-base "$BASE_FILE" --kl-divergence \
            >> "$OUT" 2> "$ERR" || STATUS=$?
        ;;
    bench)
        env -i "${BASE_ENV[@]}" "${AW_ENV[@]}" \
            HOME=/home/arian \
            PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin \
            LD_LIBRARY_PATH="$BUILD/bin:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu" \
            timeout --signal=TERM --kill-after=10s 900 taskset --cpu-list 0-11 \
            "$BUILD/bin/llama-bench" "${BENCH_ARGS[@]}" \
            -p "$TOKENS" -n 0 -r 1 -o json >> "$OUT" 2> "$ERR" || STATUS=$?
        ;;
    bench-base)
        env -i "${BASE_ENV[@]}" \
            HOME=/home/arian \
            PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin \
            LD_LIBRARY_PATH="$BUILD/bin:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu" \
            timeout --signal=TERM --kill-after=10s 900 taskset --cpu-list 0-11 \
            "$BUILD/bin/llama-bench" "${BENCH_ARGS[@]}" \
            -p "$TOKENS" -n 0 -r 1 -o json >> "$OUT" 2> "$ERR" || STATUS=$?
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        exit 2
        ;;
esac
echo "RUN END mode=$MODE status=$STATUS $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$OUT"
cat "$OUT"
if [ "$STATUS" -ne 0 ]; then
    tail -120 "$ERR"
fi
exit "$STATUS"
