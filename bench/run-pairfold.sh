#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/arian/llama.cpp-q36-moe/.worktrees/paircache-20260726
BUILD=/home/arian/llama.cpp-q36-moe/.worktrees/build-p100-paircache-20260726
RESULTS="$ROOT/results/qwen36-35b-moe-pp-20260721/pairfold-20260726"
REFERENCE=/home/arian/llama.cpp-qwen36/results/qwen36-35b-moe-pp-20260721/affinitywave-20260722
MODEL=/home/arian/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf
CORPUS=/home/arian/llama.cpp-q36-decodeopt/wikitext-2-raw/wiki.test.raw
MANIFEST="$ROOT/docs/p100-pairwave-project/evidence/frontier/pairwave-laneexact-v3.manifest"
XS=/home/arian/.xsession-errors

MODE=${1:?serial, pipeline, diagonal, diagonal-dump, serial-ppl, pipeline-ppl, serial-dump, pipeline-dump, or trace}
TOKENS=${2:-512}
TAG=${3:-"$MODE-c$TOKENS"}
DUMP_LAYER=${4:-0}
OUT="$RESULTS/$TAG"
REPS=${PAIRFOLD_REPS:-1}
HOST_WEIGHTS=${PAIRFOLD_HOST_WEIGHTS:-${GGML_CUDA_AW_PAIRFOLD_HOST_WEIGHTS:-0}}
HOST_LAYERS=${PAIRFOLD_HOST_LAYERS:-${GGML_CUDA_AW_PAIRFOLD_HOST_LAYERS:-16:23}}
HOST_PLACEMENT=${PAIRFOLD_HOST_PLACEMENT:-${GGML_CUDA_AW_PAIRFOLD_HOST_PLACEMENT:-0}}
HOST_LOOKAHEAD=${PAIRFOLD_HOST_LOOKAHEAD:-${GGML_CUDA_AW_PAIRFOLD_HOST_LOOKAHEAD:-0}}
HOST_PHASED=${PAIRFOLD_HOST_PHASED:-${GGML_CUDA_AW_PAIRFOLD_HOST_PHASED:-0}}
HOST_SERIAL_H2D=${PAIRFOLD_HOST_SERIAL_H2D:-${GGML_CUDA_AW_PAIRFOLD_HOST_SERIAL_H2D:-0}}

if [[ ! "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "invalid tag: $TAG" >&2
    exit 2
fi
if [[ "$TOKENS" != 512 && "$TOKENS" != 2048 && "$TOKENS" != 8128 ]]; then
    echo "PairFold qualification supports only c512, pp2048, and pp8128" >&2
    exit 2
fi
if [[ ! "$REPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "PAIRFOLD_REPS must be a positive integer" >&2
    exit 2
fi
if [[ "$HOST_WEIGHTS" != 0 && "$HOST_WEIGHTS" != 1 ]]; then
    echo "PAIRFOLD_HOST_WEIGHTS must be 0 or 1" >&2
    exit 2
fi
if [[ "$HOST_PLACEMENT" != 0 && "$HOST_PLACEMENT" != 1 ]]; then
    echo "PAIRFOLD_HOST_PLACEMENT must be 0 or 1" >&2
    exit 2
fi
if [[ "$HOST_PLACEMENT" == 1 && "$HOST_WEIGHTS" != 1 ]]; then
    echo "PAIRFOLD_HOST_PLACEMENT=1 requires PAIRFOLD_HOST_WEIGHTS=1" >&2
    exit 2
fi
if [[ "$HOST_PLACEMENT" == 1 &&
      ( "$MODE" == diagonal || "$MODE" == diagonal-dump ) ]]; then
    echo "PAIRFOLD_HOST_PLACEMENT=1 requires a PairFold mode" >&2
    exit 2
fi
if [[ "$HOST_LOOKAHEAD" != 0 && "$HOST_LOOKAHEAD" != 1 ]]; then
    echo "PAIRFOLD_HOST_LOOKAHEAD must be 0 or 1" >&2
    exit 2
fi
if [[ "$HOST_PHASED" != 0 && "$HOST_PHASED" != 1 ]]; then
    echo "PAIRFOLD_HOST_PHASED must be 0 or 1" >&2
    exit 2
fi
if [[ "$HOST_SERIAL_H2D" != 0 && "$HOST_SERIAL_H2D" != 1 ]]; then
    echo "PAIRFOLD_HOST_SERIAL_H2D must be 0 or 1" >&2
    exit 2
fi
if [[ ! "$HOST_LAYERS" =~ ^([0-9]|[1-3][0-9]):([0-9]|[1-3][0-9])$ ]]; then
    echo "PAIRFOLD_HOST_LAYERS must be FIRST:LAST within 0:39" >&2
    exit 2
fi
HOST_FIRST=${HOST_LAYERS%%:*}
HOST_LAST=${HOST_LAYERS##*:}
if (( HOST_FIRST > HOST_LAST )); then
    echo "PAIRFOLD_HOST_LAYERS must be an ascending range" >&2
    exit 2
fi

mkdir -p "$RESULTS"
exec 9>/tmp/affinitywave-4gpu.lock
if ! flock -n 9; then
    echo "another four-GPU job holds /tmp/affinitywave-4gpu.lock" >&2
    exit 75
fi
if pgrep -f '/bin/llama-(bench|perplexity|cli|completion|server)( |$)' >/dev/null ||
   pgrep -x nsys >/dev/null ||
   pgrep -x ncu >/dev/null; then
    echo "another llama or profiler process is active" >&2
    exit 3
fi

watch_xsession() {
    while true; do
        local xs
        local avail
        xs=$(stat -c%s "$XS" 2>/dev/null || echo 0)
        avail=$(df --output=avail / | tail -1 | tr -d ' ')
        if [[ "${xs:-0}" -gt 52428800 ||
              "${avail:-999999999}" -lt 5242880 ]]; then
            echo "WATCHDOG TRIP $(date -u '+%H:%M:%S UTC')" >&2
            truncate -s 0 "$XS" 2>/dev/null || true
            pkill -9 -f '/bin/llama-' || true
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

COMMON_ENV=(
    HOME=/home/arian
    PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin
    LD_LIBRARY_PATH="$BUILD/bin:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu"
    CUDA_VISIBLE_DEVICES=0,1,2,3
)
BASE_ENV=(
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
)
AW_ENV=(
    GGML_CUDA_AFFINITY_WAVE=1
    GGML_CUDA_AW_MAP="$REFERENCE/placement-hot16.json"
    GGML_CUDA_AW_WIRE=f32
    GGML_CUDA_AW_PARTIAL=bf16
    GGML_CUDA_AW_NCCL_SUM=0
    GGML_CUDA_AW_NCCL_ORDER_SUM=1
    GGML_CUDA_AW_CHECK=1
    GGML_CUDA_AW_Q8_LAYOUT=t64k32
    GGML_CUDA_AW_Q8_KERNEL=interleave
    GGML_CUDA_AW_Q8_ENGINE=cohortrail
    GGML_CUDA_AW_COHORTRAIL_P2_CTAS=3
    GGML_CUDA_AW_COHORTRAIL_SINGLE_CTAS=3
    GGML_CUDA_AW_SERVICE=legacy
    GGML_CUDA_AW_DIAGONAL_SERVICE=panel2048
    GGML_CUDA_AW_ROUTE_SCATTER=deterministic
    GGML_CUDA_AW_DIRECT_OWNER=0
    GGML_CUDA_AW_DENSE_T64=0
    GGML_CUDA_AW_DENSE_SELECTORS=exact
    GGML_CUDA_AW_DOWN_CACHE=0
    GGML_CUDA_AW_M64_SPLIT=2
    GGML_CUDA_AW_HOME=layer
    GGML_CUDA_AW_DECODE=t64
    GGML_CUDA_AW_WAVE_DRY=0
    GGML_CUDA_AW_WAVE_DENSE=1
    GGML_CUDA_AW_WAVE_TOKEN_SPLIT=1
    GGML_CUDA_AW_WAVE_TOKEN_PLAN=0
    GGML_CUDA_AW_WAVE_DENSE_BENCH=service
    GGML_CUDA_AW_WAVE_OUTPUT=1
    GGML_CUDA_AW_LANE_STAGGER=1
    GGML_CUDA_AW_CORRIDOR_EARLY=all
    GGML_CUDA_AW_CORRIDOR_STATE_SPLIT=1
    GGML_CUDA_AW_PRECAPTURE=0
    GGML_CUDA_AW_SHARED_SERVICE=0
    GGML_CUDA_AW_LANE_BALANCE=0
    GGML_CUDA_AW_DEBUG_NODE_SYNC=0
    GGML_CUDA_AW_DEBUG_LIVE_SYNC=0
    GGML_CUDA_AW_GROUP_CELLS=1
    GGML_CUDA_AW_GROUP_PATTERN=1111
    GGML_CUDA_AW_GDN_CHUNKED=2
    GGML_CUDA_AW_GDN_WARPS=4
    GGML_CUDA_AW_GDN_WY_GRAPH=0
    GGML_CUDA_AW_FUSED_GATE_UP=0
    GGML_CUDA_AW_FA_QUERY_TILE=4
    GGML_CUDA_AW_WAVE_TOKENS="$TOKENS"
    GGML_CUDA_AW_P100_EXACT=1
    GGML_CUDA_AW_P100_EXACT_SUM_WIDTH=4
    GGML_CUDA_AW_MEMORY=1
)
PAIR_ENV=(
    GGML_CUDA_AW_HEADFOLD=1
    GGML_CUDA_AW_HEADFOLD_GDN=1
    GGML_CUDA_AW_HEADFOLD_ATTENTION=1
    GGML_CUDA_AW_HEADFOLD_SPLIT_PRE=1
    GGML_CUDA_AW_PAIRWAVE_MANIFEST="$MANIFEST"
    GGML_CUDA_AW_PAIRWAVE_SERVICE=0
    GGML_CUDA_AW_PAIRWAVE_BUNDLE=0
    GGML_CUDA_AW_PAIRWAVE_STATS=0
    GGML_CUDA_AW_PAIRWAVE_M16_CTAS=5
    GGML_CUDA_AW_PAIRFOLD=1
    GGML_CUDA_AW_PAIRFOLD_HOST_WEIGHTS="$HOST_WEIGHTS"
    GGML_CUDA_AW_PAIRFOLD_HOST_LAYERS="$HOST_LAYERS"
    GGML_CUDA_AW_PAIRFOLD_HOST_PLACEMENT="$HOST_PLACEMENT"
    GGML_CUDA_AW_PAIRFOLD_HOST_LOOKAHEAD="$HOST_LOOKAHEAD"
    GGML_CUDA_AW_PAIRFOLD_HOST_PHASED="$HOST_PHASED"
    GGML_CUDA_AW_PAIRFOLD_HOST_SERIAL_H2D="$HOST_SERIAL_H2D"
)
DIAGONAL_ENV=(
    GGML_CUDA_AW_PAIRFOLD=0
    GGML_CUDA_AW_PAIRWAVE_SERVICE=0
    GGML_CUDA_AW_HEADFOLD=0
)

PAIR_MODE_ENV=()
case "$MODE" in
    serial|serial-ppl|serial-dump)
        PAIR_MODE_ENV=(GGML_CUDA_AW_PAIRFOLD_SERIAL=1)
        ;;
    pipeline|pipeline-ppl|pipeline-dump|trace)
        PAIR_MODE_ENV=(GGML_CUDA_AW_PAIRFOLD_SERIAL=0)
        ;;
    diagonal|diagonal-dump)
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        exit 2
        ;;
esac

echo "RUN START mode=$MODE tokens=$TOKENS $(date -u '+%Y-%m-%d %H:%M:%S UTC')" > "$OUT.out"
STATUS=0
if [[ "$MODE" == diagonal ]]; then
    env -i "${COMMON_ENV[@]}" "${BASE_ENV[@]}" "${AW_ENV[@]}" \
        "${DIAGONAL_ENV[@]}" \
        timeout --signal=TERM --kill-after=10s 900 \
        taskset --cpu-list 0-11 "$BUILD/bin/llama-bench" \
        --model "$MODEL" -ngl 99 -sm tensor -fa 1 \
        -b "$TOKENS" -ub "$TOKENS" -mmp 0 --no-warmup \
        -p "$TOKENS" -n 0 -r "$REPS" -o json \
        >> "$OUT.out" 2> "$OUT.err" || STATUS=$?
elif [[ "$MODE" == diagonal-dump ]]; then
    if [[ "$DUMP_LAYER" != -1 ]] &&
       { [[ ! "$DUMP_LAYER" =~ ^[0-9]+$ ]] ||
         [[ "$DUMP_LAYER" -gt 39 ]]; }; then
        echo "dump layer must be -1 or between 0 and 39" >&2
        exit 2
    fi
    mkdir -p "$OUT-dumps"
    env -i "${COMMON_ENV[@]}" "${BASE_ENV[@]}" "${AW_ENV[@]}" \
        "${DIAGONAL_ENV[@]}" \
        GGML_CUDA_AW_SERVICE_DUMP="$OUT-dumps" \
        GGML_CUDA_AW_SERVICE_DUMP_LAYER="$DUMP_LAYER" \
        timeout --signal=TERM --kill-after=10s 900 \
        taskset --cpu-list 0-11 "$BUILD/bin/llama-bench" \
        --model "$MODEL" -ngl 99 -sm tensor -fa 1 \
        -b "$TOKENS" -ub "$TOKENS" -mmp 0 --no-warmup \
        -p "$TOKENS" -n 0 -r "$REPS" -o json \
        >> "$OUT.out" 2> "$OUT.err" || STATUS=$?
elif [[ "$MODE" == serial-ppl || "$MODE" == pipeline-ppl ]]; then
    env -i "${COMMON_ENV[@]}" "${BASE_ENV[@]}" "${AW_ENV[@]}" \
        "${PAIR_ENV[@]}" "${PAIR_MODE_ENV[@]}" \
        timeout --signal=TERM --kill-after=10s 900 \
        taskset --cpu-list 0-11 "$BUILD/bin/llama-perplexity" \
        --model "$MODEL" -ngl 99 -sm tensor \
        -dev CUDA0,CUDA1,CUDA2,CUDA3 -fa on \
        -b "$TOKENS" -ub "$TOKENS" --no-mmap --no-warmup \
        -f "$CORPUS" -c "$TOKENS" --chunks 1 \
        --save-all-logits "$OUT-logits.bin" \
        >> "$OUT.out" 2> "$OUT.err" || STATUS=$?
elif [[ "$MODE" == serial-dump || "$MODE" == pipeline-dump ]]; then
    if [[ "$DUMP_LAYER" != -1 ]] &&
       { [[ ! "$DUMP_LAYER" =~ ^[0-9]+$ ]] ||
         [[ "$DUMP_LAYER" -gt 39 ]]; }; then
        echo "dump layer must be -1 or between 0 and 39" >&2
        exit 2
    fi
    mkdir -p "$OUT-dumps"
    env -i "${COMMON_ENV[@]}" "${BASE_ENV[@]}" "${AW_ENV[@]}" \
        "${PAIR_ENV[@]}" "${PAIR_MODE_ENV[@]}" \
        GGML_CUDA_AW_SERVICE_DUMP="$OUT-dumps" \
        GGML_CUDA_AW_SERVICE_DUMP_LAYER="$DUMP_LAYER" \
        GGML_CUDA_AW_PAIRFOLD_AUDIT_LAYER="$DUMP_LAYER" \
        timeout --signal=TERM --kill-after=10s 900 \
        taskset --cpu-list 0-11 "$BUILD/bin/llama-bench" \
        --model "$MODEL" -ngl 99 -sm tensor -fa 1 \
        -b "$TOKENS" -ub "$TOKENS" -mmp 0 --no-warmup \
        -p "$TOKENS" -n 0 -r "$REPS" -o json \
        >> "$OUT.out" 2> "$OUT.err" || STATUS=$?
elif [[ "$MODE" == trace ]]; then
    env "${COMMON_ENV[@]}" "${BASE_ENV[@]}" "${AW_ENV[@]}" \
        "${PAIR_ENV[@]}" "${PAIR_MODE_ENV[@]}" \
        GGML_CUDA_AW_TRACE=1 GGML_CUDA_AW_GEMM_MAP=1 \
        timeout --signal=TERM --kill-after=10s 900 \
        nsys profile --trace=cuda,nvtx --sample=none \
        --cpuctxsw=none --cuda-graph-trace=node \
        --force-overwrite=true --output="$OUT" \
        taskset --cpu-list 0-11 "$BUILD/bin/llama-bench" \
        --model "$MODEL" -ngl 99 -sm tensor -fa 1 \
        -b "$TOKENS" -ub "$TOKENS" -mmp 0 --no-warmup \
        -p "$TOKENS" -n 0 -r "$REPS" -o json \
        >> "$OUT.out" 2> "$OUT.err" || STATUS=$?
else
    env -i "${COMMON_ENV[@]}" "${BASE_ENV[@]}" "${AW_ENV[@]}" \
        "${PAIR_ENV[@]}" "${PAIR_MODE_ENV[@]}" \
        timeout --signal=TERM --kill-after=10s 900 \
        taskset --cpu-list 0-11 "$BUILD/bin/llama-bench" \
        --model "$MODEL" -ngl 99 -sm tensor -fa 1 \
        -b "$TOKENS" -ub "$TOKENS" -mmp 0 --no-warmup \
        -p "$TOKENS" -n 0 -r "$REPS" -o json \
        >> "$OUT.out" 2> "$OUT.err" || STATUS=$?
fi
echo "RUN END mode=$MODE status=$STATUS $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$OUT.out"
tail -80 "$OUT.out"
if [[ "$STATUS" -ne 0 ]]; then
    tail -160 "$OUT.err"
fi
exit "$STATUS"
