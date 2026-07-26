#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/arian/llama.cpp-q36-moe/.worktrees/stitchrail-20260725
BUILD=/home/arian/llama.cpp-q36-moe/.worktrees/build-p100-stitchrail-20260725
RESULTS="$ROOT/results/qwen36-35b-moe-pp-20260721/stitchrail-20260725"
REFERENCE=/home/arian/llama.cpp-qwen36/results/qwen36-35b-moe-pp-20260721/affinitywave-20260722
FRONTIER=/home/arian/llama.cpp-q36-moe/.worktrees/affinitywave-frontier-20260724/results/qwen36-35b-moe-pp-20260721/affinitywave-frontier-20260724
MODEL=/home/arian/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf
CORPUS=/home/arian/llama.cpp-q36-decodeopt/wikitext-2-raw/wiki.test.raw
XS=/home/arian/.xsession-errors

MODE=${1:?service, bench, diagonal-bench, diagonal-bench-dump, diagonal-ppl-save, diagonal-trace, bench-dump, ppl-save, or trace}
TAG=${2:-qualification}
TOKENS=${3:-2048}
ACTIVE_CELLS=${4:-2}
WIRE=${5:-f32}
PANEL_TOKENS=${6:-$TOKENS}
P100_EXACT=${7:-0}
P100_SUM_WIDTH=${GGML_CUDA_AW_P100_EXACT_SUM_WIDTH:-4}
REPORT="$RESULTS/$TAG"
DUMP_DIR="$RESULTS/$TAG-dumps"
LOGITS="$RESULTS/$TAG-logits-c$TOKENS.bin"

if [ -z "$TAG" ] || [[ "$TAG" == */* ]]; then
    echo "TAG must be a non-empty filename component" >&2
    exit 2
fi

mkdir -p "$RESULTS"
exec 9>/tmp/affinitywave-4gpu.lock
if ! flock -n 9; then
    echo "another AffinityWave job holds /tmp/affinitywave-4gpu.lock" >&2
    exit 75
fi

if pgrep -f '/bin/llama-(bench|perplexity|affinity-wave-bench)( |$)' >/dev/null ||
   pgrep -x nsys >/dev/null; then
    echo "another llama or nsys process is active" >&2
    exit 3
fi

watch_xsession() {
    while true; do
        xs=$(stat -c%s "$XS" 2>/dev/null || echo 0)
        avail=$(df --output=avail / | tail -1 | tr -d ' ')
        if [ "${xs:-0}" -gt 52428800 ] ||
           [ "${avail:-999999999}" -lt 5242880 ]; then
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
    GGML_CUDA_AW_WIRE="$WIRE"
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
    GGML_CUDA_AW_DIAGONAL_SERVICE=legacy
    GGML_CUDA_AW_ROUTE_SCATTER=deterministic
    GGML_CUDA_AW_DIRECT_OWNER=0
    GGML_CUDA_AW_DENSE_T64=0
    GGML_CUDA_AW_DENSE_SELECTORS=0
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
    GGML_CUDA_AW_PRECAPTURE=1
    GGML_CUDA_AW_PRECAPTURE_OUTPUT=1
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
    GGML_CUDA_AW_WAVE_TOKENS="$PANEL_TOKENS"
)

PAIR_ENV=(
    GGML_CUDA_AW_HEADFOLD=1
    GGML_CUDA_AW_HEADFOLD_GDN=1
    GGML_CUDA_AW_HEADFOLD_ATTENTION=1
    GGML_CUDA_AW_HEADFOLD_SPLIT_PRE=0
    GGML_CUDA_AW_PAIRWAVE_MANIFEST="$FRONTIER/pairwave-laneexact-v3.manifest"
    GGML_CUDA_AW_PAIRWAVE_SERVICE=1
    GGML_CUDA_AW_PAIRWAVE_BUNDLE=0
    GGML_CUDA_AW_PAIRWAVE_STATS=0
    GGML_CUDA_AW_PAIRWAVE_M16_CTAS=5
)

DIAGONAL_ENV=(
    GGML_CUDA_AW_DIAGONAL_SERVICE=panel2048
    GGML_CUDA_AW_DENSE_SELECTORS=exact
    GGML_CUDA_AW_P100_EXACT="$P100_EXACT"
    GGML_CUDA_AW_P100_EXACT_SUM_WIDTH="$P100_SUM_WIDTH"
    GGML_CUDA_AW_PAIRWAVE_SERVICE=0
    GGML_CUDA_AW_HEADFOLD=0
    GGML_CUDA_AW_WAVE_TOKENS="$TOKENS"
)

PAIR_EXACT_ENV=(
    GGML_CUDA_AW_P100_EXACT="$P100_EXACT"
)

STATUS=0
echo "RUN START mode=$MODE tag=$TAG tokens=$TOKENS panel=$PANEL_TOKENS active=$ACTIVE_CELLS wire=$WIRE p100_exact=$P100_EXACT $(date -u '+%Y-%m-%d %H:%M:%S UTC')" > "$REPORT.out"
case "$MODE" in
    service)
        env -i "${COMMON_ENV[@]}" "${AW_ENV[@]}" "${PAIR_EXACT_ENV[@]}" \
            timeout --signal=TERM --kill-after=10s 300 \
            taskset --cpu-list 0-11 \
            "$BUILD/bin/llama-affinity-wave-bench" \
            --tokens-per-cell "$TOKENS" --repeats 12 \
            --active-cells "$ACTIVE_CELLS" \
            --route-pattern edge-mix \
            >> "$REPORT.out" 2> "$REPORT.err" || STATUS=$?
        ;;
    bench)
        if [ "$WIRE" != f32 ]; then
            echo "production PairWave qualification requires F32 wire" >&2
            exit 2
        fi
        env -i "${COMMON_ENV[@]}" "${BASE_ENV[@]}" "${AW_ENV[@]}" "${PAIR_ENV[@]}" "${PAIR_EXACT_ENV[@]}" \
            timeout --signal=TERM --kill-after=10s 900 \
            taskset --cpu-list 0-11 "$BUILD/bin/llama-bench" \
            --model "$MODEL" -ngl 99 -sm tensor -fa 1 \
            -b "$TOKENS" -ub "$TOKENS" -mmp 0 \
            -p "$TOKENS" -n 0 -r 1 -o json \
            >> "$REPORT.out" 2> "$REPORT.err" || STATUS=$?
        ;;
    diagonal-bench)
        if [ "$WIRE" != f32 ]; then
            echo "production diagonal qualification requires F32 wire" >&2
            exit 2
        fi
        env -i "${COMMON_ENV[@]}" "${BASE_ENV[@]}" "${AW_ENV[@]}" "${DIAGONAL_ENV[@]}" \
            timeout --signal=TERM --kill-after=10s 900 \
            taskset --cpu-list 0-11 "$BUILD/bin/llama-bench" \
            --model "$MODEL" -ngl 99 -sm tensor -fa 1 \
            -b "$TOKENS" -ub "$TOKENS" -mmp 0 \
            -p "$TOKENS" -n 0 -r 1 -o json \
            >> "$REPORT.out" 2> "$REPORT.err" || STATUS=$?
        ;;
    diagonal-bench-dump)
        if [ "$WIRE" != f32 ]; then
            echo "production diagonal qualification requires F32 wire" >&2
            exit 2
        fi
        mkdir -p "$DUMP_DIR"
        env -i "${COMMON_ENV[@]}" "${BASE_ENV[@]}" "${AW_ENV[@]}" "${DIAGONAL_ENV[@]}" \
            GGML_CUDA_AW_SERVICE_DUMP="$DUMP_DIR" \
            GGML_CUDA_AW_SERVICE_DUMP_LAYER=-1 \
            timeout --signal=TERM --kill-after=10s 900 \
            taskset --cpu-list 0-11 "$BUILD/bin/llama-bench" \
            --model "$MODEL" -ngl 99 -sm tensor -fa 1 \
            -b "$TOKENS" -ub "$TOKENS" -mmp 0 \
            -p "$TOKENS" -n 0 -r 1 -o json \
            >> "$REPORT.out" 2> "$REPORT.err" || STATUS=$?
        ;;
    diagonal-ppl-save)
        if [ "$WIRE" != f32 ]; then
            echo "production diagonal qualification requires F32 wire" >&2
            exit 2
        fi
        env -i "${COMMON_ENV[@]}" "${BASE_ENV[@]}" "${AW_ENV[@]}" "${DIAGONAL_ENV[@]}" \
            timeout --signal=TERM --kill-after=10s 900 \
            taskset --cpu-list 0-11 "$BUILD/bin/llama-perplexity" \
            --model "$MODEL" -ngl 99 -sm tensor \
            -dev CUDA0,CUDA1,CUDA2,CUDA3 -fa on \
            -b "$TOKENS" -ub "$TOKENS" --no-mmap --no-warmup \
            -f "$CORPUS" -c "$TOKENS" --chunks 1 \
            --save-all-logits "$LOGITS" \
            >> "$REPORT.out" 2> "$REPORT.err" || STATUS=$?
        ;;
    bench-dump)
        if [ "$WIRE" != f32 ]; then
            echo "production PairWave qualification requires F32 wire" >&2
            exit 2
        fi
        mkdir -p "$DUMP_DIR"
        env -i "${COMMON_ENV[@]}" "${BASE_ENV[@]}" "${AW_ENV[@]}" "${PAIR_ENV[@]}" "${PAIR_EXACT_ENV[@]}" \
            GGML_CUDA_AW_SERVICE_DUMP="$DUMP_DIR" \
            GGML_CUDA_AW_SERVICE_DUMP_LAYER=-1 \
            timeout --signal=TERM --kill-after=10s 900 \
            taskset --cpu-list 0-11 "$BUILD/bin/llama-bench" \
            --model "$MODEL" -ngl 99 -sm tensor -fa 1 \
            -b "$TOKENS" -ub "$TOKENS" -mmp 0 \
            -p "$TOKENS" -n 0 -r 1 -o json \
            >> "$REPORT.out" 2> "$REPORT.err" || STATUS=$?
        ;;
    ppl-save)
        if [ "$WIRE" != f32 ]; then
            echo "production PairWave qualification requires F32 wire" >&2
            exit 2
        fi
        env -i "${COMMON_ENV[@]}" "${BASE_ENV[@]}" "${AW_ENV[@]}" "${PAIR_ENV[@]}" "${PAIR_EXACT_ENV[@]}" \
            timeout --signal=TERM --kill-after=10s 900 \
            taskset --cpu-list 0-11 "$BUILD/bin/llama-perplexity" \
            --model "$MODEL" -ngl 99 -sm tensor \
            -dev CUDA0,CUDA1,CUDA2,CUDA3 -fa on \
            -b "$TOKENS" -ub "$TOKENS" --no-mmap --no-warmup \
            -f "$CORPUS" -c "$TOKENS" --chunks 1 \
            --save-all-logits "$LOGITS" \
            >> "$REPORT.out" 2> "$REPORT.err" || STATUS=$?
        ;;
    trace)
        if [ "$WIRE" != f32 ]; then
            echo "production PairWave qualification requires F32 wire" >&2
            exit 2
        fi
        env "${COMMON_ENV[@]}" "${BASE_ENV[@]}" "${AW_ENV[@]}" "${PAIR_ENV[@]}" "${PAIR_EXACT_ENV[@]}" \
            GGML_CUDA_AW_TRACE=1 GGML_CUDA_AW_GEMM_MAP=0 \
            timeout --signal=TERM --kill-after=10s 900 \
            nsys profile --trace=cuda,nvtx --sample=none \
            --cpuctxsw=none --cuda-graph-trace=node \
            --force-overwrite=true --output="$REPORT" \
            taskset --cpu-list 0-11 "$BUILD/bin/llama-bench" \
            --model "$MODEL" -ngl 99 -sm tensor -fa 1 \
            -b "$TOKENS" -ub "$TOKENS" -mmp 0 --no-warmup \
            -p "$TOKENS" -n 0 -r 1 -o json \
            >> "$REPORT.out" 2> "$REPORT.err" || STATUS=$?
        if [ "$STATUS" -eq 0 ] && [ -s "$REPORT.nsys-rep" ]; then
            "$FRONTIER/analyze-mapped-trace.py" "$REPORT.nsys-rep" \
                --output-prefix "$REPORT" || STATUS=$?
        fi
        ;;
    diagonal-trace)
        if [ "$WIRE" != f32 ]; then
            echo "production diagonal qualification requires F32 wire" >&2
            exit 2
        fi
        env "${COMMON_ENV[@]}" "${BASE_ENV[@]}" "${AW_ENV[@]}" "${DIAGONAL_ENV[@]}" \
            GGML_CUDA_AW_TRACE=1 GGML_CUDA_AW_GEMM_MAP=1 \
            timeout --signal=TERM --kill-after=10s 900 \
            nsys profile --trace=cuda,nvtx --sample=none \
            --cpuctxsw=none --cuda-graph-trace=node \
            --force-overwrite=true --output="$REPORT" \
            taskset --cpu-list 0-11 "$BUILD/bin/llama-bench" \
            --model "$MODEL" -ngl 99 -sm tensor -fa 1 \
            -b "$TOKENS" -ub "$TOKENS" -mmp 0 --no-warmup \
            -p "$TOKENS" -n 0 -r 1 -o json \
            >> "$REPORT.out" 2> "$REPORT.err" || STATUS=$?
        if [ "$STATUS" -eq 0 ] && [ -s "$REPORT.nsys-rep" ]; then
            "$FRONTIER/analyze-mapped-trace.py" "$REPORT.nsys-rep" \
                --output-prefix "$REPORT" || STATUS=$?
        fi
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        exit 2
        ;;
esac

echo "RUN END mode=$MODE status=$STATUS $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$REPORT.out"
sed -n '1,160p' "$REPORT.out"
if [ "$STATUS" -ne 0 ]; then
    tail -120 "$REPORT.err"
fi
exit "$STATUS"
