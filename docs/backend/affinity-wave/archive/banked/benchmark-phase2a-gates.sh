#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/arian/llama.cpp-qwen36
RESULTS="$ROOT/results/qwen36-35b-moe-pp-20260721/affinitywave-20260722"
BIN="$ROOT/build-p100-affinitywave/bin/llama-bench"
CMP="$ROOT/build-p100-affinitywave/bin/llama-completion"
PPL="$ROOT/build-p100-affinitywave/bin/llama-perplexity"
MODEL=/home/arian/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf
CORPUS=/home/arian/llama.cpp-q36-decodeopt/wikitext-2-raw/wiki.test.raw
MAP="$RESULTS/placement-hot16.json"
XS=/home/arian/.xsession-errors

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
    GGML_META_SUBMIT_THREADS=1
)
AW_ENV=(
    GGML_CUDA_AFFINITY_WAVE=1
    GGML_CUDA_AW_MAP="$MAP"
    GGML_CUDA_AW_WIRE=f32
    GGML_CUDA_AW_CHECK=1
    GGML_CUDA_AW_Q8_LAYOUT=t64k32
    GGML_CUDA_AW_DOWN_CACHE=0
    GGML_CUDA_AW_M64_SPLIT=2
    GGML_CUDA_AW_HOME=layer
)
COMMON_ENV=(
    HOME=/home/arian
    PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin
    LD_LIBRARY_PATH="$ROOT/build-p100-affinitywave/bin:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu"
)

run_completion() {
    arm=$1
    shift
    env -i "${BASE_ENV[@]}" "$@" "${COMMON_ENV[@]}" timeout --signal=TERM --kill-after=10s 900 \
        taskset --cpu-list 0-11 "$CMP" --model "$MODEL" -ngl 99 -sm tensor \
        -dev CUDA0,CUDA1,CUDA2,CUDA3 -fa on -c 2048 -b 512 -ub 512 --no-mmap \
        --no-display-prompt --temp 0 --top-k 1 -s 1234 -n 32 -p Hello </dev/null \
        > "$RESULTS/phase2a-mmvq-$arm.out" 2> "$RESULTS/phase2a-mmvq-$arm.err"
}

echo "MMVQ NATIVE START $(date -u '+%H:%M:%S UTC')"
run_completion native || exit $?
echo "MMVQ T64 START $(date -u '+%H:%M:%S UTC')"
run_completion t64 "${AW_ENV[@]}" || exit $?
if cmp -s "$RESULTS/phase2a-mmvq-native.out" "$RESULTS/phase2a-mmvq-t64.out"; then
    echo "MMVQ BYTE-IDENTICAL bytes=$(wc -c < "$RESULTS/phase2a-mmvq-t64.out")"
else
    echo "MMVQ MISMATCH"
    diff -u "$RESULTS/phase2a-mmvq-native.out" "$RESULTS/phase2a-mmvq-t64.out" | head -80
    exit 4
fi

run_bench() {
    arm=$1
    shift
    echo "BENCH $arm START $(date -u '+%H:%M:%S UTC')"
    env -i "${BASE_ENV[@]}" "$@" "${COMMON_ENV[@]}" timeout --signal=TERM --kill-after=10s 1800 \
        taskset --cpu-list 0-11 "$BIN" --model "$MODEL" -ngl 99 -sm tensor -fa 1 \
        -b 8192 -ub 4096 -mmp 0 -p 2048,8192 -n 128 -r 4 -o json \
        > "$RESULTS/phase2a-bench-$arm.json" 2> "$RESULTS/phase2a-bench-$arm.err"
}

run_bench native || exit $?
run_bench t64 "${AW_ENV[@]}" || exit $?
python3 - "$RESULTS" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
for arm in ("native", "t64"):
    data = json.loads((p/f"phase2a-bench-{arm}.json").read_text())
    print(arm, " ".join((f"pp{x['n_prompt']}={x['avg_ts']:.1f}" if x['n_prompt'] else f"tg{x['n_gen']}={x['avg_ts']:.1f}") for x in data))
PY

echo "PPL T64 START $(date -u '+%H:%M:%S UTC')"
env -i "${BASE_ENV[@]}" "${AW_ENV[@]}" "${COMMON_ENV[@]}" timeout --signal=TERM --kill-after=10s 2400 \
    taskset --cpu-list 0-11 "$PPL" --model "$MODEL" -f "$CORPUS" -ngl 99 -sm tensor \
    -dev CUDA0,CUDA1,CUDA2,CUDA3 -fa on -c 4096 -b 4096 -ub 4096 --no-mmap --chunks 8 \
    > "$RESULTS/phase2a-ppl-t64.out" 2> "$RESULTS/phase2a-ppl-t64.err"
status=$?
grep -hiE 'Final estimate' "$RESULTS/phase2a-ppl-t64.out" "$RESULTS/phase2a-ppl-t64.err" | tail -1
echo "PHASE2A GATES END status=$status $(date -u '+%H:%M:%S UTC')"
exit "$status"
