#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/arian/llama.cpp-qwen36
BIN="$ROOT/build-p100-affinitywave/bin/llama-bench"
MODEL=/home/arian/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf
RESULTS="$ROOT/results/qwen36-35b-moe-pp-20260721/affinitywave-20260722"
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
    GGML_CUDA_AFFINITY_WAVE=1
    GGML_CUDA_AW_MAP="$MAP"
    GGML_CUDA_AW_WIRE=f32
    GGML_CUDA_AW_CHECK=1
    GGML_CUDA_AW_Q8_LAYOUT=t64k32
    GGML_CUDA_AW_DOWN_CACHE=0
    GGML_CUDA_AW_M64_SPLIT=2
    GGML_CUDA_AW_HOME=layer
)

for scope in all gate-up down; do
    out="$RESULTS/phase2a-scope-$scope.json"
    err="$RESULTS/phase2a-scope-$scope.err"
    echo "SCOPE START $scope $(date -u '+%H:%M:%S UTC')"
    env -i "${BASE_ENV[@]}" GGML_CUDA_AW_T64_SCOPE="$scope" \
        HOME=/home/arian PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin \
        LD_LIBRARY_PATH="$ROOT/build-p100-affinitywave/bin:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu" \
        timeout --signal=TERM --kill-after=10s 1200 taskset --cpu-list 0-11 "$BIN" \
        --model "$MODEL" -ngl 99 -sm tensor -fa 1 -b 2048 -ub 2048 -mmp 0 \
        -p 2048 -n 64 -r 2 -o json > "$out" 2> "$err" || exit $?
    python3 - "$scope" "$out" <<'PY'
import json, sys
scope, path = sys.argv[1:]
data = json.load(open(path))
print(scope, " ".join(f"pp{x['n_prompt']}={x['avg_ts']:.1f}" if x['n_prompt'] else f"tg{x['n_gen']}={x['avg_ts']:.1f}" for x in data))
PY
done
