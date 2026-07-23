#!/usr/bin/env bash
set -euo pipefail

ROOT=${AW_ROOT:-$(git rev-parse --show-toplevel)}
BUILD=${AW_BUILD:-"$ROOT/build-p100-affinitywave"}
MAP=${AW_MAP:-"$ROOT/docs/backend/affinity-wave/artifacts/placement-hot16.json"}
RESULTS=${AW_RESULTS:-"$ROOT/affinitywave-results"}
XS=${AW_XSESSION_ERRORS:-"$HOME/.xsession-errors"}
TAG=${1:-exact-t64-bf16}

mkdir -p "$RESULTS"

watch_xsession() {
    while true; do
        xs=$(stat -c%s "$XS" 2>/dev/null || echo 0)
        avail=$(df --output=avail / | tail -1 | tr -d ' ')
        if [ "${xs:-0}" -gt 52428800 ] || [ "${avail:-999999999}" -lt 5242880 ]; then
            : > "$XS" 2>/dev/null
            pkill -9 -f "$BUILD/bin/llama-affinity-wave-bench" || true
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
trap 'kill "$WATCHDOG_PID" 2>/dev/null || true' EXIT INT TERM

env -i \
    HOME="$HOME" \
    PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH="$BUILD/bin:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu" \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NCCL_IB_DISABLE=1 \
    NCCL_CUMEM_ENABLE=0 \
    NCCL_ALGO=Ring \
    GGML_CUDA_P2P=1 \
    GGML_CUDA_AFFINITY_WAVE=1 \
    GGML_CUDA_AW_MAP="$MAP" \
    GGML_CUDA_AW_WIRE=bf16 \
    GGML_CUDA_AW_CHECK=1 \
    GGML_CUDA_AW_Q8_LAYOUT=t64k32 \
    GGML_CUDA_AW_DOWN_CACHE=0 \
    GGML_CUDA_AW_M64_SPLIT=2 \
    GGML_CUDA_AW_HOME=layer \
    timeout --signal=TERM --kill-after=10s 300 \
    taskset --cpu-list 0-11 \
    "$BUILD/bin/llama-affinity-wave-bench" \
    --tokens-per-cell 2048 --repeats 12 --route-pattern uniform \
    | tee "$RESULTS/$TAG.log"
