#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}
RESULTS=${RESULTS:-"$ROOT/results/qwen36-35b-moe-pp-20260721/diagonal-wave-p100-2900"}
SOURCE=${SOURCE:-"$SCRIPT_DIR/p100-stacked-projection-probe.cu"}
BINARY=${BINARY:-"$RESULTS/p100-stacked-projection-probe"}
COORDINATION=${COORDINATION:-"$ROOT/bench/COORDINATION-diagonal-wave-p100-2900.md"}
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-12.8}
NVCC="$CUDA_ROOT/bin/nvcc"
RUN_HOME=${RUN_HOME:-${HOME:-/home/arian}}
XS=${XS:-"$RUN_HOME/.xsession-errors"}
CXX=${CXX:-/usr/bin/g++-14}
SYSTEM_LIB=${SYSTEM_LIB:-/usr/lib/x86_64-linux-gnu}
CPU_LIST=${CPU_LIST:-0-11}

TAG=${1:-p100-stacked-m512-n2032-k2048-a3}
BASE_M=${2:-512}
N=${3:-2032}
K=${4:-2048}
BASELINE_ALGORITHM=${5:-3}
WARMUP=${6:-5}
REPEATS=${7:-30}
PHYSICAL_DEVICE=${8:-0}
REPORT="$RESULTS/$TAG"
TEMP_BINARY="$BINARY.$$"

usage() {
    echo "usage: $0 [tag [base-M [N [K [baseline-algorithm [warmup [repeats [physical-device]]]]]]]]" >&2
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

is_exact_algorithm() {
    local algorithm=$1
    if [ "$algorithm" = -1 ]; then
        return 0
    fi
    if ! is_nonnegative_integer "$algorithm"; then
        return 1
    fi
    [ "$algorithm" -le 23 ] ||
        { [ "$algorithm" -ge 99 ] && [ "$algorithm" -le 115 ]; }
}

if [ -z "$TAG" ] || [[ "$TAG" == */* ]]; then
    echo "tag must be a non-empty filename component" >&2
    exit 2
fi
if ! is_positive_integer "$BASE_M" ||
   ! is_positive_integer "$N" ||
   ! is_positive_integer "$K" ||
   ! is_exact_algorithm "$BASELINE_ALGORITHM" ||
   ! is_nonnegative_integer "$WARMUP" ||
   ! is_positive_integer "$REPEATS" ||
   ! is_nonnegative_integer "$PHYSICAL_DEVICE"; then
    usage
    exit 2
fi
if [ ! -f "$SOURCE" ] ||
   [ ! -x "$NVCC" ] ||
   [ ! -x "$CXX" ] ||
   [ ! -x /usr/bin/nvidia-smi ] ||
   [ ! -f "$COORDINATION" ]; then
    echo "probe source, toolchain, nvidia-smi, or coordination file is missing" >&2
    exit 2
fi
if [ -e "$REPORT.jsonl" ] ||
   [ -e "$REPORT.err" ] ||
   [ -e "$REPORT.build.log" ]; then
    echo "tag output already exists: $TAG" >&2
    exit 2
fi

cleanup_temp() {
    rm -f "$TEMP_BINARY"
}
trap cleanup_temp EXIT

"$NVCC" \
    -O3 \
    -std=c++17 \
    -arch=sm_60 \
    -ccbin "$CXX" \
    -lineinfo \
    -Xptxas=-v \
    "$SOURCE" \
    -lcublas \
    -o "$TEMP_BINARY" \
    > "$REPORT.build.log" 2>&1
BUILD_STATUS=$?
if [ "$BUILD_STATUS" -ne 0 ]; then
    echo "probe build failed; see $REPORT.build.log" >&2
    exit "$BUILD_STATUS"
fi
if ! mv "$TEMP_BINARY" "$BINARY"; then
    echo "could not install probe binary: $BINARY" >&2
    exit 1
fi

exec 9>/tmp/affinitywave-4gpu.lock
if ! flock -n 9; then
    echo "another AffinityWave job holds /tmp/affinitywave-4gpu.lock" >&2
    exit 75
fi

if pgrep -f '/bin/llama-(bench|perplexity|affinity-wave-bench)( |$)' >/dev/null ||
   pgrep -x nsys >/dev/null ||
   pgrep -f 'p100-(router-selector|stacked-projection)-probe --' >/dev/null; then
    echo "another llama, nsys, or selector probe process is active" >&2
    exit 3
fi

COMPUTE_PIDS=$(
    /usr/bin/nvidia-smi \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null |
        awk '$1 ~ /^[0-9]+$/ { print $1 }'
)
if [ -n "$COMPUTE_PIDS" ]; then
    echo "another CUDA compute process is active: $COMPUTE_PIDS" >&2
    exit 3
fi

START_TIME=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
printf '\n%s: starting locked P100 stacked projection probe tag=%s ' \
    "$START_TIME" "$TAG" >> "$COORDINATION"
printf 'base_M=%s stacked_M=%s N=%s K=%s baseline_algorithm=%s ' \
    "$BASE_M" "$((2*BASE_M))" "$N" "$K" "$BASELINE_ALGORITHM" \
    >> "$COORDINATION"
printf 'warmup=%s repeats=%s physical_device=%s.\n' \
    "$WARMUP" "$REPEATS" "$PHYSICAL_DEVICE" >> "$COORDINATION"

watch_xsession() {
    while true; do
        xs=$(stat -c%s "$XS" 2>/dev/null || echo 0)
        avail=$(df --output=avail / | tail -1 | tr -d ' ')
        if [ "${xs:-0}" -gt 52428800 ] ||
           [ "${avail:-999999999}" -lt 5242880 ]; then
            echo "WATCHDOG TRIP $(date -u '+%H:%M:%S UTC')" >&2
            truncate -s 0 "$XS" 2>/dev/null || true
            pkill -9 -f 'p100-stacked-projection-probe --' || true
            break
        fi
        sleep 2
    done
}

watch_xsession 9>&- 2>> "$REPORT.err" &
WATCHDOG_PID=$!
cleanup() {
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
    cleanup_temp
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

COMMON_ENV=(
    HOME="$RUN_HOME"
    PATH="$CUDA_ROOT/bin:/usr/bin:/bin"
    LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$SYSTEM_LIB"
    CUDA_DEVICE_ORDER=PCI_BUS_ID
    CUDA_VISIBLE_DEVICES="$PHYSICAL_DEVICE"
    LC_ALL=C
)

STATUS=0
env -i "${COMMON_ENV[@]}" \
    timeout --signal=TERM --kill-after=10s 300 \
    taskset --cpu-list "$CPU_LIST" \
    "$BINARY" \
    --base-m "$BASE_M" \
    --n "$N" \
    --k "$K" \
    --baseline-algorithm "$BASELINE_ALGORITHM" \
    --warmup "$WARMUP" \
    --repeats "$REPEATS" \
    --device 0 \
    > "$REPORT.jsonl" 2>> "$REPORT.err" || STATUS=$?

END_TIME=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
printf '%s: stacked projection probe tag=%s completed status=%s; ' \
    "$END_TIME" "$TAG" "$STATUS" >> "$COORDINATION"
printf 'outputs=%s.jsonl,%s.err,%s.build.log.\n' \
    "$REPORT" "$REPORT" "$REPORT" >> "$COORDINATION"

exit "$STATUS"
