#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}
RESULTS=${RESULTS:-"$ROOT/results/qwen36-35b-moe-pp-20260721/diagonal-wave-p100-2900"}
SOURCE=${SOURCE:-"$SCRIPT_DIR/p100-router-selector-probe.cu"}
BINARY=${BINARY:-"$RESULTS/p100-router-selector-probe"}
COORDINATION=${COORDINATION:-"$ROOT/bench/COORDINATION-diagonal-wave-p100-2900.md"}
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-12.8}
NVCC="$CUDA_ROOT/bin/nvcc"
RUN_HOME=${RUN_HOME:-${HOME:-/home/arian}}
XS=${XS:-"$RUN_HOME/.xsession-errors"}
CXX=${CXX:-/usr/bin/g++-14}
SYSTEM_LIB=${SYSTEM_LIB:-/usr/lib/x86_64-linux-gnu}
CPU_LIST=${CPU_LIST:-0-11}

TAG=${1:-p100-router-selector-m256-n2032-k2048}
M=${2:-256}
N=${3:-2032}
K=${4:-2048}
WARMUP=${5:-5}
REPEATS=${6:-30}
PHYSICAL_DEVICE=${7:-0}
REPORT="$RESULTS/$TAG"
TEMP_BINARY="$BINARY.$$"

usage() {
    echo "usage: $0 [tag [M [N [K [warmup [repeats [physical-device]]]]]]]" >&2
}

is_positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

if [ -z "$TAG" ] || [[ "$TAG" == */* ]]; then
    echo "tag must be a non-empty filename component" >&2
    exit 2
fi
if ! is_positive_integer "$M" ||
   ! is_positive_integer "$N" ||
   ! is_positive_integer "$K" ||
   ! is_nonnegative_integer "$WARMUP" ||
   ! is_positive_integer "$REPEATS" ||
   ! is_nonnegative_integer "$PHYSICAL_DEVICE"; then
    usage
    exit 2
fi
if [ ! -f "$SOURCE" ] ||
   [ ! -x "$NVCC" ] ||
   [ ! -x "$CXX" ]; then
    echo "probe source or CUDA toolchain is missing" >&2
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
   pgrep -f 'p100-router-selector-probe --' >/dev/null; then
    echo "another llama, nsys, or router probe process is active" >&2
    exit 3
fi

START_TIME=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
printf '\n%s: starting locked P100 router selector probe tag=%s ' \
    "$START_TIME" "$TAG" >> "$COORDINATION"
printf 'M=%s N=%s K=%s warmup=%s repeats=%s physical_device=%s.\n' \
    "$M" "$N" "$K" "$WARMUP" "$REPEATS" "$PHYSICAL_DEVICE" \
    >> "$COORDINATION"

watch_xsession() {
    while true; do
        xs=$(stat -c%s "$XS" 2>/dev/null || echo 0)
        avail=$(df --output=avail / | tail -1 | tr -d ' ')
        if [ "${xs:-0}" -gt 52428800 ] ||
           [ "${avail:-999999999}" -lt 5242880 ]; then
            echo "WATCHDOG TRIP $(date -u '+%H:%M:%S UTC')" >&2
            truncate -s 0 "$XS" 2>/dev/null || true
            pkill -9 -f 'p100-router-selector-probe --' || true
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
trap cleanup EXIT INT TERM

COMMON_ENV=(
    HOME="$RUN_HOME"
    PATH="$CUDA_ROOT/bin:/usr/bin:/bin"
    LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$SYSTEM_LIB"
    CUDA_VISIBLE_DEVICES="$PHYSICAL_DEVICE"
    LC_ALL=C
)

STATUS=0
env -i "${COMMON_ENV[@]}" \
    timeout --signal=TERM --kill-after=10s 300 \
    taskset --cpu-list "$CPU_LIST" \
    "$BINARY" \
    --m "$M" \
    --n "$N" \
    --k "$K" \
    --warmup "$WARMUP" \
    --repeats "$REPEATS" \
    --device 0 \
    > "$REPORT.jsonl" 2>> "$REPORT.err" || STATUS=$?

END_TIME=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
printf '%s: router selector probe tag=%s completed status=%s; ' \
    "$END_TIME" "$TAG" "$STATUS" >> "$COORDINATION"
printf 'outputs=%s.jsonl,%s.err,%s.build.log.\n' \
    "$REPORT" "$REPORT" "$REPORT" >> "$COORDINATION"

exit "$STATUS"
