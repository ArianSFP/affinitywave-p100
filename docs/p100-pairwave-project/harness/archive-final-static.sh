#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/arian/llama.cpp-q36-moe/.worktrees/stitchrail-20260725
BUILD=/home/arian/llama.cpp-q36-moe/.worktrees/build-p100-stitchrail-20260725
RESULTS="$ROOT/results/qwen36-35b-moe-pp-20260721/stitchrail-20260725"
OUT="$RESULTS/p100-exact-final-static-20260726"
CUDA=/usr/local/cuda-12.8/bin
LIB="$BUILD/bin/libggml-cuda.so.0.15.3"
COMPILE_DB="$BUILD/compile_commands.json"

mkdir -p "$OUT/cubins" "$OUT/ptxas-objects"

git -C "$ROOT" rev-parse HEAD > "$OUT/git-head.txt"
git -C "$ROOT" status --short > "$OUT/git-status.txt"
git -C "$ROOT" diff --binary > "$OUT/source.diff"
sha256sum \
    "$OUT/source.diff" \
    "$LIB" \
    "$ROOT/ggml/src/ggml-cuda/affinity-wave.cu" \
    "$ROOT/ggml/src/ggml-cuda/cpy.cu" \
    "$ROOT/ggml/src/ggml-cuda/gated_delta_net.cu" \
    "$ROOT/ggml/src/ggml-cuda/ggml-cuda.cu" \
    > "$OUT/sha256.txt"

{
    "$CUDA/nvcc" --version
    "$CUDA/cuobjdump" --version
    "$CUDA/nvdisasm" --version
    /usr/bin/g++-14 --version
} > "$OUT/toolchain.txt"

"$CUDA/cuobjdump" --dump-resource-usage --gpu-architecture sm_60 \
    "$LIB" > "$OUT/resource.txt"
"$CUDA/cuobjdump" --dump-sass --gpu-architecture sm_60 --sort-functions \
    "$LIB" > "$OUT/all.sass"
"$CUDA/cuobjdump" --dump-elf-symbols --gpu-architecture sm_60 \
    "$LIB" > "$OUT/elf-symbols.txt"
"$CUDA/cuobjdump" --list-elf "$LIB" > "$OUT/elf-list.txt"

(
    cd "$OUT/cubins"
    "$CUDA/cuobjdump" --extract-elf \
        .3.sm_60.cubin,.18.sm_60.cubin,.28.sm_60.cubin "$LIB"
)
find "$OUT/cubins" -maxdepth 1 -type f -name '*.cubin' -print0 |
    sort -z | xargs -0 sha256sum > "$OUT/cubin-sha256.txt"
for cubin in "$OUT"/cubins/*.cubin; do
    "$CUDA/nvdisasm" --print-code --print-line-info \
        --print-instruction-encoding "$cubin" > "$cubin.nvdisasm"
done

compile_with_ptxas() {
    local source=$1
    local label=$2
    local directory
    local command
    local object="$OUT/ptxas-objects/$label.o"
    local log="$OUT/ptxas-objects/$label.log"

    directory=$(jq -r --arg source "$source" \
        '.[] | select(.file == $source) | .directory' "$COMPILE_DB")
    command=$(jq -r --arg source "$source" \
        '.[] | select(.file == $source) | .command' "$COMPILE_DB")
    if [[ -z "$directory" || -z "$command" ]]; then
        printf 'missing compile command for %s\n' "$source" >&2
        return 1
    fi
    command=${command% -o *}
    (
        cd "$directory"
        eval "$command -Xptxas=-v -o $(printf '%q' "$object")"
    ) > "$log" 2>&1
}

compile_with_ptxas \
    "$ROOT/ggml/src/ggml-cuda/affinity-wave.cu" affinity-wave
compile_with_ptxas \
    "$ROOT/ggml/src/ggml-cuda/cpy.cu" cpy
compile_with_ptxas \
    "$ROOT/ggml/src/ggml-cuda/gated_delta_net.cu" gated-delta-net
compile_with_ptxas \
    "$ROOT/ggml/src/ggml-cuda/ggml-cuda.cu" ggml-cuda

sed -n '2012,2112p' "$ROOT/ggml/src/ggml-cuda/ggml-cuda.cu" \
    > "$OUT/dense-selectors.source.txt"
rg -n \
    'aw_live_check_plan64|aw_r44_check_plan|P100_EXACT_PLAN_CHECK|vec_dot_q8_0_q8_1_ldg|get_int_b2_ldg' \
    "$ROOT/ggml/src/ggml-cuda" \
    "$RESULTS/run-qualification.sh" \
    > "$OUT/removed-symbol-check.txt" || true

printf 'archive=%s\n' "$OUT"
