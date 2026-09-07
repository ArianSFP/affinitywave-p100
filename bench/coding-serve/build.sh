#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
build_dir="$project_dir/build-coding-serve"
if [[ -d "$build_dir/tools/ui/dist" ]]; then
    # The serving profile uses --no-webui. Remove stale generated assets so
    # an incomplete network-fetched bundle cannot break the server build.
    cmake -E remove_directory "$build_dir/tools/ui/dist"
fi
cmake -S "$project_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
    -DCMAKE_CUDA_ARCHITECTURES=60 -DGGML_CUDA=ON \
    -DGGML_CUDA_NCCL=ON -DGGML_CUDA_FA_ALL_QUANTS=OFF \
    -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_TESTS=ON -DGGML_CCACHE=OFF
cmake --build "$build_dir" --target llama-server llama-bench llama-perplexity test-backend-ops --parallel 8
/usr/bin/g++-14 -std=c++17 -O2 \
    -I"$project_dir/include" -I"$project_dir/common" \
    -I"$project_dir/ggml/include" -I"$project_dir/vendor" \
    "$project_dir/bench/coding-serve/validate.cpp" \
    -L"$build_dir/bin" -Wl,-rpath,"$build_dir/bin" \
    -lllama-common -lllama -lggml -lggml-base \
    -o "$build_dir/bin/test-aw-serving"
