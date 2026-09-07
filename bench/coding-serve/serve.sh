#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
run_tag="serve-$(date -u +%Y%m%dT%H%M%SZ)-$$"
exec python3 "$project_dir/bench/coding-serve/run.py" \
    --build "$project_dir/build-coding-serve" \
    --tag "$run_tag" --variant serve --mode serve --timeout 0 \
    --tokens 8128 --ctx 16384 --port 18097 \
    --prewarm 64,128,256,512,1024 \
    --speculative dspark --draft-n-max 2 \
    --verbose \
    "$@"
