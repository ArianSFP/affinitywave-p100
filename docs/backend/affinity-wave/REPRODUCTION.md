# Build and reproduction guide

## Safety and scope

The retained benchmark requires four GPUs at once. Run one job at a time and
install a disk watchdog before launching. On the original desktop rig, a
Vulkan crash loop could grow `.xsession-errors` at about 12 MB/s and fill the
disk.

The model is not included in this repository. Obtain it separately and verify:

```text
Qwen3.6-35B-A3B-Q8_0.gguf
size: 37,801,097,504 bytes
SHA-256: c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131
```

Never substitute sub-Q8 weights when comparing with the registered results.

## Reference hardware

| Item | Reference |
|---|---|
| GPUs | 4x Tesla P100-PCIE-16GB |
| Compute capability | 6.0 |
| Driver | 580.159.03 |
| CUDA compiler | 12.8.61 |
| Host compiler | GCC/G++ 14 |
| CMake | 4.2.3 |
| CPU affinity | logical CPUs 0-11 |
| Topology | all four GPUs behind one PCIe switch |

Explicit AffinityWave mode rejects hardware that is not exactly four sm_60
CUDA devices.

## Build

From the repository root:

```bash
cmake -S . -B build-p100-affinitywave \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-14 \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_ARCHITECTURES=60 \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON

cmake --build build-p100-affinitywave \
  --target llama-affinity-wave-bench llama-bench -j"$(nproc)"
```

The original build linked NCCL through the existing meta-backend configuration.
If the local checkout requires an explicit NCCL option, enable the option used
by that parent revision.

## Required calibration files

The benchmark expects:

- `placement-hot16.json`, SHA-256
  `96de3b381bb197b5d843bc9e536496114b74db2cfb77d6f7bf254c75dd851030`;
- `placement-primary.eplb`, SHA-256
  `e7ea71d1a6939a49152f8aae3b6792cc1d98a6be31e7abdd411c6e8466a5de85`.

Curated copies are under `docs/backend/affinity-wave/artifacts/`. The
hot-expert manifest is an aggregate-histogram proxy and must not be presented
as a production placement.

## Watchdog

Adapt this to the host:

```bash
watch_xsession() {
    while true; do
        xs=$(stat -c%s "$HOME/.xsession-errors" 2>/dev/null || echo 0)
        avail=$(df --output=avail / | tail -1 | tr -d ' ')
        if [ "$xs" -gt 52428800 ] || [ "$avail" -lt 5242880 ]; then
            : > "$HOME/.xsession-errors" 2>/dev/null
            pkill -9 -f 'build-p100-affinitywave/bin/llama-' || true
            break
        fi
        sleep 2
    done
}

watch_xsession &
WATCHDOG_PID=$!
trap 'kill "$WATCHDOG_PID" 2>/dev/null || true' EXIT INT TERM
```

Verify that no other four-GPU job is active before every run.

## Isolated exactness and service test

Set paths:

```bash
export AW_ROOT="$PWD"
export AW_BUILD="$AW_ROOT/build-p100-affinitywave"
export AW_MAP="$AW_ROOT/docs/backend/affinity-wave/artifacts/placement-hot16.json"
```

Run the retained exact service:

```bash
env -i \
  HOME="$HOME" \
  PATH="/usr/local/cuda-12.8/bin:/usr/bin:/bin" \
  LD_LIBRARY_PATH="$AW_BUILD/bin:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu" \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  NCCL_IB_DISABLE=1 \
  NCCL_CUMEM_ENABLE=0 \
  NCCL_ALGO=Ring \
  GGML_CUDA_P2P=1 \
  GGML_CUDA_AFFINITY_WAVE=1 \
  GGML_CUDA_AW_MAP="$AW_MAP" \
  GGML_CUDA_AW_WIRE=bf16 \
  GGML_CUDA_AW_CHECK=1 \
  GGML_CUDA_AW_Q8_LAYOUT=t64k32 \
  GGML_CUDA_AW_DOWN_CACHE=0 \
  GGML_CUDA_AW_M64_SPLIT=2 \
  GGML_CUDA_AW_HOME=layer \
  timeout --signal=TERM --kill-after=10s 300 \
  taskset --cpu-list 0-11 \
  "$AW_BUILD/bin/llama-affinity-wave-bench" \
  --tokens-per-cell 2048 --repeats 12 --route-pattern uniform
```

Expected exactness fields for every GPU:

```text
check_ran=1
check_passed=1
check_count=16777216
check_mismatches=0
check_expected=48020
check_observed=48020
```

Wall-derived rates vary slightly. The clean retained rerun ranged from 5.128
to 5.285 effective TFLOP/s/GPU.

## Output-withheld pp8128 service benchmark

Set:

```bash
export AW_MODEL=/absolute/path/to/Qwen3.6-35B-A3B-Q8_0.gguf
export AW_EPLB="$AW_ROOT/docs/backend/affinity-wave/artifacts/placement-primary.eplb"
```

The complete retained environment is:

```text
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
GGML_CUDA_AW_WIRE=bf16
GGML_CUDA_AW_CHECK=1
GGML_CUDA_AW_Q8_LAYOUT=t64k32
GGML_CUDA_AW_DOWN_CACHE=0
GGML_CUDA_AW_M64_SPLIT=2
GGML_CUDA_AW_HOME=layer
GGML_CUDA_AW_DECODE=t64
GGML_CUDA_AW_WAVE_INSPECT=0
GGML_CUDA_AW_WAVE_DRY=0
GGML_CUDA_AW_WAVE_DENSE=1
GGML_CUDA_AW_WAVE_TOKEN_SPLIT=1
GGML_CUDA_AW_WAVE_TOKEN_PLAN=0
GGML_CUDA_AW_WAVE_DENSE_BENCH=service
GGML_CUDA_AW_LANE_STAGGER=1
GGML_CUDA_AW_PRECAPTURE=1
GGML_CUDA_AW_SHARED_SERVICE=0
GGML_CUDA_AW_LANE_BALANCE=0
CUDA_LAUNCH_BLOCKING=0
GGML_CUDA_AW_DEBUG_NODE_SYNC=0
GGML_CUDA_AW_GROUP_CELLS=1
GGML_CUDA_AW_GDN_CHUNKED=2
GGML_CUDA_AW_GROUP_PATTERN=1111
GGML_CUDA_AW_GDN_WARPS=4
GGML_CUDA_AW_FUSED_GATE_UP=0
GGML_CUDA_AW_WAVE_TOKENS=8128
```

Use `scripts/affinitywave/benchmark-retained.sh`, which adds the model, map,
EPLB, library, timeout, CPU affinity, and watchdog paths.

Expected terminal line:

```text
AffinityWave: dense lane microbench mode=service tokens=8128 cells=160 corridors=668.4 MiB elapsed=... projected=... tok/s; output withheld
```

Expected process status is 1. The benchmark deliberately returns
`GGML_STATUS_ABORTED` after timing because it does not produce a valid model
output.

## Accuracy requirements for new work

- Arithmetic-preserving changes: exact owner-partial check.
- Full model changes: byte identity where possible.
- Otherwise: perplexity within +/-0.003 and KLD-pair behavior no worse than
  control self-variance.
- Run full-position perplexity before claiming state-corridor correctness.
- Never infer correctness from a greedy smoke test alone.

## Benchmark discipline

- Use fresh processes.
- Use interleaved controls for any adoption claim.
- Record full command, environment, commit, binary hashes, and raw output.
- Test p1024 before pp8128 for changes that affect tail shapes.
- Run only one four-GPU job at a time.
- Keep experimental builds separate from the retained build.
- Do not modify the checkpoint when exploring.
