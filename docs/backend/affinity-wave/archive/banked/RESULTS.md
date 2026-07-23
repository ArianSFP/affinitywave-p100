# AffinityWave Phase 0 and prototype results - 2026-07-22

## User-directed prototype override

The Phase-0 NO-GO below remains the feasibility verdict. The user explicitly
overrode its stop condition and requested a prototype as a springboard for
further development. Phase 1 was therefore implemented and measured in a clean,
isolated source tree after the read-only analysis; the shared production source
and protected builds were not modified.

## Verdict

**NO-GO. Do not build the backend.**

The registered gate requires a pessimistic pp8192 estimate at or below 2.05 s.
The discrete-event result is **2.1810 s (3756 tok/s)** at the proposed Phase-1
service floor of 5.5 effective TFLOP/s/GPU and 10 GB/s peer transport.

More decisively, the optimistic fixed-work plus ideal expert-compute lower bound
is already **2.1122 s**. That bound excludes all NCCL time, all measured
`k_get_rows_float` time, all existing MoE-plan kernels, and the full measured
p-fold wall saving. It also assumes perfect expert balance and charges no
packing or transport. It misses the gate by 62.2 ms before those costs exist.

No AffinityWave backend source, build directory, or GPU run was started after
this failed gate.

## Inputs

- Model: `Qwen3.6-35B-A3B-Q8_0.gguf`, 37,801,097,504 bytes.
- Model SHA-256:
  `c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131`.
- Baseline: 1697 tok/s, or 4.8273 s for pp8192.
- Routing: existing code-domain `round3-phase0/hist-code.txt`.
- Kernel costs: existing A6 `grouped-20260722/plan-pp8192.sqlite` trace.
- Trace contains two pp8192 executions (warm-up plus measured pass).
- Measured topology constants: 10 GB/s pessimistic sustained peer transport;
  the prior all-to-all probe measured approximately 9.7 GB/s/GPU.
- Source references: analysis checkout `beac5309f`; actual measured 35B source
  checkout `05dcabf9a`.

The routing histogram has exact per-expert counts but not per-token route
co-occurrence. The report therefore labels it `aggregate_histogram_proxy` and
uses a lower bound for remote token-owner pairs. A production placement manifest
still requires a token-level trace.

## Conservative compute bound

The saved nsys trace gives the following per-GPU times per pp8192 pass. The
maximum rank is used wherever a single scalar is needed.

| Component | Per-GPU range (s) | Treatment |
|---|---:|---|
| All kernels | 3.9377-3.9645 | starting point |
| NCCL | 1.0644-1.1331 | removed entirely |
| existing expert kernels | 0.8115-0.8868 | removed and repriced at 5.5 TFLOP/s |
| all `k_get_rows_float` | 0.5735-0.5757 | removed entirely |
| MoE plan kernels | 0.0159-0.0161 | removed entirely |
| p-fold saving | 0.1475 | subtracted in full |
| remaining fixed work | 1.2638-1.2675 | retained |

The 43-diagonal fill turns the 1.2675 s maximum fixed-work value into a
**1.3626 s fixed wavefront floor**.

Routed expert arithmetic is fixed by the model:

- 8192 tokens x 40 MoE layers x top-8 routes.
- Three 2048x512 projections per route.
- 16.4927 TFLOP total.
- At 4 x 5.5 TFLOP/s, the ideal balanced expert time is **0.7497 s/GPU**.

Therefore `1.3626 + 0.7497 = 2.1122 s`, already above the 2.05 s gate.
This is intentionally more favorable than an implementable backend.

## Discrete-event result

The simulator executes the 40x4 dependency grid as 43 diagonals. Each diagonal
has one home-cell fixed workload per active chunk plus owner-routed expert work.
It allows P2P state and request/response copies to overlap compute completely
and never overlaps two throughput-bound compute kernels on one P100.

| Expert service | Simulated pp8192 | Throughput | Gate |
|---:|---:|---:|---|
| 5.5 TFLOP/s/GPU | 2.1810 s | 3756 tok/s | fail |
| 6.0 TFLOP/s/GPU | 2.1128 s | 3877 tok/s | fail |
| 6.5 TFLOP/s/GPU | 2.0551 s | 3986 tok/s | fail |
| 7.0 TFLOP/s/GPU | 2.0056 s | 4084 tok/s | pass only as an unmeasured sensitivity |
| 9.3 TFLOP/s/GPU | 1.8466 s | 4436 tok/s | physical-peak sensitivity, not attainable end-to-end |

Even the ideal balance/no-transport arithmetic requires 5.998 TFLOP/s/GPU to
reach 2.05 s. The resource-aware model needs slightly more than 6.5 TFLOP/s;
7.0 TFLOP/s is the first registered sensitivity point with useful margin. The
plan's 5.5 TFLOP/s Phase-1 threshold is insufficient for the breakthrough gate.

## Placement result

`placement-hot16.json` is deterministic and bound to the exact GGUF hash. For
every one of 40 layers it contains:

- 256 primary owners, exactly 64 per GPU;
- 16 replicated expert IDs;
- exactly four hot-expert primary copies per GPU, making the additional hot-16
  expert storage exactly 1.4941 GiB/GPU for the 40 MoE layers;
- calibration and held-out skew statistics.

Mean held-out cold-owner skew is 1.0859; the mean per-layer held-out p95 is
1.1682. Because the input is aggregate rather than token-level, this manifest is
an analysis artifact, not a valid production `GGML_CUDA_AW_MAP`.

## Implemented artifacts and checks

- `affinitywave_phase0.py`: deterministic placement optimizer, manifest
  validator, nsys cost extractor, and 43-diagonal simulator.
- `affinitywave-trace-05dcabf9.patch`: compact token-level top-8 trace
  instrument for the actual 35B source. `git apply --check` passes against
  `05dcabf9`; it was not applied to the shared worktree after the compute gate
  failed.
- `test_affinitywave_phase0.py`: four unit tests covering binary trace parsing,
  deterministic/capacity-constrained placement, trace cost exclusions, and the
  43-diagonal schedule.
- `phase0-report.json`: complete machine-readable costs, bounds, stages,
  sensitivity table, and gate decision.
- `placement-hot16.json`: deterministic aggregate-proxy placement.

Verification:

```text
python3 -m unittest -v test_affinitywave_phase0.py
Ran 4 tests in 4.088s - OK

python3 -m py_compile affinitywave_phase0.py test_affinitywave_phase0.py
PASS

git -C /home/arian/llama.cpp-q36-moe apply --check affinitywave-trace-05dcabf9.patch
PASS
```

Analysis command (exit 2 is the intentional valid NO-GO status):

```bash
python3 affinitywave_phase0.py \
  --routes ../round3-phase0/hist-code.txt \
  --nsys ../grouped-20260722/plan-pp8192.sqlite \
  --gguf /home/arian/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf \
  --gguf-sha256 c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131 \
  --output .
```

## Reopening condition

The present plan stops here. Reopening it needs new evidence, not another
backend implementation attempt: a token-level trace plus an isolated
cross-layer expert-service result of roughly 7.0 effective TFLOP/s/GPU including
packing, while retaining at least 10 GB/s peer transport. Any revised model must
then remain at or below 2.05 s with exact token-owner traffic and measured state
copy costs.

## Phase 1: isolated expert-service prototype

### Implementation

- Source: `/home/arian/llama.cpp-qwen36/affinitywave-src`, clean clone of
  `05dcabf9a` before the uncommitted prototype changes.
- Build: `/home/arian/llama.cpp-qwen36/build-p100-affinitywave`, Release,
  CUDA 12.8.61, `CMAKE_CUDA_ARCHITECTURES=60`, Flash Attention and NCCL on.
- New backend files: `ggml/src/ggml-cuda/affinity-wave.{cu,cuh}`.
- New executable: `bin/llama-affinity-wave-bench`; a full `llama-bench` target
  also links successfully in the isolated build.
- No public llama API change. The service is exposed through the CUDA backend
  proc-address table.
- `GGML_CUDA_AFFINITY_WAVE`, `GGML_CUDA_AW_MAP`, `GGML_CUDA_AW_WIRE`, and
  `GGML_CUDA_AW_CHECK` are implemented. Explicit mode strictly validates the
  v1 manifest and the four-sm_60-device hardware contract.
- The service pools 256 `(layer, expert)` descriptors from four active cells in
  one launch per projection. Timed work includes F32/BF16 request packing,
  native-Q8 gate/up/down, SwiGLU, fixed-rank owner summation, and one BF16
  response rounding.
- Normal llama graphs deliberately remain on EP4 in Phase 1. Dense/hot loading,
  the diagonal scheduler, state corridors, append-prefill, and decode-state
  normalization are not claimed as implemented.

### Registered F32 result

Shape: four GPUs concurrently, four active cells x 2048 tokens, 256 descriptors
and 16,384 routed rows per GPU, 12 timed repetitions after two warm-ups. Every
run was bounded by `timeout` and the `.xsession-errors` watchdog.

| Variant | Min GPU TFLOP/s | Accuracy | Disposition |
|---|---:|---|---|
| initial generic loads | 4.625 | exact BF16 check, 4/4 | retained log only |
| two LDG.128 F32 loads | 4.763 | exact BF16 check, 4/4 | improvement |
| stage-timed persistent service | **4.814** | exact BF16 check, 4/4 | retained implementation |
| 32x128 shared-A tile | 4.042 | exact BF16 check, 4/4 | reverted; occupancy loss |
| direct 2-D launch grid | 4.720 | exact BF16 check, 4/4 | reverted |

The retained stage-timed run measured the following tightly grouped per-GPU
stage ranges:

| Stage | Time range (ms) |
|---|---:|
| request pack | 0.259-0.265 |
| gate | 6.201-6.222 |
| up | 6.195-6.276 |
| SwiGLU | 0.185-0.186 |
| down | 7.030-7.130 |
| fixed-order BF16 owner reduction | 0.521-0.524 |

The slowest device completed the full service at 4.814 TFLOP/s, so the 5.5
TFLOP/s Phase-1 compute gate **fails**. Accuracy passed: the observed and
reference owner partial were both BF16 bits `48020` on every GPU. The service
used 1,258,530,816 bytes per GPU and retained far more than the required 768 MiB
headroom in this isolated run.

The independent same-rig transport gate remains green: the archived P7
copy-engine test measured **11.96 GB/s per GPU under full GEMM load**, with GEMM
time 5.21 versus 5.19 ms/round alone. See
`../p7-pcie-gate/pcie-concurrency-v2.txt`.

### Phase-1 conclusion

This satisfies the user's request for a buildable springboard but not the plan's
performance continuation gate. The measurements explain the gap: packing and
owner reduction total less than 0.8 ms; the native-Q8 projections dominate.
Widening the tile loses occupancy on P100, and removing the persistent grid does
not help. A full wave scheduler should not be represented as performance-ready
from these results. The next credible compute experiment is a service-boundary
fusion that preserves two resident CTAs; the closed HFMA2, same-SM overlap, and
lossy/sub-Q8 paths remain out of scope.

Complete logs are `phase1-service-f32.log` and
`phase1-service-f32-v2.log` through `phase1-service-f32-v5-direct-grid.log`.

## Retained Phase-2a checkpoint - 2026-07-23

### Banked result

The exact retained configuration is now checkpointed locally. The two clean
pp8128 confirmation runs were:

| Run | Elapsed | Throughput |
|---|---:|---:|
| confirmation 1 | 2874.222 ms | 2827.9 tok/s |
| confirmation 2 | 2874.839 ms | 2827.3 tok/s |
| arithmetic mean | 2874.531 ms | 2827.6 tok/s |

The elapsed-time spread is 0.617 ms, or 0.0215% of the mean. This confirms the
previous 2873.529 ms / 2828.6 tok/s result within normal run-to-run variation.
The rate banked for this checkpoint is therefore approximately 2828 tok/s.

Both full-model runs intentionally end with status 1 after printing the timed
line because the Phase-2a service benchmark withholds the model output and
returns `GGML_STATUS_ABORTED`. No CUDA error, watchdog trip, or OOM occurred.

### Retained configuration

| Setting | Retained value |
|---|---|
| expert weight layout | exact Q8_0 T64/K32 |
| expert projection kernel | M64, split-K2 |
| owner wire and partial | BF16, fixed owner order |
| dense token layout | four token lanes, replicated dense weights |
| recurrent kernel | GDN chunk columns 2, 4 warps |
| wave schedule | lane stagger 1 |
| service grouping | singleton groups, pattern `1111` |
| pre-capture | 2 untimed wave passes |
| disabled variants | down cache, shared service, lane balancing, fused gate/up |

The abandoned ring-mirror prototype and its environment control were removed
before rebuilding. The temporary direct CUDA failure print, graph-disable
switch, and live VRAM probes were also removed. Other exploratory variants are
default-off and the checkpoint command pins each relevant control explicitly.

### Complete pp8128 command and environment

The two confirmations used the identical watchdog wrapper command:

```bash
bash /home/arian/llama.cpp-qwen36/results/qwen36-35b-moe-pp-20260721/affinitywave-20260722/benchmark-phase2a-smoke.sh \
  t64 t64 0 0 1 1 0 8128 1 service 1 1 0 0 0 0 bf16 1 2 1111 4 0
```

The wrapper uses `env -i`, the `.xsession-errors` watchdog, a 900-second
timeout, and `taskset --cpu-list 0-11`. Its complete effective environment was:

```text
HOME=/home/arian
PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin
LD_LIBRARY_PATH=/home/arian/llama.cpp-qwen36/build-p100-affinitywave/bin:/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu
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
GGML_CUDA_MOE_EPLB_MAP=/home/arian/llama.cpp-qwen36/results/qwen36-35b-moe-pp-20260721/affinitywave-20260722/placement-primary.eplb
GGML_META_SUBMIT_THREADS=1
GGML_CUDA_AFFINITY_WAVE=1
GGML_CUDA_AW_MAP=/home/arian/llama.cpp-qwen36/results/qwen36-35b-moe-pp-20260721/affinitywave-20260722/placement-hot16.json
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

The executable invocation inside the wrapper was:

```text
timeout --signal=TERM --kill-after=10s 900 taskset --cpu-list 0-11 \
  /home/arian/llama.cpp-qwen36/build-p100-affinitywave/bin/llama-bench \
  --model /home/arian/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-Q8_0.gguf \
  -ngl 99 -sm tensor -fa 1 -b 8128 -ub 8128 -mmp 0 -p 8128 -n 1 -r 1 -o json
```

### Accuracy evidence

The cleaned build was rechecked with:

```bash
bash /home/arian/llama.cpp-qwen36/results/qwen36-35b-moe-pp-20260721/affinitywave-20260722/benchmark-phase1b.sh t64-bf16
```

All four GPUs passed the exact BF16 owner-partial check. Each GPU checked
16,777,216 values with zero mismatches; `check_expected` and `check_observed`
were both 48020. The clean rerun ranged from 5.128 to 5.285 effective
TFLOP/s/GPU by wall time. This retains the exact T64/M64 split-K2 path; the
earlier retained winner was 5.237 TFLOP/s/GPU, also exact. The checkpoint does
not use sub-Q8 weights.

### Source and build identity

- Isolated source: `/home/arian/llama.cpp-qwen36/affinitywave-src`.
- Parent commit: `05dcabf9a97b8a91bec5621d2b4c5ce1ce9b2ca8`.
- Local checkpoint commit: `9d3983b89952c7fc1c6aa38fc1a7bd3182992382`.
- Checkpoint tree: `acbecf68f512212b95015540e3774864bf935aed`.
- Complete pre-commit `git diff --cached --binary` SHA-256:
  `62ec789b8e55fcc9a3128dc7f27c8d6f3157b9b7b65b428f26cbf1e269f656aa`.
- Diff size: 14 files, 4,979 insertions, 71 deletions. The complete diff is
  recoverable with `git show --binary 9d3983b89952c7fc1c6aa38fc1a7bd3182992382`.
- Build: Release, `CMAKE_CUDA_ARCHITECTURES=60`, CUDA FA on, NCCL on,
  CUDA 12.8.61, CMake 4.2.3.
- `llama-bench` SHA-256:
  `eb949e2cf086b0e6dab6fd018418f2f2ec356a1585c035ff06e93c439e8a26c8`.
- `llama-affinity-wave-bench` SHA-256:
  `0f49ab934b6a80e5109b43dbe4276ccdf4cf4a8f96893a1aa344ee2109a093a8`.
- `libggml-cuda.so.0.15.3` SHA-256:
  `d38b3a4f9fea749a0edfdab6ded9d88135bc3e8ec87291a2b5b1802aaf82e223`.
- Model SHA-256:
  `c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131`.
- Benchmark wrapper SHA-256:
  `26f6bfb95eab8fec2628b6b1af0ec59f286325933c728e242047f94f092ecc76`.
- Placement manifest SHA-256:
  `96de3b381bb197b5d843bc9e536496114b74db2cfb77d6f7bf254c75dd851030`.
- EPLB map SHA-256:
  `e7ea71d1a6939a49152f8aae3b6792cc1d98a6be31e7abdd411c6e8466a5de85`.

Raw checkpoint logs and SHA-256 values:

```text
574b1465f068c16c3ffab87b2daba1e7ecd2136dca6c7f72c7d639fdcf3936ed  checkpoint-pp8128-confirm1.err
65e49b1951e680527a9977b8f155c29119ca6dbaaefa6103423ff970fbd0c7bc  checkpoint-pp8128-confirm2.err
64efe823b251307e9704dff95df09ad84e8f86229ae7344edf462724028d2550  checkpoint-exact-t64-bf16.log
```

No production source or protected build was modified. The checkpoint is local;
no push or PR was made.

The immutable source checkout remains at
`/home/arian/llama.cpp-qwen36/affinitywave-src` on the checkpoint commit. A clean
experimental worktree was created at
`/home/arian/llama.cpp-qwen36/affinitywave-experimental` on local branch
`affinitywave-experimental-20260723`, initially at the same commit.

### Remaining limitations

- This is still an output-withheld service benchmark, not ordinary end-to-end
  model execution. It does not yet return usable logits from the 40x4 schedule.
- The exact check covers the isolated synthetic expert service and fixed-order
  BF16 owner partials. An end-to-end logits or KLD-pair validation of the full
  wavefront remains required before production use.
- Append-prefill, decode normalization, and a production request lifecycle are
  not implemented.
- Dense replication, state corridors, and the wavefront are exercised through
  the benchmark-only meta-backend path. CUDA graph support on P100 remains an
  experimental override.
- The placement manifest is based on an aggregate routing proxy rather than a
  token-level production trace; hot-expert replication is not used by the
  retained service schedule.
- The retained rate is approximately 2828 tok/s, not the theoretical 3500
  tok/s target. Further experiments must start from a separate state and must
  not modify commit `9d3983b89952c7fc1c6aa38fc1a7bd3182992382`.
