# Reproducing the qualified result

## Hardware and software

The result is specific to:

- four Tesla P100 PCIe 16 GB GPUs;
- compute capability 6.0;
- CUDA 12.8.61;
- cuBLAS 12.8.3;
- g++ 14.3.0;
- Release build with CUDA, Flash Attention, and NCCL enabled;
- Qwen3.6-35B-A3B Q8_0 weights;
- the exact source checkpoint described in [RESULTS.md](RESULTS.md).

The host was an Intel Xeon E5-2650 v4 system. All four GPUs were PIX peers
behind one PCIe switch on one NUMA node. The harness pins CPU work to cores
0-11. GPU application clocks were not fixed for the final campaign.

The dense selector path validates its complete compile/runtime signature and
falls back when it does not match. A result on another CUDA/cuBLAS version is
a new qualification, not a reproduction.

Exact external inputs:

| Input | Bytes | SHA-256 |
| --- | ---: | --- |
| `Qwen3.6-35B-A3B-Q8_0.gguf` | 37,801,097,504 | `c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131` |
| WikiText-2 `wiki.test.raw` | 1,290,590 | `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08` |
| `placement-primary.eplb` | 36,670 | `e7ea71d1a6939a49152f8aae3b6792cc1d98a6be31e7abdd411c6e8466a5de85` |
| `placement-hot16.json` | 136,307 | `96de3b381bb197b5d843bc9e536496114b74db2cfb77d6f7bf254c75dd851030` |

The maps are included. The model and corpus are not, so exact reproduction
is not self-contained from this branch alone.

## Build outline

Use an isolated build directory and target `sm_60`. The exact local compiler
paths depend on the installed CUDA toolkit. A representative configuration
is:

```bash
cmake -S . -B ../build-p100-pairwave \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_NCCL=ON \
  -DCMAKE_CUDA_ARCHITECTURES=60

cmake --build ../build-p100-pairwave -j \
  --target llama-bench llama-perplexity \
           llama-affinity-wave-bench test-backend-ops
```

The archived [toolchain record](evidence/stitchrail/static/toolchain.txt) is
the source of truth for the qualified binary.

This branch contains the production diagonal source checkpoint. The later
PairFold scheduler and its `test-pairfold-scheduler` target are documented
for historical continuity but are not part of this branch's source.

Verify the immutable production source delta independently of this later
documentation commit:

```bash
git diff --binary 36bad6bb3^ 36bad6bb3 | sha256sum
```

Expected SHA-256:

```text
82d5d9826e31e7f238b9d33fca0483ac08094121d66df38f86e15cfb864c624f
```

## Production environment

The full environment is preserved in
[run-qualification.sh](harness/run-qualification.sh). The important
selectors are:

```text
GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1
GGML_CUDA_MOE_CSORT=1
GGML_CUDA_MOE_EP=1
GGML_CUDA_MOE_GROUPED=1
GGML_CUDA_MOE_PLAN=1
GGML_CUDA_AFFINITY_WAVE=1
GGML_CUDA_AW_WIRE=f32
GGML_CUDA_AW_PARTIAL=bf16
GGML_CUDA_AW_Q8_LAYOUT=t64k32
GGML_CUDA_AW_Q8_ENGINE=cohortrail
GGML_CUDA_AW_ROUTE_SCATTER=deterministic
GGML_CUDA_AW_DIAGONAL_SERVICE=panel2048
GGML_CUDA_AW_DENSE_SELECTORS=exact
GGML_CUDA_AW_P100_EXACT=1
GGML_CUDA_AW_P100_EXACT_SUM_WIDTH=4
```

The qualification also needs the code-calibrated EPLB map and AffinityWave
placement map referenced by the harness. The exact small text inputs are
included for reproduction:

- [placement-primary.eplb](evidence/runtime/placement-primary.eplb)
- [placement-hot16.json](evidence/runtime/placement-hot16.json)

The copied harness defaults `REFERENCE` to `evidence/runtime` and
`FRONTIER` to the included evidence directory. The original campaign paths
remain visible in the archived result records.

## Benchmark shape

The retained pp8128 benchmark uses:

```text
llama-bench
--model Qwen3.6-35B-A3B-Q8_0.gguf
-ngl 99
-sm tensor
-fa 1
-b 8128
-ub 8128
-mmp 0
-p 8128
-n 0
-r 1
-o json
```

Run under a clean environment and the same CPU affinity:

```text
env -i <production environment> taskset --cpu-list 0-11 ...
```

Under Nsight Systems, use plain `env`; `env -i` removes the profiler
injection variables.

## Four-GPU safety protocol

Every four-GPU job must:

1. acquire `/tmp/affinitywave-4gpu.lock`;
2. reject a second llama or Nsight process;
3. start the `.xsession-errors` and free-disk watchdog;
4. use the production environment and CPU taskset;
5. release the lock and verify no project GPU process remains.

The P100 desktop Vulkan loop can grow `.xsession-errors` at roughly
12 MB/s and fill the root filesystem. Do not run the campaign without the
watchdog. The archived harness implements the lock and watchdog.

At archive time `nvidia-smi` could not initialize because NVML 580.173 did
not match the loaded 580.159.03 kernel driver. Process and device-handle
inspection was therefore used as a fallback; only desktop handles remained.
That archive-time mismatch is not a complete record of the driver state
during every final campaign leg. Requalification should record the loaded
driver, clocks, power state, and `nvidia-smi topo -m` output explicitly.

## Qualification sequence

The final campaign order was:

1. two c512 service-dump arms, umbrella OFF and ON;
2. two c512 saved-logits arms;
3. five randomized pp2048 OFF/ON pairs;
4. three randomized pp8128 OFF/ON pairs;
5. static retained-kernel resource audit.

Acceptance required:

- byte identity for all 640 service-boundary comparisons;
- byte-identical saved logits with the expected SHA-256;
- every paired performance comparison positive;
- no retained target stack, local memory, or spills;
- production memory reclamation approximately 0.8 GiB/GPU or greater.

The campaign definition and results are preserved in
[evidence/stitchrail/campaign/](evidence/stitchrail/campaign/) and
[the campaign summary](archive/stitchrail/p100-exact-campaigns/stitchrail-final-prefill-v2/summary.md).

## Useful harness modes

The archived harness supports:

- `service`
- `bench`
- `diagonal-bench`
- `diagonal-bench-dump`
- `diagonal-ppl-save`
- `diagonal-trace`
- `bench-dump`
- `ppl-save`
- `trace`

The harness derives `ROOT`, `REFERENCE`, and `FRONTIER` from its checkout.
Override `BUILD`, `MODEL`, `CORPUS`, `RUN_HOME`, `XS`, `CUDA_ROOT`,
`SYSTEM_LIB`, `CPU_LIST`, or `RESULTS` for another machine. Inspect the
watchdog and process-matching rules before running it.

## PairFold checkpoint

The commands in this section require the later PairFold checkpoint and are
not runnable from this production-diagonal source branch alone. They are
retained to document the subsequent architecture and qualification method.

The later default-off PairFold runtime uses the same qualified arithmetic
stack plus:

```text
GGML_CUDA_AW_HEADFOLD=1
GGML_CUDA_AW_HEADFOLD_GDN=1
GGML_CUDA_AW_HEADFOLD_ATTENTION=1
GGML_CUDA_AW_HEADFOLD_SPLIT_PRE=1
GGML_CUDA_AW_PAIRWAVE_MANIFEST=<absolute manifest path>
GGML_CUDA_AW_PAIRWAVE_SERVICE=0
GGML_CUDA_AW_PAIRWAVE_BUNDLE=0
GGML_CUDA_AW_PAIRFOLD=1
```

`GGML_CUDA_AW_PAIRFOLD_SERIAL=1` selects the test-only serial arithmetic
control. The normal pipelined scheduler uses
`GGML_CUDA_AW_PAIRFOLD_SERIAL=0`.

The original-rig guarded harness is:

```text
bench/run-pairfold.sh
```

It contains the production environment, `/tmp/affinitywave-4gpu.lock`,
process exclusion, CPU taskset, timeout, and `.xsession-errors` watchdog.
Its `ROOT`, `BUILD`, `REFERENCE`, `MODEL`, and `CORPUS` constants are
absolute original-rig paths. Its `MANIFEST`, `XS`, `HOME`, CUDA paths,
library paths, visible devices, and CPU list are also rig-specific. Adapt
all of them before using the script elsewhere. The watchdog truncates the
configured `XS` file and SIGKILLs matching `/bin/llama-` processes on a
trip. The referenced EPLB and AffinityWave placement files are external
machine-specific runtime inputs.

Run the CPU scheduler test first:

```bash
ctest --test-dir <build-dir> \
  -R '^test-pairfold-scheduler$' \
  --output-on-failure
```

Run the c512 saved-logits oracle:

```bash
./bench/run-pairfold.sh pipeline-ppl 512 \
  checkpoint-exact-pipeline-c512-v1
```

Run an accurate warm pp8128 measurement:

```bash
PAIRFOLD_REPS=6 ./bench/run-pairfold.sh pipeline 8128 \
  warm-pipeline-pp8128-v1

PAIRFOLD_REPS=6 ./bench/run-pairfold.sh diagonal 8128 \
  warm-diagonal-pp8128-v1
```

Both commands deliberately retain the full-shape initialization call as
sample 0. Discard sample 0 and take the median of samples 1 through 5. Do
not use llama-bench's aggregate because it includes cold initialization.

The measured checkpoint and its limitations are in
[PAIRFOLD-CHECKPOINT.md](PAIRFOLD-CHECKPOINT.md).

## PairFold host streaming

The host-streaming extension adds:

```text
GGML_CUDA_AW_PAIRFOLD_HOST_WEIGHTS=1
GGML_CUDA_AW_PAIRFOLD_HOST_PLACEMENT=1
GGML_CUDA_AW_PAIRFOLD_HOST_LAYERS=FIRST:LAST
```

Qualify a small loader-host window before attempting all 40 layers:

```bash
env \
  PAIRFOLD_HOST_WEIGHTS=1 \
  PAIRFOLD_HOST_PLACEMENT=1 \
  PAIRFOLD_HOST_LAYERS=16:23 \
  bash bench/run-pairfold.sh \
    pipeline-ppl 512 loaderhost-c512
```

The expected saved-logits SHA-256 is:

`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`

Exact execution is not self-contained from this repository alone. The Q8_0
model, `placement-primary.eplb`, and `placement-hot16.json` are excluded.
Their exact hashes and the original-rig adaptation procedure are recorded in
[PAIRFOLD-HOST-STREAMING.md](PAIRFOLD-HOST-STREAMING.md).

The all-40 path requires 31.875 GiB of pinned expert storage. Measure host
headroom and stop other memory-heavy workloads first:

```bash
env \
  PAIRFOLD_HOST_WEIGHTS=1 \
  PAIRFOLD_HOST_PLACEMENT=1 \
  PAIRFOLD_HOST_LAYERS=0:39 \
  PAIRFOLD_REPS=6 \
  bash bench/run-pairfold.sh \
    pipeline 8128 loaderhost-all40-pp8128
```

Discard sample 0 and take the median of samples 1 through 5. Complete
architecture, safety, exactness, capacity, and reproduction details are in
[PAIRFOLD-HOST-STREAMING.md](PAIRFOLD-HOST-STREAMING.md).
