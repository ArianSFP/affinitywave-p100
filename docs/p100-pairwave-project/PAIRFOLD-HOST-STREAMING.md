# PairFold host-resident expert streaming

## Status

This checkpoint implements and measures exact Q8_0 expert streaming for the
PairFold runtime. It is a default-off proof of concept, not the production
path.

The exact PairFold parent is:

`a7c8786703e11a544c6d9d69603d70b9c0ac3152`

That parent does not contain host streaming. This document is part of the
follow-up commit containing the implementation. The source-only diff over
the parent has SHA-256:

`420fe139769a64c8ae7ac91844a52dcc39688e870de4ba17a977d03870b2fbef`

The qualified `libggml-cuda.so.0.15.3` has SHA-256:

`e7580770cbfa0cd7c7865295e74f8ebb884a6e51393d0e463905ad3cd3f548ed`

The corresponding `libggml.so.0.15.3` and `libllama.so.0.0.9876` hashes are
recorded in [ARTIFACTS.md](ARTIFACTS.md).

The implementation was qualified on:

- four Tesla P100 PCIe 16 GB GPUs;
- CUDA 12.8.61 and cuBLAS 12.8.3;
- Qwen3.6-35B-A3B Q8_0;
- the tracked PairWave lane-exact manifest;
- exact CohortRail, dense-selector, recurrent, attention, and ordered-sum
  arithmetic.

No sub-Q8 weights were used or evaluated.

## Result summary

| Path | Host expert layers | Warm pp8128 | VRAM recovered |
| --- | ---: | ---: | ---: |
| Resident PairFold | 0 | 3,027.038 ms / 2,685.133 tok/s | comparator |
| Shadow stream | 16:23 | 3,247.809 ms / 2,502.610 tok/s | none; adds about 828 MiB/GPU |
| Loader-host stream | 16:23 | 3,289.595 ms / 2,470.821 tok/s | 804 MiB/GPU |
| Loader-host stream | 0:39 | 4,238.061 ms / 1,917.858 tok/s | 7,280 MiB/GPU |
| Production diagonal | 0 | 2,790.956 ms / 2,912.264 tok/s | production comparator |

The successful qualified loader-host c512 runs report PPL 4.0783 and the
accepted saved-logits SHA-256:

`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`

The all-40 result covers PairFold inference layers `blk.0` through `blk.39`.
The model's MTP `blk.40` and all non-expert tensors remain device-resident.

## Why this path exists

PairFold divides four GPUs into two physical pairs. Each layer runs two
chronological prompt panels on the pair selected by the PairWave manifest.
After pipeline fill, one pair can compute while the other pair prepares a
future layer.

That pair alternation creates a bounded expert-storage opportunity. Instead
of retaining every expert projection in VRAM, the runtime can keep raw Q8_0
expert shards in pinned host memory and copy the next layer into two stable
device slots per GPU.

The implemented flow is:

```text
GGUF Q8_0 expert tensor
    |
    v
PairWave expert permutation and four-way split
    |
    v
model-owned CUDA-pinned host shards
    |
    v
CPU Q8_0 -> T64 byte permutation during model load
    |
    v
two generation-safe 408 MiB device slots per GPU
    |
    v
active-pair PairWave gate/up/down service
```

The arithmetic service is unchanged. Only expert-weight placement and
transport differ.

## Loader-time placement

The default-off selector is:

```text
GGML_CUDA_AW_PAIRFOLD_HOST_PLACEMENT=1
```

It requires host streaming and an inclusive layer range:

```text
GGML_CUDA_AW_PAIRFOLD_HOST_WEIGHTS=1
GGML_CUDA_AW_PAIRFOLD_HOST_LAYERS=FIRST:LAST
```

The loader validates model, layout, manifest, mmap, and selector
prerequisites before selecting tensor allocations. Remaining hardware and
exact-arithmetic checks run before the first graph or generation can use the
weights. Selected `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps`
weights use a dedicated four-way PairWave host meta buffer.

Each physical buffer is allocated with CUDA portable pinned memory. The
outer meta buffer deliberately does not claim ordinary host addressability,
so the loader enters the existing meta split and permutation path. This
preserves:

- the manifest-selected physical pair;
- the 128/128/0/0 expert-axis split;
- router and physical expert order;
- one copy of every selected expert;
- the existing T64 service layout.

After the loader gathers a physical projection, a tested CPU helper applies
the exact Q8_0-to-T64 byte permutation. The final pointer is registered in a
host-only catalog. It is never inserted into the device T64 catalog.

At first PairFold execution, the runtime checks that the selected catalog is
complete and that no selected device duplicate exists. It then borrows the
three model-owned host projections for each active layer/device.

There is no startup D2H snapshot in loader-placement mode.

## Streaming schedule and slot safety

Each GPU owns two device slots. A slot contains one complete physical
gate/up/down layer shard:

```text
slot bytes = 427,819,008
two slots  = 855,638,016 bytes/GPU nominal
```

The streamer issues three H2D copies from the loader-owned projections. It
records separate gate/up and full-layer readiness events so the service can
begin as soon as its required phase is ready.

Slot leases carry generation, layer, and panel identity. A slot is not
reused until chronological panel 1 releases the last reader. The scheduler
tests cover:

- both generation banks and wraparound;
- delayed completion;
- the repeated-pair layer-19/20 boundary;
- full-range slot reuse;
- malformed and overlapping CPU pack requests.

The qualified default uses parallel task-local H2D streams. Eager lookahead,
phased shadow copies, and globally serialized ingress were measured and
retained only as diagnostic controls because they were slower.

## Fail-closed contract

Loader placement cannot fall back to a path that expects resident experts.
It rejects incompatible execution before ordinary graph dispatch.

The qualified contract requires:

- exactly four P100 GPUs;
- Qwen3.6 MoE with 40 PairFold inference layers;
- Q8_0 expert tensors with exact gate/up/down shapes;
- `GGML_CUDA_AFFINITY_WAVE=1`;
- `GGML_CUDA_MOE_EP=1`;
- `GGML_CUDA_MOE_PLAN=1`;
- `GGML_CUDA_AW_PAIRFOLD=1`;
- `GGML_CUDA_AW_PAIRFOLD_HOST_WEIGHTS=1`;
- exact HeadFold, GDN, attention, and split-pre selectors;
- `GGML_CUDA_AW_P100_EXACT=1`;
- `GGML_CUDA_AW_P100_EXACT_SUM_WIDTH=4`;
- `GGML_CUDA_AW_WIRE=f32`;
- `GGML_CUDA_AW_PARTIAL=bf16`;
- `GGML_CUDA_AW_Q8_LAYOUT=t64k32`;
- `GGML_CUDA_AW_Q8_ENGINE=cohortrail`;
- `GGML_CUDA_AW_DENSE_SELECTORS=exact`;
- `GGML_CUDA_AW_ROUTE_SCATTER=deterministic`;
- `GGML_CUDA_AW_WAVE_OUTPUT=1`;
- `GGML_CUDA_AW_WAVE_DENSE_BENCH=service`;
- an absolute `GGML_CUDA_AW_PAIRWAVE_MANIFEST`;
- `--no-mmap`;
- qualified c512, pp2048, or pp8128 prefill geometry.

Decode, diagonal execution, ragged shapes, and unsupported graphs are
rejected while loader placement is active.

The proof supports one loaded placement model and one inference context per
process. Duplicate registration and a second model after preparation are
rejected. A second inference context is unsupported but is not detected and
rejected by a dedicated guard. Model destruction while preparation or
evaluation is in flight is outside the single-job lifetime contract.

When loader placement is disabled, ordinary CUDA pinned allocations retain
their original allocation flags and the resident PairFold path is
unchanged.

## Exactness evidence

The small-window and all-40 loader runs produce the accepted saved-logits
hash above. The all-40 generated logits file is 126,647,308 bytes and is
intentionally excluded from Git.

The CPU Q8_0-to-T64 test covers:

- exact gate/up shape, 512x2048;
- exact down shape, 2048x512;
- multiple matrices;
- inverse reconstruction;
- malformed dimensions and byte counts;
- overlapping input/output rejection.

The result records are:

- [shadow-stream proof](../../results/qwen36-35b-moe-pp-20260721/pairfold-20260726/HOST-STREAMING.md);
- [loader-placement and all-40 qualification](../../results/qwen36-35b-moe-pp-20260721/pairfold-20260726/LOADER-HOST-PLACEMENT.md);
- [cumulative PairFold result record](../../results/qwen36-35b-moe-pp-20260721/pairfold-20260726/RESULTS.md).

## Memory result

The initial loader window, layers 16 through 23, registers 48 physical
projections and 6,845,104,128 host bytes. With both device slots live, it
recovers 843,055,104 bytes, exactly 804 MiB, per GPU.

The all-40 path registers 240 physical projections and 34,225,520,640 host
bytes, exactly 31.875 GiB. It recovers the same amount at c512 and pp8128:

```text
7,633,633,280 bytes/GPU
7,280 MiB/GPU
7.109375 GiB/GPU
```

The proof fits the original 46 GiB host, but with little operating margin.
System-wide monitoring observed approximately 5.4 GiB minimum available
memory and effectively full 8 GiB swap. These samples are not a synchronized
process RSS high-water:

[loaderhost-all40-host-memory-monitor.txt](../../results/qwen36-35b-moe-pp-20260721/pairfold-20260726/loaderhost-all40-host-memory-monitor.txt)

Do not co-locate this all-40 configuration with another memory-heavy job on
a similarly sized host.

## Performance result

The all-40 pp8128 process retained five warm samples after discarding the
initialization call:

```text
4222.524, 4213.449, 4238.061, 4246.818, 4239.941 ms
```

The median is 4,238.061 ms / 1,917.858 tok/s. Peak-to-peak spread is 0.788%
of the warm mean.

Relative to the current resident PairFold measurement, all-40 streaming:

- adds 1,211.023 ms, or 40.007% wall time;
- reduces throughput by 28.575%.

Relative to production throughput, it is 34.145% slower.

Every generation moves 34,225,520,640 bytes. The median reported
serialized-pair rate is 5.632 GiB/s. Summed ready-event waits are
65.897-70.331 ms per warm generation. Most weight readiness is hidden, but
PCIe, copy-engine, and memory-system contention remain visible to compute.

The earlier eight-layer bytes-linear estimate was 4,075.253 ms. The
measured all-40 wall is 162.808 ms, or 3.995%, slower.

The all-40 benchmark process took approximately 300 seconds end to end.
Only 26.898 seconds belonged to the six timed inference samples. Model load,
CPU T64 packing, initialization, and teardown consumed the remaining
approximately 273 seconds. CPU T64 packing accounted for 66.285 seconds.
This is a one-time cost for a persistent process.

## Reproduction

This checkpoint is understandable without the original rig, but exact
execution is not self-contained. The Q8_0 model and two generated placement
inputs are deliberately excluded.

For an original-rig reproduction, first adapt the absolute `ROOT`, `BUILD`,
`REFERENCE`, `MODEL`, `CORPUS`, `MANIFEST`, and `XS` paths in:

`bench/run-pairfold.sh`

Also adapt `HOME`, CUDA `PATH`, `LD_LIBRARY_PATH`, `CUDA_VISIBLE_DEVICES`,
and the `taskset --cpu-list 0-11` affinity for the target machine.

The external EPLB and AffinityWave placement inputs used on the original
rig are not copied into this source tree. They must be copied or regenerated
separately; hashes verify the exact files but cannot reconstruct them:

```text
placement-primary.eplb
e7ea71d1a6939a49152f8aae3b6792cc1d98a6be31e7abdd411c6e8466a5de85

placement-hot16.json
96de3b381bb197b5d843bc9e536496114b74db2cfb77d6f7bf254c75dd851030
```

The tracked PairWave manifest is:

`docs/p100-pairwave-project/evidence/frontier/pairwave-laneexact-v3.manifest`

Its SHA-256 is:

`316c55cb76f0663190c333680c1e6574afc63ad2f15f8c26621459345d7735da`

Build:

```bash
cmake -S . -B ../build-p100-pairfold-host \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_NCCL=ON \
  -DCMAKE_CUDA_ARCHITECTURES=60

cmake --build ../build-p100-pairfold-host -j \
  --target llama-bench llama-perplexity test-pairfold-scheduler
```

Run local tests:

```bash
ctest --test-dir ../build-p100-pairfold-host \
  -R '^test-pairfold-scheduler$' \
  --output-on-failure

bash -n bench/run-pairfold.sh
```

Qualify a small window first:

```bash
env \
  PAIRFOLD_HOST_WEIGHTS=1 \
  PAIRFOLD_HOST_PLACEMENT=1 \
  PAIRFOLD_HOST_LAYERS=16:23 \
  bash bench/run-pairfold.sh \
    pipeline-ppl 512 external-loaderhost-c512
```

Verify the generated logits:

```bash
sha256sum \
  results/qwen36-35b-moe-pp-20260721/pairfold-20260726/\
external-loaderhost-c512-logits.bin
```

Only after the small window passes and host headroom is measured, run all 40
inference layers:

```bash
env \
  PAIRFOLD_HOST_WEIGHTS=1 \
  PAIRFOLD_HOST_PLACEMENT=1 \
  PAIRFOLD_HOST_LAYERS=0:39 \
  bash bench/run-pairfold.sh \
    pipeline-ppl 512 external-loaderhost-all40-c512

env \
  PAIRFOLD_HOST_WEIGHTS=1 \
  PAIRFOLD_HOST_PLACEMENT=1 \
  PAIRFOLD_HOST_LAYERS=0:39 \
  PAIRFOLD_REPS=6 \
  bash bench/run-pairfold.sh \
    pipeline 8128 external-loaderhost-all40-pp8128
```

The runner acquires the four-GPU lock, excludes competing llama and profiler
processes, applies the CPU taskset, and starts the `.xsession-errors`
watchdog. On a watchdog trip it truncates the configured `XS` file and
SIGKILLs matching `/bin/llama-` processes. Adapt that behavior before using
the harness on another host. Do not bypass the guards on the original P100
desktop rig.

For pp8128, discard sample 0 and take the median of samples 1 through 5. The
JSON aggregate includes initialization and is not the warm result.

## Source map

| File | Role |
| --- | --- |
| `src/llama-model-loader.cpp` | selector validation and tensor placement |
| `src/llama-model-loader.h` | loader placement state and selected-tensor count |
| `src/llama-model.cpp` | model-level qualification and tensor-count gate |
| `ggml/include/ggml-backend.h` | experimental PairWave host meta-buffer API |
| `ggml/src/ggml-backend-meta.cpp` | four-way pinned host meta buffer and PairWave permutation |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | placement-gated portable allocation, cleanup, and exported callbacks |
| `ggml/src/ggml-cuda/affinity-wave.cu` | host catalog, CPU pack integration, slots, copies, events, diagnostics |
| `ggml/src/ggml-cuda/affinity-wave.cuh` | host-weight callbacks and live-cell slot identity |
| `ggml/src/ggml-pairfold-scheduler.h` | static weight-slot mapping and CPU Q8_0-to-T64 helper |
| `tests/test-pairfold-scheduler.cpp` | scheduler, lease, packing, inverse, and malformed-contract tests |
| `bench/run-pairfold.sh` | guarded original-rig qualification harness |
| `bench/analyze-pairfold-trace.py` | H2D copy and compute-overlap accounting |

## Decision

The architecture works:

- selected expert GPU allocations are absent;
- c512 output remains byte-identical;
- all 40 inference layers fit in pinned host memory;
- two bounded slots recover 7.109375 GiB/GPU;
- the retained two-pair scheduler can overlap compute and ingress.

It is not production-qualified on this P100 PCIe rig. The next useful work
is reducing bytes or copy/memory contention, improving resident PairFold
compute, using a faster host link, and restoring host-capacity margin.
