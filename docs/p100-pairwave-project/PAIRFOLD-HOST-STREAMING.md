# PairFold host-resident expert streaming

## Status

This checkpoint implements and measures exact Q8_0 expert streaming for the
PairFold runtime. It is a default-off proof of concept, not the production
path.

The original host-streaming source checkpoint is:

`f177771d8f0411e8f547bd2aff742d0e4889cca1`

Its exact PairFold parent is:

`a7c8786703e11a544c6d9d69603d70b9c0ac3152`

That parent does not contain host streaming. The original `f177771d8`
source-only diff over the parent has SHA-256:

`420fe139769a64c8ae7ac91844a52dcc39688e870de4ba17a977d03870b2fbef`

The corresponding original qualified `libggml-cuda.so.0.15.3` has SHA-256:

`e7580770cbfa0cd7c7865295e74f8ebb884a6e51393d0e463905ad3cd3f548ed`

The corresponding `libggml.so.0.15.3` and `libllama.so.0.0.9876` hashes are
recorded in [ARTIFACTS.md](ARTIFACTS.md).

The later projection-residency, owner-return, and physical-H2D findings in
this document postdate that checkpoint. Their source is not included in
this production-diagonal branch, so the hashes above must not be interpreted
as identities for those later changes.

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
| Loader-host down-only, bypass off | 0:39 | 3,353.908 ms / 2,423.44 tok/s | 1.840 GiB/GPU |
| Loader-host up+down | 0:39 | 3,553.836 ms / 2,287.11 tok/s | 4.475 GiB/GPU |
| Loader-host high capacity | all except 0/19/20 | 3,988.963 ms / 2,037.62 tok/s | 6.717 GiB GPU0/1; 6.318 GPU2/3 |
| Loader-host all projections | 0:39 | 4,131.291 ms / 1,967.42 tok/s | 7.109375 GiB/GPU |
| Production diagonal | 0 | 2,790.956 ms / 2,912.264 tok/s | production comparator |

The successful qualified loader-host c512 runs report PPL 4.0783 and the
accepted saved-logits SHA-256:

`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`

The all-40 result covers PairFold inference layers `blk.0` through `blk.39`.
The model's MTP `blk.40` and all non-expert tensors remain device-resident.

The legacy-equivalent layers-16:23 manifest and all three mixed policies
also pass c512 at PPL 4.0783 with the accepted SHA-256 above. The current
all-host row improves on the original 4,238.061 ms / 1,917.858 tok/s
measurement; both results remain recorded below.

The longer pp8128 investigation isolated a pipeline-only defect in the
home-local owner bypass. With that bypass enabled, repeated down-only
pipeline runs produced four different logits hashes:

- `aabb2ba6feb49f8832a3eb887d7665ab7a444190a98e21d4ce4074a42449e214`;
- `9fdbf2f2f4a94356409cef289e85bc842a6653a332e0dca101a689fa347934de`;
- `1fbec941e0076e63211fa7554db4d031a372b09db5c43b080ca6ad185a5559fe`;
- `e734393e0dae263ba97f70e1851699437c66ed079e4389f5a8c818fa2304bb8b`
  after adding the proposed direct-local wait.

The pipeline with bypass disabled and the serial scheduler with bypass
enabled both report PPL 6.7466 and the resident literal SHA-256:

`8cf55039bf0107fee294dba9d3c4413d61c5e1cac4f170c68a3f66303520f86e`

Host packing, projection selection, H2D publication, and local PairWave
arithmetic are therefore cleared. The runtime and harness now allow the
bypass only with the serial scheduler. The prior bypass-on mixed-policy
timings remain useful diagnostic performance/capacity estimates, but they
are not strict pp8128 exactness qualifications.

The final bypass-disabled down-only timing run retained:

```text
3372.835585, 3353.908010, 3329.440313, 3368.096146, 3330.874014 ms
```

Its median is 3,353.908010 ms / 2,423.44 tok/s while reclaiming 1.840
GiB/GPU. It is 33.526 ms, or 1.01%, slower than the unsafe bypass-on
3,320.382 ms result. It is also slower than resident PairFold at 3,027.038
ms / 2,685.133 tok/s and 16.79% lower in throughput than production at
2,790.956 ms / 2,912.264 tok/s.

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

It requires host streaming and either an inclusive whole-layer range:

```text
GGML_CUDA_AW_PAIRFOLD_HOST_WEIGHTS=1
GGML_CUDA_AW_PAIRFOLD_HOST_LAYERS=FIRST:LAST
```

or an absolute per-projection residency manifest:

```text
GGML_CUDA_AW_PAIRFOLD_HOST_WEIGHTS=1
GGML_CUDA_AW_PAIRFOLD_HOST_RESIDENCY_MANIFEST=/absolute/path
```

The two placement selectors are mutually exclusive. See
[PAIRFOLD-PROJECTION-RESIDENCY.md](PAIRFOLD-PROJECTION-RESIDENCY.md) for the
strict grammar and tracked policies.

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

The readiness policy is explicit:

- Without `GGML_CUDA_AW_PAIRFOLD_HOST_PHASED=1`, a legacy whole-layer range
  can use one contiguous 408 MiB slab copy. A residency manifest copies its
  hosted gate/up projections before a combined gate/up readiness boundary,
  then copies down. Gate therefore does not have independent early
  readiness.
- With `GGML_CUDA_AW_PAIRFOLD_HOST_PHASED=1`, the H2D stream records separate
  gate, up, and down events immediately after each selected projection.
  PairWave waits for gate before gate compute, for up between gate and up
  compute, and for down before down compute.

Per-projection readiness is not the default and must not be inferred merely
from the existence of three slot events.

Slot leases carry generation, layer, and panel identity. The current
resource is one complete `(generation, layer, slot)` lease, not three
projection leases. A slot is not reused until chronological panel 1 releases
the last output consumer. The scheduler tests cover:

- both generation banks and wraparound;
- delayed completion;
- the repeated-pair layer-19/20 boundary;
- full-range slot reuse;
- malformed and overlapping CPU pack requests.

The original all-40 qualification used parallel task-local H2D streams and
nonphased whole-layer transfers. A later small-window speed screen found the
phased path exact and modestly faster. The follow-up campaign then qualified
all four residency manifests at c512 and measured the mixed and all-host
policies at pp2048/pp8128.

The complete 408 MiB lease remains conservative even when gate or up has no
remaining reader. Releasing those regions early would require
projection-specific generation/layer identity, projection free events
recorded after the final panel-1 reader, and a prefetch schedule that cannot
consume a stale event from an older lease. Independent projection leases
are not implemented.

## Transfer scheduling and contention

The host-weight stream is created at CUDA's least-priority setting. That is
not a copy-engine quality-of-service guarantee: it does not preempt an H2D
already in flight and does not prevent contention with destination HBM,
owner returns, hidden handoffs, or attention-state traffic.

The runtime therefore exposes explicit diagnostic controls:

```text
GGML_CUDA_AW_PAIRFOLD_HOST_CHUNK_MIB=4|8|16|32|64
GGML_CUDA_AW_PAIRFOLD_HOST_PAIR_H2D=1
GGML_CUDA_AW_PAIRFOLD_HOST_SERIAL_H2D=1
```

These are diagnostic scheduling controls, not a completed critical-path
admission controller. Chunk fragments are still queued back-to-back on one
H2D stream; pair and global serialization operate at whole-layer
granularity. A future controller would need a scheduler-owned critical-copy
signal between chunks and a pair-tail event published before enqueue.

On the exact layers-16:23 pp2048 speed arm with phased readiness, local-owner
bypass, and parallel submission, unchunked parallel ingress had a 1,321.334
ms warm median. The 16 MiB and 64 MiB chunk controls measured 1,338.936 ms
and 1,324.366 ms; pair-exclusive admission measured 1,501.410 ms. Those
policies were rejected for this rig. Chunking remains useful as an explicit
control for different host links, but stream priority alone is not an
admission controller.

The same speed screen measured 1,348.788 ms before the new controls,
1,344.163 ms with phased readiness, 1,339.991 ms after adding local-owner
bypass, and 1,321.334 ms after parallel submission. These are small-window
pp2048 results, not all-40 or production-promotion results.

## Local owner-return traffic

`GGML_CUDA_AW_PAIRFOLD_LOCAL_OWNER_BYPASS=1` lets the canonical ordered sum
read a home-local BF16 partial directly while remote logical owners still
use the receive arena. Logical pointer order and FP32 sum order remain
unchanged.

At pp2048 this removes 1,280 local 524,288-byte D2D copies per graph, or 640
MiB and 6.020 ms of summed copy duration in the terminal trace. The
pp8128-equivalent traffic is 2.480469 GiB. The combined path passes c512
exactness, but repeated bypass-on pp8128 pipeline runs are nondeterministic.
Adding an explicit direct-local wait did not repair the long-shape result.
The selector remains default off and is now restricted to the serial
scheduler; the pipelined runtime and harness reject it.

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

The mixed-residency measurements recover:

| Policy | GPU0/1 | GPU2/3 |
| --- | ---: | ---: |
| All-40 down-only | 1.840 GiB/GPU | 1.840 GiB/GPU |
| All-40 up+down | 4.475 GiB/GPU | 4.475 GiB/GPU |
| High capacity, keep 0/19/20 | 6.717 GiB/GPU | 6.318 GiB/GPU |

The high-capacity asymmetry follows the manifest-selected layer-to-pair
schedule: retaining layers 0, 19, and 20 does not retain the same projection
count on both physical pairs.

The proof fits the original 46 GiB host, but with little operating margin.
System-wide monitoring observed approximately 5.4 GiB minimum available
memory and effectively full 8 GiB swap. These samples are not a synchronized
process RSS high-water:

[loaderhost-all40-host-memory-monitor.txt](../../results/qwen36-35b-moe-pp-20260721/pairfold-20260726/loaderhost-all40-host-memory-monitor.txt)

Do not co-locate this all-40 configuration with another memory-heavy job on
a similarly sized host.

## Performance result

The original all-40 pp8128 process retained five warm samples after
discarding the initialization call:

```text
4222.524, 4213.449, 4238.061, 4246.818, 4239.941 ms
```

The median is 4,238.061 ms / 1,917.858 tok/s. Peak-to-peak spread is 0.788%
of the warm mean.

Relative to the current resident PairFold measurement, all-40 streaming:

- adds 1,211.023 ms, or 40.007% wall time;
- reduces throughput by 28.575%.

Relative to production throughput, it is 34.145% slower.

Every generation moves 34,225,520,640 bytes. The old runtime report produced
a 5.632 GiB/s median by summing per-layer pair envelopes. Pair copies can
overlap, so this is not physical root-link throughput or a hardware ceiling.
Runtime now labels that statistic `summed-layer-envelope-GiB/s`.

The available eight-layer pp2048 trace provides the correct physical
calculation: 6,845,104,128 H2D bytes over a 689.645 ms all-device interval
union, or 9.244 GiB/s aggregate. Four H2Ds overlap for 580.852 ms. The later
all-40 down-only trace measures 8.534 GiB/s over its own union, as recorded
below. The all-host result still needs its own interval-union trace before a
physical rate can be assigned to that policy. Use the H2D interval union,
concurrency histogram, destination-GPU SM overlap, native-copy overlap, and
kernel stretch rather than summing per-layer event durations.

Summed ready-event waits are 65.897-70.331 ms per warm generation. Most
weight readiness is hidden, but PCIe, copy-engine, and memory-system
contention remain visible to compute.

The earlier eight-layer bytes-linear estimate was 4,075.253 ms. The
measured all-40 wall is 162.808 ms, or 3.995%, slower.

The all-40 benchmark process took approximately 300 seconds end to end.
Only 26.898 seconds belonged to the six timed inference samples. Model load,
CPU T64 packing, initialization, and teardown consumed the remaining
approximately 273 seconds. CPU T64 packing accounted for 66.285 seconds.
This is a one-time cost for a persistent process.

The retained campaign measurements are:

| Policy | pp2048 median ms | pp8128 median ms | pp8128 tok/s |
| --- | ---: | ---: | ---: |
| Resident PairFold | 897.343 | not rerun | not rerun |
| All-40 down-only, safe bypass off | not rerun | 3,353.908 | 2,423.44 |
| All-40 down-only, prior unsafe bypass on | 1,430.897 | 3,320.382 | 2,447.91 |
| All-40 up+down | 2,295.943 | 3,553.836 | 2,287.11 |
| High capacity, keep 0/19/20 | 3,137.983 | 3,988.963 | 2,037.62 |
| All-40 all-host | not rerun | 4,131.291 | 1,967.42 |

The all-host rerun is 106.770 ms faster than the original all-host median.
Safe down-only is the best exactness-qualified capacity/performance
compromise on this rig, but it remains below both resident PairFold and
production throughput.

Except for the safe down-only row, every mixed-policy row above was measured
before the bypass was restricted to serial scheduling. Those timing and VRAM
results remain useful diagnostic estimates, but are not strict pp8128
exactness-qualified pipeline results. The safe down-only row uses the
all-40 down-only manifest with local-owner bypass disabled.

Write-combined pinned host allocation reduced the down-only pp2048 warm
median from 1430.897 to 1417.850 ms, but regressed pp8128 from 3320.382 to
3351.076 ms. It therefore remains a default-off diagnostic. Both arms used
the now-disallowed bypass-on pipeline configuration.

The all-40 down-only pp2048 trace moves 11,408,506,880 bytes over a
1,244.951171 ms H2D interval union, or 8.534 GiB/s aggregate. Concurrency
widths 1, 2, 3, and 4 occupy 1.454, 793.573, 5.580, and 444.345 ms.
Host-weight copies overlap any SM work for 72.706% of their union and
same-pair SM work for 34.914% / 36.127% on pairs 0 / 1. There is no
cross-stream same-GPU SM-kernel overlap.

At concurrency widths 1, 2, 3, and 4, proportional byte attribution within
each CUPTI DMA record gives aggregate rates of 3.819, 6.919, 9.349, and
11.424 GiB/s. Width 1 covers only 1.454 ms and should not be interpreted as
a stable single-copy benchmark.

Native copies slow while host H2D is active. Mean P2P latency is 885.752 us
versus 352.756 us outside host ingress for 4 MiB records, and 64.552 us
versus 22.177 us for 256 KiB records. Mean 256 KiB D2D latency is 11.471 us
versus 3.224 us. These ratios are 2.511x, 2.911x, and 3.558x and quantify
the contention that stream priority cannot prevent.

The analyzer also partitions every kernel name by any-host and
same-device-host H2D overlap. In this trace, same-device inside/outside mean
duration ratios include 2.837x for route packing, 2.698x for cuBLAS split-K
reduction, and 1.483x for the canonical local owner sum, with at least five
records in both buckets. These are descriptive trace ratios rather than a
controlled causal experiment because layer and work-size mix can differ
between the buckets.

Generation attribution is time-window based, so cross-generation prefetch
must be disabled for this analysis. The analyzer requires all four terminal
markers by default; its explicit legacy override is not used for qualified
results.

All measurements use the literal exact attention path. Packed pp8128
attention now matches Q, K, V, and mask staging, but FlashAttention output
and restored pregate still diverge. It remains default off and was not
timed.

The down-only host path also remains default off. Bypass-on pp8128 pipeline
hashes are nondeterministic, while bypass-disabled pipeline and bypass-on
serial controls match the resident literal output. The safe bypass-disabled
performance result is 2,423.44 tok/s, still below production. The c512
byte-exact result is preserved, but it did not expose this long-shape
pipeline race.

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

The equivalent tracked residency policy is:

```bash
POLICY="$PWD/docs/p100-pairwave-project/residency-policies/\
layers16-23-all-host.manifest"

env \
  PAIRFOLD_HOST_WEIGHTS=1 \
  PAIRFOLD_HOST_PLACEMENT=1 \
  PAIRFOLD_HOST_RESIDENCY_MANIFEST="$POLICY" \
  PAIRFOLD_HOST_PHASED=1 \
  bash bench/run-pairfold.sh \
    pipeline-ppl 512 external-residency-c512
```

Do not set `PAIRFOLD_HOST_LAYERS` in the same process as a residency
manifest.

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

This map describes the later experimental implementation, not files added
by the production-diagonal source checkpoint on this branch.

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
| `ggml/src/ggml-pairfold-residency.h` | strict projection-residency manifest parser and selection helpers |
| `tests/test-pairfold-scheduler.cpp` | scheduler, lease, packing, inverse, and malformed-contract tests |
| `tests/test-pairfold-residency.cpp` | manifest grammar, masks, range compatibility, and malformed-policy tests |
| `bench/run-pairfold.sh` | guarded original-rig qualification harness |
| `bench/analyze-pairfold-trace.py` | H2D union, concurrency, copy stretch, and per-kernel duration accounting |

## Decision

The architecture works:

- selected expert GPU allocations are absent;
- c512 output remains byte-identical;
- all 40 inference layers fit in pinned host memory;
- two bounded slots recover 7.109375 GiB/GPU;
- the retained two-pair scheduler can overlap compute and ingress.

It is not production-qualified on this P100 PCIe rig. The next useful work
is reducing bytes or copy/memory contention, improving resident PairFold
compute, using a faster host link, and restoring host-capacity margin. Any
new pp8128 pipeline campaign must keep local-owner bypass disabled.
