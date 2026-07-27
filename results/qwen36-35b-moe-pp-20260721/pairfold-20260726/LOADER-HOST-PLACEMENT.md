# PairFold loader-time host placement proof

Date: 2026-07-26

Source HEAD: `a7c878670`

## Outcome

Loader-time host placement works for both the initial PairWave expert window
and all 40 inference layers, `blk.0` through `blk.39`. The selected expert
tensors never receive device-resident model allocations. They remain in
portable pinned host buffers, are packed to T64 during model load, and feed
the existing two-slot PairFold H2D streamer. The model's MTP `blk.40` is
outside the PairFold inference schedule and remains resident.

The path is byte-exact at c512 and recovers 843,055,104 bytes (804 MiB) on
every GPU relative to the preceding resident control at the final PairWave
checkpoint, with both streaming slots live in the loader arm. Its warm
pp8128 median is 3289.595 ms, or 2470.821 tok/s.

The all-40 follow-up is also byte-exact at c512. It recovers 7,633,633,280
bytes (7,280 MiB, 7.109375 GiB) on every GPU. Its original warm pp8128
median was 4238.061 ms / 1917.858 tok/s; the current speed-path rerun
measures 4131.291 ms / 1967.42 tok/s.

Projection-selective placement is byte-exact at c512. Safe down-only, with
local-owner bypass disabled, measures 3353.908 ms / 2423.44 tok/s and
recovers 1.840 GiB/GPU. The earlier bypass-on campaign measured down-only at
3320.382 ms / 2447.91 tok/s, up+down at 3553.836 ms / 2287.11 tok/s, and
high capacity at 3988.963 ms / 2037.62 tok/s. Those earlier pp8128 rows are
diagnostic because the local-owner bypass is nondeterministic under the
pipeline.

This proves the full expert-host-resident architecture and substantial VRAM
recovery. It remains default-off and experimental because it is below both
resident PairFold and production throughput, and it leaves little host
memory margin on this 46 GiB machine.

## Implementation

The additional default-off selector is:

`GGML_CUDA_AW_PAIRFOLD_HOST_PLACEMENT=1`

It requires:

- `GGML_CUDA_AW_PAIRFOLD_HOST_WEIGHTS=1`;
- either an inclusive `GGML_CUDA_AW_PAIRFOLD_HOST_LAYERS` range or an
  absolute `GGML_CUDA_AW_PAIRFOLD_HOST_RESIDENCY_MANIFEST`;
- the qualified four-device tensor-parallel Qwen MoE model;
- Q8_0 expert weights and T64 layout;
- `--no-mmap`;
- exact PairFold service, output, HeadFold GDN, and HeadFold attention
  selectors.

Selected `blk.N.ffn_{gate,up,down}_exps.weight` tensors use a dedicated
four-way meta buffer. Its physical buffers are CUDA-pinned host buffers, but
the outer meta buffer is intentionally not reported as directly host
addressable so the loader enters the normal meta set path. That preserves:

- the existing PairWave 128/128/0/0 expert-axis split;
- the manifest-selected physical pair;
- the existing PairWave expert permutation;
- the exact logical-to-physical expert order.

After each physical projection is gathered, a CPU implementation of the
qualified Q8_0-to-T64 byte permutation packs it in place and registers it in
a host-only catalog. Host pointers are never inserted in the ordinary
device-T64 set. The CPU helper is independently tested at the exact gate/up
and down shapes, including inverse reconstruction and malformed contracts.

For the legacy range, the streamer borrows three model-owned host projection
pointers per layer/device. For a residency manifest it borrows only selected
host projections and uses validated resident T64 pointers for the
complement. Selected H2Ds land at the same down/gate/up offsets in the
stable 408 MiB slot. `GGML_CUDA_AW_PAIRFOLD_HOST_PHASED=1` publishes
gate/up/down readiness independently; without it, gate/up retains the
conservative combined dependency. There is no loader D2H snapshot. Pinned
allocation fallback is fatal. Unsupported, ragged, decode, diagonal, or
non-service graphs fail before ordinary graph execution.

The proof supports one loaded model and inference context per process.
Concurrent duplicate registration is rejected, model-buffer destruction
removes the host catalog entries and invalidates borrowed pointers, and a
second placement model after preparation is rejected. Model destruction
while preparation or evaluation is in flight is outside this proof's
single-job lifetime contract.

## Exactness

The preceding resident control and loader-host c512 run both report:

- PPL: `4.0783`;
- saved-logits SHA-256:
  `47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`.

The loader registered 48 physical projections and 6,845,104,128 host bytes.
CPU T64 packing took 9.650 seconds in the c512 process. The runtime reported
`snapshot-ms=0.000`.

The later manifest qualification repeats the same accepted PPL and SHA-256
for:

- legacy-equivalent layers 16:23 all-host;
- all-40 down-only;
- all-40 up+down;
- high capacity, keeping layers 0, 19, and 20 resident.

Artifacts:

- `loaderhost-resident-c512-v1.out` / `.err`;
- `loaderhost-resident-c512-v1-logits.bin`;
- `loaderhost-c512-l16-23-v3.out` / `.err`;
- `loaderhost-c512-l16-23-v3-logits.bin`;
- `exact-residency-l16-23-c512-v1.out` / `.err`;
- `exact-residency-down-only-c512-v1.out` / `.err`;
- `exact-residency-up-down-c512-v1.out` / `.err`;
- `exact-residency-high-cap-c512-v1.out` / `.err`.

The pp8128 isolation found no loader or projection-streaming arithmetic
error. A down-only serial run with local-owner bypass enabled and a down-only
pipeline run with bypass disabled both report PPL 6.7466 and resident
SHA-256:

`8cf55039bf0107fee294dba9d3c4413d61c5e1cac4f170c68a3f66303520f86e`

In contrast, bypass-on pipeline runs produced four different SHA-256 values:

- `aabb2ba6feb49f8832a3eb887d7665ab7a444190a98e21d4ce4074a42449e214`;
- `9fdbf2f2f4a94356409cef289e85bc842a6653a332e0dca101a689fa347934de`;
- `1fbec941e0076e63211fa7554db4d031a372b09db5c43b080ca6ad185a5559fe`;
- `e734393e0dae263ba97f70e1851699437c66ed079e4389f5a8c818fa2304bb8b`
  after the direct-local wait was added.

The runtime and harness now allow local-owner bypass only with the serial
scheduler. The production-safe pipeline configuration uses bypass disabled.

The safe down-only timing run retained these five pp8128 walls after
discarding the cold sample:

```text
3372.835585, 3353.908010, 3329.440313, 3368.096146, 3330.874014 ms
```

Its median is 3,353.908010 ms / 2,423.44 tok/s while reclaiming 1.840
GiB/GPU. It is 1.01% slower in wall time than the unsafe 3,320.382 ms
bypass-on result, 9.75% lower in throughput than resident PairFold at
3,027.038 ms / 2,685.133 tok/s, and 16.79% lower than production at
2,790.956 ms / 2,912.264 tok/s.

Artifacts:

- `exact-residency-down-only-safe-pp8128-v1.out` / `.err`;
- `perf-residency-down-only-safe-pp8128-v1.out` / `.err`.

## VRAM recovery

The table uses the final `cudaMemGetInfo` PairWave checkpoint after PairFold
hidden storage and PairWave scratch are allocated. The loader-host arm also
has both host-weight slots allocated.

| GPU | Resident bytes | Loader-host bytes | Recovered bytes |
| ---: | ---: | ---: | ---: |
| 0 | 11,119,820,800 | 10,276,765,696 | 843,055,104 |
| 1 | 11,061,231,616 | 10,218,176,512 | 843,055,104 |
| 2 | 11,489,050,624 | 10,645,995,520 | 843,055,104 |
| 3 | 11,489,050,624 | 10,645,995,520 | 843,055,104 |

The nominal two-slot storage is 855,638,016 bytes/GPU. The measured complete
slot allocation is 857,735,168 bytes/GPU, while the complete path retains
additional stream/event/runtime overhead. The observed 804 MiB recovery is
12 MiB below the nominal 816 MiB estimate.

This is 0.785156 GiB/GPU, so the 16:23 window narrowly misses a strict
`>= 0.8 GiB` gate by about 15.2 MiB. It nevertheless proves real recovery.
A larger balanced window is required if that binary threshold is retained.

At pp8128, the same loader path reports:

| GPU | Resident bytes | Loader-host bytes | Recovered bytes |
| ---: | ---: | ---: | ---: |
| 0 | 13,416,202,240 | 12,573,147,136 | 843,055,104 |
| 1 | 13,357,613,056 | 12,514,557,952 | 843,055,104 |
| 2 | 13,785,432,064 | 12,942,376,960 | 843,055,104 |
| 3 | 13,785,432,064 | 12,942,376,960 | 843,055,104 |

Compared with the earlier shadow-stream path, which retained the originals
and the slots, loader placement removes exactly 1,711,276,032 bytes
(1632 MiB) per GPU.

Mixed-residency recovery at both c512 and pp8128 is:

| Policy | GPU0/1 recovered | GPU2/3 recovered |
| --- | ---: | ---: |
| All-40 down-only | 1,975,517,184 bytes / 1.840 GiB | 1,975,517,184 bytes / 1.840 GiB |
| All-40 up+down | 4,804,575,232 bytes / 4.475 GiB | 4,804,575,232 bytes / 4.475 GiB |
| High capacity, keep 0/19/20 | 7,212,105,728 bytes / 6.717 GiB | 6,784,286,720 bytes / 6.318 GiB |

The high-capacity asymmetry follows the PairWave layer-to-pair schedule.

## Warm pp8128 performance

Sample 0 is initialization and is excluded. The five loader-host warm walls
are:

`3276.769, 3373.924, 3289.595, 3273.122, 3355.532 ms`

| Path | Median wall ms | tok/s |
| --- | ---: | ---: |
| Qualified production diagonal | 2790.956 | 2912.264 |
| Warm resident PairFold reported by the current campaign | 3027.038 | 2685.133 |
| Preceding resident PairFold control | 3040.948 | 2672.85 |
| Shadow-stream layers 16:23 | 3247.809 | 2502.61 |
| Loader-host layers 16:23 | 3289.595 | 2470.821 |

Loader placement is 41.786 ms, or 1.287% wall time, slower than the
shadow-stream median. This is consistent with using three independently
allocated projection sources instead of one contiguous shadow snapshot.

Relative to the preceding resident PairFold control, loader-host streaming
adds 248.647 ms, or 8.177% wall time, and reduces throughput by 7.559%.
Relative to the qualified production comparator, throughput is 15.158%
lower.

Across generations 1 through 5, the summed gate/up and down ready-event
waits are only 0.237-0.273 ms. The weights arrive on time; transfer and
memory-system contention remain the performance cost.

Artifacts:

- `loaderhost-warm-pp8128-l16-23-v1.out` / `.err`;
- resident control: `hoststream-resident-control-pp8128-v1.out` / `.err`;
- shadow-stream control:
  `hoststream-final-default-warm-pp8128-l16-23-v5.out` / `.err`.

The retained projection-policy measurements discard sample 0 and report:

| Policy | pp2048 median ms | pp8128 median ms | pp8128 tok/s |
| --- | ---: | ---: | ---: |
| Resident PairFold | 897.343 | not rerun | not rerun |
| All-40 down-only, safe bypass off | not rerun | 3,353.908 | 2,423.44 |
| All-40 down-only, prior unsafe bypass on | 1,430.897 | 3,320.382 | 2,447.91 |
| All-40 up+down | 2,295.943 | 3,553.836 | 2,287.11 |
| High capacity, keep 0/19/20 | 3,137.983 | 3,988.963 | 2,037.62 |
| All-40 all-host | not rerun | 4,131.291 | 1,967.42 |

Except for safe down-only, every streamed row used the bypass-on pipeline
before it was restricted to serial scheduling. Those numbers remain useful
performance/capacity estimates, but are not strict pp8128
exactness-qualified results.

## All-40 follow-up

The expanded selector is:

`GGML_CUDA_AW_PAIRFOLD_HOST_LAYERS=0:39`

This covers all 40 inference layers. The loader registers 240 physical
projections totaling 34,225,520,640 pinned host bytes (31.875 GiB). The
c512 process spent 71.858 seconds in CPU T64 packing, and the pp8128 process
spent 66.285 seconds. Both report `snapshot-ms=0.000`.

The all-40 c512 run reports PPL 4.0783 and the accepted saved-logits
SHA-256:

`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`

Artifacts:

- `loaderhost-all40-c512-v1.out` / `.err`;
- `loaderhost-all40-c512-v1-logits.bin`;
- `loaderhost-all40-warm-pp8128-v1.out` / `.err`;
- `loaderhost-all40-host-memory-monitor.txt`.

At the final c512 PairWave checkpoint:

| GPU | Resident bytes | All-40 host bytes | Recovered bytes |
| ---: | ---: | ---: | ---: |
| 0 | 11,119,820,800 | 3,486,187,520 | 7,633,633,280 |
| 1 | 11,061,231,616 | 3,427,598,336 | 7,633,633,280 |
| 2 | 11,489,050,624 | 3,855,417,344 | 7,633,633,280 |
| 3 | 11,489,050,624 | 3,855,417,344 | 7,633,633,280 |

At the final pp8128 PairWave checkpoint:

| GPU | Resident bytes | All-40 host bytes | Recovered bytes |
| ---: | ---: | ---: | ---: |
| 0 | 13,416,202,240 | 5,782,568,960 | 7,633,633,280 |
| 1 | 13,357,613,056 | 5,723,979,776 | 7,633,633,280 |
| 2 | 13,785,432,064 | 6,151,798,784 | 7,633,633,280 |
| 3 | 13,785,432,064 | 6,151,798,784 | 7,633,633,280 |

The five retained warm pp8128 walls are:

`4222.524, 4213.449, 4238.061, 4246.818, 4239.941 ms`

Their median is 4238.061 ms / 1917.858 tok/s. Peak-to-peak spread is
0.788% of the warm mean.

| Path | Median wall ms | tok/s |
| --- | ---: | ---: |
| Qualified production diagonal | 2790.956 | 2912.264 |
| Warm resident PairFold reported by the current campaign | 3027.038 | 2685.133 |
| Loader-host layers 16:23 | 3289.595 | 2470.821 |
| Loader-host layers 0:39 | 4238.061 | 1917.858 |

Relative to the current warm resident PairFold result, all-40 streaming adds
1211.023 ms, or 40.007% wall time, and reduces throughput by 28.575%.
Relative to production, throughput is 34.145% lower. Expanding from the
eight-layer loader window adds 948.466 ms, or 28.832% wall time.

Each generation moves 34,225,520,640 bytes. The old report's 5.632 GiB/s
median sums per-layer pair envelopes and is not physical root-link
throughput because pair copies can overlap. Nsight interval-union and
concurrency analysis is required for a physical aggregate rate. Summed
gate/up and down ready-event waits are 65.897-70.331 ms per generation. Most
ingress is still ready before use, but unlike the eight-layer window it is
no longer fully hidden.

The earlier bytes-linear estimate was 4075.253 ms / 1994.5 tok/s. Measured
all-40 wall time is 162.808 ms, or 3.995%, slower than that estimate.

The pp8128 benchmark process ran for approximately 300 seconds. Its six
timed samples account for 26.898 seconds, leaving approximately 273 seconds
in model load, initialization, and process teardown. CPU T64 packing
accounts for 66.285 seconds of that untimed work. This is a one-time cold
start cost for a persistent process, but it is material on this host.

Host capacity is the other constraint. Preflight reported 41,471,455,232
bytes available. Read-only samples during allocation and packing saw
available memory fall as low as approximately 5.4 GiB, and the machine's
8 GiB swap became effectively full by the second process. Both runs
completed, but this configuration should not share the host with another
memory-heavy workload. These system-wide samples are recorded in
`loaderhost-all40-host-memory-monitor.txt`; they are not an exact process
high-water measurement.

The current all-host rerun retains warm walls of 4190.715, 4131.291,
4128.041, 4131.289, and 4135.033 ms. Its median is 4131.291 ms / 1967.42
tok/s, 106.770 ms faster than the original measurement.

## All-40 down-only pp2048 trace

The trace artifact is:

`trace-residency-down-only-pp2048-v1.nsys-rep`

It contains 11,408,506,880 host-weight H2D bytes in a 1244.951171 ms
all-device union, or 8.534 GiB/s aggregate. Concurrency widths 1/2/3/4
occupy 1.454/793.573/5.580/444.345 ms. Host H2D overlaps any SM work for
72.706% of its union and same-pair SM work for 34.914% / 36.127% on pairs
0 / 1. There is no cross-stream same-GPU SM-kernel overlap.

Mean native-copy latency during versus outside host ingress is:

| Native record | Outside host H2D | During host H2D | Ratio |
| --- | ---: | ---: | ---: |
| P2P, 4 MiB | 352.756 us | 885.752 us | 2.511x |
| P2P, 256 KiB | 22.177 us | 64.552 us | 2.911x |
| D2D, 256 KiB | 3.224 us | 11.471 us | 3.558x |

The trace demonstrates that copy/HBM/PCIe contention remains material even
when ready-event waits are small. CUDA stream priority cannot preempt an
in-flight transfer.

## Decision

The proof succeeds:

- loader-owned host placement preserves PairWave sharding and c512
  exactness;
- selected expert GPU allocations are genuinely absent;
- the existing inactive-pair streaming schedule consumes them correctly;
- the measured VRAM reduction is stable across c512 and pp8128;
- all 40 inference layers fit and the c512 result remains byte-exact;
- the current all-40 all-host warm rate is approximately 1967 tok/s;
- safe down-only is the best exactness-qualified mixed policy at
  approximately 2423 tok/s.

The path remains default-off. The layer-range expansion is complete.
Performance work should focus on PairFold itself and on reducing
copy/memory contention or transferred bytes. A larger-memory host would
also restore operational headroom; faster ingress would address the more
important throughput limit.

The initial pp8128 down-only mismatch was not stable: repeated bypass-on
pipeline runs produced different hashes. Bypass-disabled pipeline and
bypass-on serial controls both match the resident PPL and hash. This
localizes the blocker to the owner shortcut under pipeline overlap, not the
loader placement. The measured production-safe result keeps bypass disabled
and reaches 3,353.908 ms / 2,423.44 tok/s.

All performance measurements use literal exact attention. Packed pp8128
attention matches Q, K, V, and mask staging but still diverges at
FlashAttention output and restored pregate. It remains default off and
untimed.

No sub-Q8 weights were used or evaluated.
