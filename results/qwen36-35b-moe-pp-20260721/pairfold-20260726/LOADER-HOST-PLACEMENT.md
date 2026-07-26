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
bytes (7,280 MiB, 7.109375 GiB) on every GPU and has a warm pp8128 median of
4238.061 ms, or 1917.858 tok/s.

This proves the full expert-host-resident architecture and substantial VRAM
recovery. It remains default-off and experimental because it is below both
resident PairFold and production throughput, and it leaves little host
memory margin on this 46 GiB machine.

## Implementation

The additional default-off selector is:

`GGML_CUDA_AW_PAIRFOLD_HOST_PLACEMENT=1`

It requires:

- `GGML_CUDA_AW_PAIRFOLD_HOST_WEIGHTS=1`;
- an inclusive `GGML_CUDA_AW_PAIRFOLD_HOST_LAYERS` range;
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

At runtime, the streamer borrows three model-owned host projection pointers
per layer/device and issues three H2D copies into the stable 408 MiB slot.
There is no loader D2H snapshot. Pinned allocation fallback is fatal.
Unsupported, ragged, decode, diagonal, or non-service graphs fail before
ordinary graph execution.

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

Artifacts:

- `loaderhost-resident-c512-v1.out` / `.err`;
- `loaderhost-resident-c512-v1-logits.bin`;
- `loaderhost-c512-l16-23-v3.out` / `.err`;
- `loaderhost-c512-l16-23-v3-logits.bin`.

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

Each generation moves 34,225,520,640 bytes. Across warm generations, the
reported serialized-pair rate has a 5.632 GiB/s median. Summed gate/up and
down ready-event waits are 65.897-70.331 ms per generation. Most ingress is
still ready before use, but unlike the eight-layer window it is no longer
fully hidden.

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

## Decision

The proof succeeds:

- loader-owned host placement preserves PairWave sharding and exactness;
- selected expert GPU allocations are genuinely absent;
- the existing inactive-pair streaming schedule consumes them correctly;
- the measured VRAM reduction is stable across c512 and pp8128;
- all 40 inference layers fit and remain byte-exact;
- the all-40 warm rate is approximately 1918 tok/s on this P100 PCIe rig.

The path remains default-off. The layer-range expansion is complete.
Performance work should focus on PairFold itself and on reducing
copy/memory contention or transferred bytes. A larger-memory host would
also restore operational headroom; faster ingress would address the more
important throughput limit.

No sub-Q8 weights were used or evaluated.
