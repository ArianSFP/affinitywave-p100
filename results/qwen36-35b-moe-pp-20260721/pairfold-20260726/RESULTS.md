# PairFold runtime results

Date: 2026-07-26

Source base: `619e7cc7c`

Qualified arithmetic source: `36bad6bb3`

## Runtime implementation

The default-off `GGML_CUDA_AW_PAIRFOLD=1` path now has:

- an 80-task `(layer, panel, pair, generation)` scheduler;
- chronology, hidden, and pair-resource dependencies;
- a serial arithmetic control and a two-pair pipelined scheduler;
- two generation banks of task and terminal events;
- two stable F32 hidden handoff slots per GPU;
- pair-local recurrent preparation and exact two-lane GDN execution;
- pair-local literal exact attention with chronological K/V exchange;
- the lane-exact PairWave service restricted to the active pair;
- explicit BF16 owner-return DMA followed by a home-local ordered sum;
- active-pair-only joins and publication;
- unsupported/ragged-shape fallback diagnostics;
- manifest, schedule, task, generation, mode, scratch, and memory diagnostics;
- NVTX task, layer, panel, pair, stage, copy, and generation ranges.

The selector accepts only four P100 GPUs and lane sizes 128, 512, and 2032,
corresponding to c512, pp2048, and pp8128. It also requires the absolute
PairWave manifest, exact dense selectors, CohortRail, Q8 T64 layout, F32
wire, BF16 owner partials, deterministic routes, and P100-exact arithmetic.

## Exactness

The serial c512 control and pipelined c512 path both report PPL 4.0783 and
produce the accepted saved-logits SHA-256:

`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`

The owner-return revision produces the same SHA-256. Serial and pipeline
saved-logits files are byte-identical.

The final checkpoint candidate binary was re-run after the trace-only
terminal marker was added. It again reported PPL 4.0783 and produced the
accepted saved-logits SHA-256. Its curated text artifact is
`checkpoint-exact-pipeline-c512-v1.out`; the generated diagnostic log and
logits binary remain local.

All 640 common c512 PairFold service artifacts match the same-build diagonal
oracle byte-for-byte. These cover all 160 layer/lane service boundaries and
their F32 service input, route IDs, route weights, and reduced F32 output.

The CPU scheduler test covers all 80 tasks, dependency and pair-resource
edges, the layer-19/20 repeated-pair boundary, randomized delayed
completions, handoff/scratch-slot consumption, two banks, and repeated
generation wraparound. A three-repetition runtime process completed
generations 0, 1, and 2, including bank-0 reuse. Its two post-initialization
c512 samples were 428.439 and 425.515 ms.

## pp2048 trace

Artifact:

`trace-pipeline-pp2048-terminal-v3.nsys-rep`

Analysis:

`trace-pipeline-pp2048-terminal-v3-analysis.json`

The profiled benchmark wall was 2,025.835 ms (1,010.941 tok/s). The 80-task
window, measured through four trace-only terminal marker kernels, was
1,247.053 ms. An earlier analysis that ended 10 ms after the final host task
range was rejected because it omitted about 105 ms of queued terminal GPU
work.

Per-GPU SM busy union:

| GPU | Busy ms | Busy % | Idle ms | Cross-stream SM overlap ms |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 625.314 | 50.143 | 621.739 | 0 |
| 1 | 622.631 | 49.928 | 624.423 | 0 |
| 2 | 623.528 | 50.000 | 623.525 | 0 |
| 3 | 630.275 | 50.541 | 616.778 | 0 |

Time by number of active GPUs:

| Active GPUs | Time ms |
| ---: | ---: |
| 0 | 331.780 |
| 1 | 152.458 |
| 2 | 281.914 |
| 3 | 138.141 |
| 4 | 342.759 |

Time by active physical pairs was 331.780 ms with no pair active, 396.139
ms with one pair active, and 519.134 ms with both pairs active. Host task
ranges show 454.135 ms with two active pairs.

The trace contains 8,224,997,856 copy bytes over a 261.663 ms copy union.
59.299 ms of that union is outside all SM work, for 22.662% measured copy
exposure. This replaces the replay's assumed 25%.

There are 640 `aw_pair_sum_groups_local_panel_p100` kernels, no peer-load
kernel, and no same-GPU cross-stream SM overlap. CUPTI graph-node timestamps
report only 0.017-0.026 ms of same-stream interval overlap per GPU.

## Memory

At pp8128 the explicit PairWave scratch arena is 199,793,408 bytes/GPU.
The two requested F32 hidden slots are 33,292,288 bytes/GPU.

The in-process CUDA free-memory delta from immediately before PairFold
generation allocation to the post-scratch high-water point is 289,406,976
bytes on every GPU. This includes CUDA allocation granularity, events, live
metadata, the hidden slots, and PairWave scratch. The largest absolute
reported used-memory high-water is 13,785,432,064 bytes.

Against the documented 1,166,661,424-byte legacy live state, the measured
runtime delta reclaims 877,254,448 bytes/GPU, or 0.817007 GiB/GPU. This
passes the 0.8 GiB floor. It does not claim the replay-only
131,017,728-byte arena.

## Randomized performance campaign

Every arm in this original campaign was a fresh process with
`--no-warmup -r 1`. These results measure first-call behavior and the
same-condition PairFold delta, but they are not steady-state throughput
samples.

pp2048 walls in milliseconds:

| Pair | PairFold | Diagonal | PairFold delta |
| ---: | ---: | ---: | ---: |
| 1 | 2131.181 | 2025.181 | +5.234% |
| 2 | 2106.356 | 1993.206 | +5.677% |
| 3 | 2097.128 | 1986.037 | +5.594% |
| 4 | 1787.150 | 1833.829 | -2.545% |
| 5 | 2301.730 | 1949.680 | +18.057% |

The PairFold median is 2,106.356 ms and the diagonal median is 1,986.037
ms, a 6.058% PairFold regression. The arm dispersion is material.

pp8128 walls in milliseconds:

| Pair | PairFold | Diagonal | PairFold delta |
| ---: | ---: | ---: | ---: |
| 1 | 4042.305 | 3830.735 | +5.523% |
| 2 | 4050.309 | 3899.365 | +3.871% |
| 3 | 4142.574 | 3967.911 | +4.402% |

All pp8128 PairFold arms are within 1.574% of their 4,078.396 ms mean. All
diagonal arms are within 1.759% of their 3,899.337 ms mean. The PairFold
median is 4,050.309 ms (2,006.761 tok/s), 45.123% above the 2,790.956 ms
production wall gate when compared as a cold sample. Every paired pp8128
result is a material regression.

## Warm pp8128 qualification

A later matched measurement used six repetitions in each process with the
same production environment, build, model, manifest, CPU affinity, lock, and
watchdog. Sample 0 was the full-shape initialization and graph-capture pass.
Samples 1 through 5 were same-process warm evaluations; the JSON aggregate
was not used because it includes sample 0.

PairFold walls in milliseconds:

`4120.189 cold; 3027.038, 3037.403, 3023.974, 3044.781, 3025.481 warm`

The PairFold warm median is 3,027.038 ms, or 2,685.133 tok/s. Its warm
peak-to-peak spread is 0.687%.

Same-build diagonal walls in milliseconds:

`3850.423 cold; 2815.923, 2797.259, 2783.959, 2788.594, 2786.640 warm`

The diagonal warm median is 2,788.594 ms, or 2,914.731 tok/s. Its warm
peak-to-peak spread is 1.146%, and it reproduces the qualified production
comparator of 2,790.956 ms / 2,912.264 tok/s within 0.085% wall time.

PairFold is 238.444 ms, or 8.551% wall time, slower than the matched warm
diagonal median. Against the qualified production comparator it is
236.082 ms, or 8.459% wall time, slower and delivers 7.799% fewer tokens per
second. Generations 0 through 5 completed, including repeated reuse of both
event banks, with no fallback diagnostic.

Artifacts:

- `warm-pipeline-pp8128-v1.out` / `.err`
- `warm-diagonal-pp8128-v1.out` / `.err`

## Qualification decision and remaining gap

PairFold remains a default-off experimental path. It is not promoted to
production because its warm pp8128 median remains 236.082 ms above the
production wall gate. The original 4,050.309 ms result must not be quoted as
steady-state PairFold throughput; the qualified warm estimate is
3,027.038 ms / 2,685.133 tok/s.

The qualified attention path is the literal exact two-lane control. A
default-off pp8128 `c0+c3 / c1+c2` and `c4+c7 / c5+c6` packed
redistribution now exists. Q, K, V, and mask staging match the literal
control, but the FlashAttention output and restored pregate still diverge.
The packed path therefore remains default off and untimed. Standalone
retained-descriptor recurrent, attention, and composed endpoint probe
executables were not added; qualification used the end-to-end runtime,
service dumps, scheduler test, and trace instead.

The checkpoint commit includes source, tests, harnesses, this result record,
and selected small text evidence. Generated logits, binary service dumps,
and Nsight databases remain local. No push or PR is part of the checkpoint
procedure.

## Host-resident expert-streaming proof of concept

A subsequent default-off proof streams the raw Q8 T64 expert weights for
layers 16 through 23 from pinned host snapshots into two 408 MiB slots per
GPU. It is byte-exact at c512. The accepted saved-logits SHA-256 remains:

`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`

The warm pp8128 median is 3247.809 ms / 2502.61 tok/s, versus 3040.948 ms /
2672.85 tok/s for a same-binary resident PairFold control. Eight streamed
layers therefore add 206.861 ms or 6.802% wall time. Ready-event waits total
only about 0.24 ms per warm generation, so the penalty is transfer/compute
contention rather than late weights.

A pp2048 trace measures 6.375 GiB of host-weight copies in a 689.645 ms
all-device union. 54.567% of that union overlaps SM work. The current proof
retains the original resident weights and therefore adds about 828 MiB at
the worst GPU high-water rather than reclaiming memory. Loader-time host
placement would theoretically net 7.171875 GiB/GPU for all 40 Q8 layers
after retaining two slots.

The implementation, measurements, trace analysis, memory accounting, and
approximately 1994.5 tok/s full-Q8 first-order estimate are documented in
`HOST-STREAMING.md`. The path remains default-off and experimental.

## Loader-time host-placement follow-up

PairWave expert projections for layers 16 through 23 can now be allocated in
CUDA-pinned host memory at load time instead of receiving device-resident
model allocations. The loader preserves the four-way expert split and exact
PairWave permutation, CPU-packs each physical projection to T64 in place,
and supplies those buffers directly to the existing two-slot streamer.
There is no D2H snapshot.

The loader-host c512 run reports PPL 4.0783 and the accepted saved-logits
SHA-256:

`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`

Relative to the preceding resident control, the final PairWave memory
checkpoint recovers 843,055,104 bytes (804 MiB, 0.785156 GiB) on every GPU.
The loader arm includes both allocated streaming slots. The proof therefore
demonstrates actual VRAM recovery, while its eight-layer window remains about
15.2 MiB short of a literal 0.8 GiB/GPU gate.

The five warm pp8128 walls are 3276.769, 3373.924, 3289.595, 3273.122, and
3355.532 ms. Their median is 3289.595 ms / 2470.821 tok/s. This is 1.287%
slower in wall time than the prior shadow-source stream, 8.177% slower than
the preceding resident PairFold control, and 15.158% below the production
throughput comparator.

The feature remains default-off and experimental. The full record is in
`LOADER-HOST-PLACEMENT.md`.

## All-40 loader placement

The loader-host range was expanded to all PairFold inference layers,
`blk.0` through `blk.39`. The model's MTP `blk.40` remains outside the
PairFold inference schedule. The loader registers 240 physical projections
and 34,225,520,640 pinned host bytes, with `snapshot-ms=0.000`.

The all-40 c512 run reports PPL 4.0783 and the accepted saved-logits
SHA-256:

`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`

The final c512 used-memory values are 3,486,187,520 / 3,427,598,336 /
3,855,417,344 / 3,855,417,344 bytes. The final pp8128 values are
5,782,568,960 / 5,723,979,776 / 6,151,798,784 / 6,151,798,784 bytes.
Both shapes recover exactly 7,633,633,280 bytes (7,280 MiB, 7.109375 GiB)
per GPU relative to their preceding resident controls.

The all-40 warm pp8128 walls are 4222.524, 4213.449, 4238.061, 4246.818,
and 4239.941 ms after discarding initialization. The median is
4238.061 ms / 1917.858 tok/s and peak-to-peak spread is 0.788% of the warm
mean. Relative to the current warm resident PairFold result, all-40
streaming adds 1211.023 ms or 40.007% wall time and reduces throughput by
28.575%. It is 34.145% below the production throughput comparator.

Each warm generation moves 34,225,520,640 bytes. The old report's
5.632 GiB/s median is a sum of per-layer pair envelopes, not physical
root-link throughput; pair copies can overlap. Nsight interval-union and
concurrency analysis is required for a physical aggregate rate. Summed
ready-event waits are 65.897-70.331 ms per generation. The earlier
4075.253 ms bytes-linear estimate was 3.995% faster than the measured result.

The pp8128 process took approximately 300 seconds end to end; the six timed
samples account for 26.898 seconds. Model load, initialization, and teardown
therefore consumed approximately 273 seconds, including 66.285 seconds of
CPU T64 packing.

The host-capacity proof succeeds but has little operating margin. Read-only
samples saw available memory fall to approximately 5.4 GiB, and the
machine's 8 GiB swap became effectively full. The path remains default-off.
Artifacts and full accounting are in `LOADER-HOST-PLACEMENT.md`.

## Projection-residency speed follow-up

Date: 2026-07-27

The strict projection-residency manifest was tested with phased gate/up/down
readiness, home-local owner bypass, parallel submission, and literal exact
attention. Four c512 policies report PPL 4.0783 and the accepted logits
SHA-256:

`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`

They are:

- legacy-equivalent layers 16:23 all-host;
- all-40 down-only;
- all-40 up+down;
- high capacity, with layers 0, 19, and 20 resident and all other
  projections hosted.

The pp8128 saved-logits investigation found that the pipeline configuration
used for these performance runs was nondeterministic. Down-only pipeline
runs with home-local owner bypass enabled produced:

- `aabb2ba6feb49f8832a3eb887d7665ab7a444190a98e21d4ce4074a42449e214`;
- `9fdbf2f2f4a94356409cef289e85bc842a6653a332e0dca101a689fa347934de`;
- `1fbec941e0076e63211fa7554db4d031a372b09db5c43b080ca6ad185a5559fe`;
- `e734393e0dae263ba97f70e1851699437c66ed079e4389f5a8c818fa2304bb8b`
  after adding the direct-local wait.

Disabling parallel graph submission did not remove the failure. Disabling
local-owner bypass did: the down-only pipeline reported PPL 6.7466 and
SHA-256
`8cf55039bf0107fee294dba9d3c4413d61c5e1cac4f170c68a3f66303520f86e`,
byte-identical to the resident literal control. A serial-scheduler down-only
run with bypass enabled produced the same PPL and hash. This clears host
packing, projection selection, H2D publication, and local PairWave
arithmetic, and localizes the defect to the direct-local pointer under
two-pair overlap.

The runtime and harness now restrict local-owner bypass to the serial
scheduler. Pipeline qualification must use bypass disabled.

The final bypass-disabled all-40 down-only timing run discarded the cold
sample and retained:

```text
3372.835585, 3353.908010, 3329.440313, 3368.096146, 3330.874014 ms
```

Its median is 3,353.908010 ms / 2,423.44 tok/s and it reclaims 1.840
GiB/GPU. It is 33.526 ms, or 1.01%, slower than the prior unsafe bypass-on
3,320.382 ms / 2,447.91 tok/s result. Against resident PairFold at 3,027.038
ms / 2,685.133 tok/s it adds 326.870 ms and loses 9.75% throughput. Against
production at 2,790.956 ms / 2,912.264 tok/s it is 16.79% lower in
throughput.

The six-repetition pp2048 campaign discarded sample 0:

| Path | Warm median ms |
| --- | ---: |
| Resident PairFold | 897.343 |
| All-40 down-only, prior unsafe bypass on | 1,430.897 |
| All-40 up+down | 2,295.943 |
| High capacity, keep 0/19/20 | 3,137.983 |

The retained pp8128 policy measurements are:

| Path | Warm median ms | tok/s |
| --- | ---: | ---: |
| All-40 down-only, safe bypass off | 3,353.908 | 2,423.44 |
| All-40 down-only, prior unsafe bypass on | 3,320.382 | 2,447.91 |
| All-40 up+down | 3,553.836 | 2,287.11 |
| High capacity, keep 0/19/20 | 3,988.963 | 2,037.62 |
| All-40 all-host, current | 4,131.291 | 1,967.42 |
| All-40 all-host, original | 4,238.061 | 1,917.858 |

Except for the safe down-only row, the mixed-policy performance rows above
were measured with the bypass-on pipeline configuration before that
restriction. They remain useful diagnostic timing/capacity estimates, but
are not strict pp8128 exactness-qualified results.

The current all-host path improves by 106.770 ms over the original
measurement. Safe down-only is the best exactness-qualified
performance/capacity compromise, but its 2,423.44 tok/s remains 16.79%
below the 2,912.264 tok/s production comparator.

A write-combined host-allocation screen reduced down-only pp2048 from
1430.897 to 1417.850 ms, then regressed pp8128 from 3320.382 to 3351.076
ms. This screen also used the bypass-on pipeline. The option remains
default off and is rejected for the pp8128 path.

The chunk and pair/global serialization selectors are measurement controls,
not dynamic critical-path admission. Chunks remain back-to-back on their
H2D stream, and pair/global serialization operates at whole-layer
granularity.

VRAM recovery is stable between c512 and pp8128:

| Policy | GPU0/1 recovered | GPU2/3 recovered |
| --- | ---: | ---: |
| All-40 down-only | 1,975,517,184 bytes / 1.840 GiB | 1,975,517,184 bytes / 1.840 GiB |
| All-40 up+down | 4,804,575,232 bytes / 4.475 GiB | 4,804,575,232 bytes / 4.475 GiB |
| High capacity, keep 0/19/20 | 7,212,105,728 bytes / 6.717 GiB | 6,784,286,720 bytes / 6.318 GiB |

The high-capacity asymmetry follows the manifest-selected pair schedule and
the decision to keep layers 0, 19, and 20 resident.

### All-40 down-only pp2048 trace

Artifact:

`trace-residency-down-only-pp2048-v1.nsys-rep`

The trace contains 11,408,506,880 host-weight H2D bytes in a
1,244.951171 ms all-device union, equivalent to 8.534 GiB/s aggregate.
Time at copy concurrency widths 1, 2, 3, and 4 is 1.454, 793.573, 5.580,
and 444.345 ms. H2D overlaps at least one SM kernel for 72.706% of its
union. Same-pair SM overlap is 34.914% for pair 0 and 36.127% for pair 1.
There is no cross-stream same-GPU SM-kernel overlap.

With bytes apportioned uniformly within each CUPTI DMA record, aggregate
bandwidth at widths 1, 2, 3, and 4 is 3.819, 6.919, 9.349, and 11.424
GiB/s. The width-1 bucket spans only 1.454 ms.

Native copy latency stretches during host ingress:

| Native record | Outside host H2D mean | During host H2D mean | Ratio |
| --- | ---: | ---: | ---: |
| P2P, 4 MiB | 352.756 us | 885.752 us | 2.511x |
| P2P, 256 KiB | 22.177 us | 64.552 us | 2.911x |
| D2D, 256 KiB | 3.224 us | 11.471 us | 3.558x |

For same-device H2D overlap, the corresponding ratios are 3.439x, 3.896x,
and 3.363x. This confirms material copy/HBM/PCIe contention despite weights
being ready before most compute deadlines.

Per-kernel same-device inside/outside mean-duration ratios include 2.837x
for route packing, 2.698x for cuBLAS split-K reduction, and 1.483x for the
canonical local owner sum. Each example has at least five records in both
buckets. These trace partitions are descriptive; differing layer and
work-size mixes prevent treating each ratio as an isolated causal slowdown.

Artifacts:

- `exact-residency-{l16-23,down-only,up-down,high-cap}-c512-v1.out` /
  `.err`;
- `perf-final-resident-ownerbypass-pp2048-v1.out` / `.err`;
- `perf-residency-{down-only,up-down,high-cap}-pp2048-v1.out` / `.err`;
- `perf-residency-{down-only,up-down,high-cap}-pp8128-v1.out` / `.err`;
- `perf-residency-down-only-safe-pp8128-v1.out` / `.err`;
- `perf-current-all40-host-pp8128-v1.out` / `.err`;
- `exact-residency-down-only-{serial,submitserial,nobypass,ownerwait}-pp8128-v1.out` /
  `.err`;
- `exact-residency-down-only-safe-pp8128-v1.out` / `.err`;
- `trace-residency-down-only-pp2048-v1.nsys-rep` / `.sqlite`.

Every policy remains default off. None passes the production throughput
gate, and packed attention is not enabled in these measurements.
