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

The implemented attention path is the literal exact two-lane control. The
planned pp8128 `c0+c3 / c1+c2` and `c4+c7 / c5+c6` packed redistribution
with a qualified three-partition FlashAttention signature is not present.
Standalone retained-descriptor recurrent, attention, and composed endpoint
probe executables were not added; qualification used the end-to-end runtime,
service dumps, scheduler test, and trace instead. Consequently, this result
does not claim that the replay's optimized attention architecture was
materialized, and pp2048/pp8128 do not have retained intermediate
byte-comparison dumps comparable to the full c512 oracle.

The checkpoint commit includes source, tests, harnesses, this result record,
and selected small text evidence. Generated logits, binary service dumps,
and Nsight databases remain local. No push or PR is part of the checkpoint
procedure.
