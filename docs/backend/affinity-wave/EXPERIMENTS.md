# AffinityWave experiment log

This file records measured variants, including failures. Results from different
benchmark semantics are kept separate.

## Interpretation rules

- "Exact" below means the synthetic expert-service BF16 owner-partial check,
  unless explicitly stated otherwise.
- Projected pp rates come from an output-withheld service benchmark.
- Ordinary llama.cpp pp numbers return model output and are not directly
  comparable with projected service rates.
- A default-off or removed experiment is not part of checkpoint `9d3983b8`.

## Pre-AffinityWave constraints

The project reached AffinityWave only after these relevant paths were explored:

| Path | Result | Consequence |
|---|---|---|
| Stable expert counting sort | about +22% pp8192 | retained in parent line |
| Expert parallelism | large gain | required baseline architecture |
| Per-device host submission | about +29% on ordinary EP | retained in parent line |
| True four-way expert granularity | about +20% pp8192 | fixed accidental two-GPU expert execution |
| Pinned route buffers | about +1% to +2% | retained in parent line |
| Same-SM two-batch overlap/TBO | wall time flat despite higher busy time | kernel-overlap class closed on P100 |
| All-weight AsyncEP | about -37% pp8192 | expert thinning and all-pull transport made it structural loss |
| NCCL Simple and tuning | no useful prefill gain | collective knob axis closed |
| FP16/HFMA2 expert arithmetic | repeatedly slower or unacceptable | exact Q8_0 retained |
| Sub-Q8 weights | prohibited by project accuracy policy | never evaluate |

These results motivated a different decomposition rather than another tweak to
the ordinary layer-by-layer path.

## Phase 0: offline feasibility

Inputs:

- ordinary pp8192 baseline: 1697 tok/s;
- 16.4927 TFLOP total routed expert arithmetic;
- 10 GB/s pessimistic peer transport;
- aggregate code-domain expert histogram;
- saved nsys kernel costs from the preceding production path.

Simulation sensitivity:

| Complete service rate | Simulated pp8192 | Rate | Verdict |
|---:|---:|---:|---|
| 5.5 TFLOP/s/GPU | 2.1810 s | 3756 tok/s | fail |
| 6.0 TFLOP/s/GPU | 2.1128 s | 3877 tok/s | fail |
| 6.5 TFLOP/s/GPU | 2.0551 s | 3986 tok/s | fail |
| 7.0 TFLOP/s/GPU | 2.0056 s | 4084 tok/s | sensitivity pass |
| 9.3 TFLOP/s/GPU | 1.8466 s | 4436 tok/s | physical-peak sensitivity only |

The optimistic 2.1122 s lower bound failed before packing, transport, NCCL,
route gathering, or imperfect balance. Phase 0 therefore registered NO-GO.
The user explicitly requested a prototype anyway.

## Phase 1: initial native-Q8 service

All rows below used four GPUs concurrently and passed the synthetic owner
partial check.

| Variant | Minimum effective TFLOP/s/GPU | Disposition |
|---|---:|---|
| Generic F32 loads | 4.6251 | superseded |
| Two LDG.128 F32 loads | 4.7626 | improved |
| Stage-timed persistent grid | 4.8144 | retained Phase-1 baseline |
| Wider 32x128 shared-A tile | 4.0421 | reverted, occupancy loss |
| Direct 2-D launch grid | 4.7198 | reverted |

Stage timing for the retained F32 service:

| Stage | Range |
|---|---:|
| Request pack | 0.259-0.265 ms |
| Gate | 6.201-6.222 ms |
| Up | 6.195-6.276 ms |
| SwiGLU | 0.185-0.186 ms |
| Down | 7.030-7.130 ms |
| Owner reduction | 0.521-0.524 ms |

The service failed the registered 5.5 TFLOP/s gate, but showed that packing and
reduction were not the principal bottleneck.

## Phase 1b: T64/K32 kernel search

The T64 layout preserves Q8_0 values and scales while reorganizing memory for
the service kernel.

| Variant | Minimum effective TFLOP/s/GPU | Exact | Verdict |
|---|---:|---|---|
| Native layout control | 5.0254 | yes | control |
| T64, M64 split-K2, F32 requests | 5.2375 | yes | retained winner |
| T64, M64 split-K2, BF16 requests | 5.1282 minimum, 5.2849 maximum | yes | retained wire |
| T64, all M32 tiles | 5.2120 | yes | close but no reason to replace winner |
| T64, M64 128-thread variant | 4.8411 | yes | rejected |
| T64, N128 down projection | 4.2526 | yes | rejected |
| T64, down-cache expansion | 4.9828 | yes | rejected; expansion and memory cost |
| T64, M64 split-K1 | 3.5613 | yes | decisively rejected |
| T64 mixed M64/M32/M16 tail pattern | 4.3690 useful | yes | rejected for synthetic mix |

Split-K1 removed a reduction but added enough shared-memory traffic to lose
about 32% versus split-K2. This path is closed.

## Ordinary graph integration checks

T64 integration into the ordinary graph path built and produced coherent
outputs, but was not the wave benchmark:

| Scope | pp2048 | tg result |
|---|---:|---:|
| Native control | 1780.2 tok/s | 66.58 tok/s |
| T64 all expert projections | 1806.1 tok/s | 63.91 tok/s |
| T64 gate/up scope | 1773.3 tok/s | 55.72 tok/s |
| T64 down scope | 1782.6 tok/s | 59.68 tok/s |

These mixed results are not a production adoption claim. The checkpoint's
decode mode and scope controls remain experimental.

## Phase 2a: banked output-withheld wave benchmark

Retained settings:

- T64/K32;
- M64 split-K2;
- BF16 requests and owner partials;
- equal token quarters;
- GDN chunk columns 2, 4 warps;
- lane stagger 1;
- singleton groups `1111`;
- two untimed pre-capture passes.

| Run | Elapsed | Projected rate |
|---|---:|---:|
| Confirmation 1 | 2874.222 ms | 2827.9 tok/s |
| Confirmation 2 | 2874.839 ms | 2827.3 tok/s |
| Mean | 2874.531 ms | 2827.6 tok/s |

Removed before banking:

- unfinished ring-mirror transport;
- temporary direct CUDA failure diagnostics;
- graph-disable probe;
- live VRAM diagnostics.

Default-off shared service, lane balance, fused gate/up, and down-cache variants
were pinned off in the banked command.

## Post-checkpoint scheduler experiments

These experiments ran in a separate dirty worktree. Unless noted, their source
is not included in the immutable checkpoint.

### Per-group compute channels

| Variant | pp8128 elapsed | Rate | Verdict |
|---|---:|---:|---|
| Checkpoint-like control | 2874.462 ms | 2827.7 tok/s | control |
| Four unconstrained channels | 3129.528 ms | 2597.2 tok/s | rejected |
| Two paired channels | 3163.889 ms | 2569.0 tok/s | rejected |
| Unfair CAS admission lock | 3015.671 ms | 2695.3 tok/s | rejected |
| FIFO ticket admission | 3002.328 ms | 2707.2 tok/s | rejected |

Independent streams removed head-of-line blocking but made persistent Q8 grids
time-slice on Pascal. Admission locks recovered only part of the loss. The code
was removed.

### Static and dynamic wave scheduling

| Variant | pp8128 elapsed | Rate | Verdict |
|---|---:|---:|---|
| Lane offsets `[0,1,2,4]` | 2914.783 ms | 2788.5 tok/s | rejected |
| Lane offsets `[0,1,3,4]` | 2910.398 ms | 2792.7 tok/s | rejected |
| Trace-fitted token boundaries | 3009.239 ms | 2701.0 tok/s | rejected |
| Attention lookahead and catch-up | 3262.119 ms | 2491.6 tok/s | rejected |

Extra wave fragmentation and nonuniform offsets cost more fill/drain time than
they recovered. Equal token quarters remained best.

### State-corridor timing

| Variant | pp8128 mean or sample | Rate | Improvement |
|---|---:|---:|---:|
| Generic asynchronous source copy | 2868.091 ms | 2834.0 tok/s | 0.222% |
| Early recurrent state only | 2867.301 ms | 2834.7 tok/s | small |
| Early attention state only | 2846.505 ms | 2855.4 tok/s | most of corridor gain |
| Early all state, two-run mean | 2836.304 ms | 2865.7 tok/s | 1.327% |
| Attention K/V boundary split, two-run mean | 2826.097 ms | 2876.0 tok/s | 0.406% beyond concurrent control |

The early corridor path demonstrated correct dependency ordering at the
benchmark level, but its low-single-digit gain did not justify folding the
extra scheduling complexity into the immutable checkpoint.

### Host and launch hypotheses

| Variant | Result | Verdict |
|---|---|---|
| Cache validated SM counts and remove duplicate event | 2837.584 ms mean | neutral |
| Route every tail through M64 | 2832.636 ms pp8128, but 1011.501 ms p1024 | rejected; 75% p1024 regression |
| Persistent per-owner compute submission pool | 2825.306 ms one sample | 0.028%, removed |
| `CUDA_SCALE_LAUNCH_QUEUES=4x` | 2825.745 vs 2827.447 tok/s control | -0.0602%, closed |

The old profile showed 395.305 ms of `cudaGetDeviceProperties` API duration and
1336.753 ms across 9280 kernel-launch calls. Direct A/B tests showed these were
primarily driver backpressure, not recoverable host work.

The launch-queue test used fresh processes in ABBA order:

| Order | Queue | Elapsed | Rate |
|---|---|---:|---:|
| A1 | omitted | 2874.676 ms | 2827.4 tok/s |
| B1 | `4x` | 2875.897 ms | 2826.2 tok/s |
| B2 | `4x` | 2876.923 ms | 2825.2 tok/s |
| A2 | omitted | 2874.680 ms | 2827.4 tok/s |

### Incomplete profile

A final nsys capture ended before the service printed a timed result. It must
not be treated as a benchmark. Its SHA-256 was
`1ca0c50161303e3772defedba0d2a6746ae6fa8877f233d1218e0ece580ce52a`.

## Closed paths

Do not retry without new evidence:

- split-K1;
- 32x128 shared-A service tile;
- direct-grid service dispatch;
- down-cache expansion;
- N128 down projection;
- unconstrained multi-stream expert service;
- admission-lock variants;
- nonuniform lane staggering;
- trace-fitted token balance as previously modeled;
- attention lookahead/catch-up fragmentation;
- M64-for-all-tails;
- persistent owner submission threads;
- scaled CUDA launch queues;
- same-SM compute overlap/TBO;
- all-weight AsyncEP;
- HFMA2/FP16 expert substitution;
- sub-Q8 weights;
- NCCL knob resweeps without new transport evidence.

## Surviving research questions

1. Can the exact T64/K32 service reach about 7.0 effective TFLOP/s/GPU without
   adding more packing or shared-memory traffic?
2. What placement follows from token-level route co-occurrence rather than an
   aggregate histogram?
3. Can T64/K32 be harvested into the ordinary production MoE plan with a
   material end-to-end win?
4. Is the chunked GDN kernel mathematically and numerically valid under a full
   perplexity gate?
5. If the compute gate is met, can full state normalization and output
   validation be implemented without consuming the performance margin?
