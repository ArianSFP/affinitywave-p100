# Results and qualification status

## Production checkpoint

| Item | Value |
| --- | --- |
| Source base | `ae2b41d682e0c18f5c7277860dd0566244413fcb` |
| PairWave checkpoint commit | `36bad6bb3ad4c9b31edc4dae9fb6c3716b95704b` |
| Source diff SHA-256 | `82d5d9826e31e7f238b9d33fca0483ac08094121d66df38f86e15cfb864c624f` |
| Qualified CUDA library SHA-256 | `44d0db6d838953a57fe5dc18d9045f1d3274f933c94d7e5494816a0d254403b7` |
| Production switch | `GGML_CUDA_AW_P100_EXACT=1` |
| Q8 engine | CohortRail |
| Hardware | 4x Tesla P100 PCIe 16 GB |
| Toolchain | CUDA 12.8.61, cuBLAS 12.8.3, `sm_60` |

The isolated rebuild made from the same tracked source has a different
library SHA because it used another build directory and build invocation.
Only the library hash above is tied to the complete final campaign.

## Exactness

The final `stitchrail-final-prefill-v2` campaign ran two c512 service-dump
arms and two saved-logits arms:

- 640 OFF and 640 ON service-boundary files;
- identical relative layouts;
- all 640 corresponding files byte-identical;
- OFF and ON saved logits byte-identical;
- saved-logits SHA-256
  `47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`.

An independent comparison also covered 1,920 route and owner-partial
diagnostics. All 2,560 files and 2,015,313,920 bytes per arm matched.

The 28 retained kernel targets have zero stack, local memory, spill loads,
and spill stores. Dormant research kernels in the large AffinityWave
translation unit are not covered by that statement.

## Paired performance

The final campaign randomized arm order with seed
`exact-p100-20260726`.

| Prompt | Pairs | OFF median | ON median | Paired median saved | Paired wall gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 5 | 849.235 ms | 816.850 ms | 32.706 ms | 3.851% |
| 8128 | 3 | 2857.911 ms | 2790.956 ms | 67.309 ms | 2.355% |

All eight pairs favored the retained path. At pp8128:

- ON median: 2912.264 tok/s;
- OFF median: 2844.036 tok/s;
- paired median throughput gain: 2.412%;
- improvement over the frozen 2834.559 tok/s comparator: 2.741%.

The three retained ON wall measurements were 2790.956, 2790.443, and
2791.422 ms.

## Memory

The retained bounded service reclaims 892,000,048 bytes/GPU, approximately
0.831 GiB/GPU, relative to the legacy live-state allocation. This met the
accepted approximate capacity goal.

## Status of headline alternatives

These numbers use different evidence classes. Only the first row is a
production rate.

| Path | Result | Evidence class | Status |
| --- | ---: | --- | --- |
| Exact diagonal N2048 + CohortRail | 2912.264 tok/s | paired pp8128 runtime, complete exactness | production-qualified, default-off |
| PairFold two-pair wavefront | 2685.133 tok/s | five-sample warm pp8128 runtime, c512 byte identity | experimental, default-off |
| PairWave N512, 25% copy | 2921.533 tok/s | calibrated replay | not a runtime rate |
| PairWave zero-copy/tax bound | 2969.340 tok/s | optimistic replay bound | not attainable evidence |
| PairCache static R16 | 3108.549 tok/s | untouched-test replay | blocked, not implemented |
| InterferenceWave TileFrontier | 3545.585 tok/s | offline replay | proof-of-concept gate only |
| HeadFold R44/N256 HalfPipe | about 2299 tok/s | measured pp8128 runtime | exact and banked, slower |
| Early AffinityWave checkpoint | about 2828 tok/s | service-only, no logits | invalid as end-to-end comparison |

## PairWave exactness result

The PairWave one-copy loader and lane-exact service primitive are real:

- balanced 20-layer ownership per pair;
- exactly 60 T64 tensors per GPU;
- all four layer-0 inputs, route IDs, route weights, and reduced outputs
  byte-identical;
- all 160 c512 service boundaries byte-identical;
- complete c512 saved logits match the accepted SHA-256.

The published 2912.264 tok/s rate is still not a PairWave-only rate.
PairWave's full pair-local recurrent/attention scheduler, called PairFold in
the replays, was implemented later as an exact default-off runtime. Its
accurate warm pp8128 median is 3027.038 ms / 2685.133 tok/s, versus
2788.594 ms / 2914.731 tok/s for its same-build diagonal control. The
PairFold runtime is therefore measured but not production-qualified.

The original fresh-process 4050.309 ms result was cold and must not be used
as steady-state throughput. Full PairFold architecture, exactness, trace,
memory, prior experiments, and raw warm samples are documented in
[PAIRFOLD-CHECKPOINT.md](PAIRFOLD-CHECKPOINT.md).

## PairCache decision

The sealed 32-window PairCache replay used 16 train, eight validation, and
eight untouched test windows. Validation selected a static R16 exact-weight
policy:

| Split | Wall | Rate | Gain over replay R0 | Minimum window gain |
| --- | ---: | ---: | ---: | ---: |
| validation | 2596.673 ms | 3130.160 tok/s | 185.428 ms | 171.904 ms |
| untouched test | 2614.725 ms | 3108.549 tok/s | 167.376 ms | 86.223 ms |

All windows were positive and descriptor/scheduler invariants passed.
Runtime work was nevertheless blocked at the time of that replay because:

- PairFold pair-local pre was unmeasured;
- recurrent/attention transport was not endpoint-modeled;
- the phase-reused arena and PairFold live metadata were not implemented;
- compact BF16 combine and generation-safe shadow publication were absent;
- the modeled wall had only 22.608 ms margin.

The authoritative status is
`replay-blocked-unresolved-model-boundaries`. PairCache is a promising
hypothesis, not a production rate. The later PairFold implementation does
not retroactively qualify or implement PairCache.

## Historical performance context

The pre-AffinityWave work raised pp8192-class performance through counting
sort, expert parallelism, per-GPU submit threads, the EP4 granularity fix,
pinned route IDs, grouped GEMM, and the A6 device plan. Prompts, binaries,
and harnesses changed during that progression, so it is architectural
context rather than a valid paired speedup table.

The authoritative historical documents are:

- [Qwen MoE prefill execution](archive/prehistory/qwen36-35b-moe-pp-execution.md)
- [round-3 results](archive/prehistory/round3-phase0-RESULTS.md)
- [parked first AffinityWave](archive/prehistory/p100-affinitywave-parked-20260723.md)
- [final exact result](archive/stitchrail/FINAL-RESULTS.md)
