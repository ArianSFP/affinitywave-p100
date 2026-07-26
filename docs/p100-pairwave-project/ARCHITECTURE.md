# Architecture

## Target

The target is Qwen3.6-35B-A3B Q8_0 prefill on four Tesla P100 PCIe 16 GB
GPUs. The qualified toolchain is CUDA 12.8.61, cuBLAS 12.8.3, and `sm_60`.
The model has 40 trunk MoE layers, 256 experts per layer, and top-8 routing.
Stored block 40 is the MTP block and is not part of the 40-layer prefill
trunk.

The optimization problem is unusual:

- Pascal has no tensor cores and no native `dp4a` path suitable for this
  exact Q8_0 workload.
- The four GPUs share a uniform PCIe switch. Bulk P2P is approximately
  12.9 GB/s per pair, while all-to-all application traffic is approximately
  9.7 GB/s/GPU.
- Exact expert arithmetic is split across many small and highly skewed
  expert matrices.
- Kernel overlap does not create free SM throughput on P100. Copy engines
  can overlap useful work, but concurrent SM kernels mostly time-slice.
- The dependency graph and host submission path matter as much as isolated
  kernel throughput.

## Evolution of the execution model

The earlier production path progressed from counting-sort and expert
parallelism to a device-side MoE plan, grouped native-Q8 GEMM, and
per-GPU submit workers. Those changes reduced host synchronization and made
the later wavefront experiments possible. The background is preserved in
[archive/prehistory/](archive/prehistory/).

AffinityWave then explored cross-layer owner service. Its first service-only
checkpoint was not end-to-end correct and was parked. The later frontier
reworked the model around exact logical-owner boundaries, chronological
lanes, bounded arenas, and explicit dependency replays.

The production-qualified runtime is still a diagonal wave. It services four
chronological lanes through a bounded N2048 arena and uses exact CohortRail
kernels. A later default-off PairFold checkpoint completes the separate
two-pair PairWave experiment end-to-end, but remains slower than production.

## PairWave

PairWave divides the four GPUs into fixed physical pairs:

```text
pair 0: GPU0, GPU1
pair 1: GPU2, GPU3
```

Layers 0 through 19 alternate between the pairs. Layer 20 repeats pair 1,
then layers 20 through 39 alternate pair 1 and pair 0. Each pair owns 20
layers: 15 recurrent layers and five attention layers.

A pp8128 prompt is represented as two chronological 4064-token panels.
Each panel contains two 2032-token lanes. The active pair owns all 256
experts for its layer, with two complete canonical 64-expert groups per
active GPU. Each weight exists once.

The retained PairWave service uses 256 descriptors per active GPU:

```text
2 lanes x 2 canonical groups x 64 experts
```

Descriptors and pointers may be duplicated, but weights are not. Each lane
keeps its original route order and M64, M32, and M16 tile boundaries. Ready
logical tiles can share a device launch only when doing so does not change
the descriptor identity or arithmetic order.

The lane-exact service passed all 160 c512 service boundaries and the saved
logits oracle. PairFold now implements the complete 80-task pair-local
recurrent, attention, expert, and hidden-state wavefront. Its accurate warm
pp8128 median is 3027.038 ms, or 2685.133 tok/s. The earlier 2782.101 ms /
2921.533 tok/s value remains a replay and is not a runtime result.

See [PAIRWAVE-AUDIT.md](archive/frontier/PAIRWAVE-AUDIT.md) and the
[lane-exact manifest](evidence/frontier/pairwave-laneexact-v3.manifest).
The runtime design, exactness, trace, warm benchmark, and remaining gaps are
in [PAIRFOLD-CHECKPOINT.md](PAIRFOLD-CHECKPOINT.md).

## PairFold runtime

PairFold turns each layer into two chronological panel tasks. The static
80-task DAG includes panel chronology, per-panel hidden dependencies, one
serialized resource per physical pair, and the repeated pair-1 boundary
between layers 19 and 20. Two event-generation banks and two stable F32
handoff slots per GPU allow the two pairs to overlap without a four-GPU
barrier for ordinary task completion.

Recurrent layers use pair-local exact two-lane GDN. Attention layers retain
chronological K/V state and currently use the literal exact two-lane
control. PairWave expert service joins only the active pair, returns BF16
owner partials into home-local scratch, and performs the original canonical
ordered sum.

The runtime is selected by `GGML_CUDA_AW_PAIRFOLD=1`, accepts only the
qualified c512, pp2048, and pp8128 shapes on four P100 GPUs, and requires
the exact PairWave manifest and P100 arithmetic selectors. It is
default-off and not production-qualified.

## CohortRail

CohortRail is the retained exact-Q8 projection engine. It preserves the two
ordered FP32 accumulator rails used by the accepted exact service. Work is
described by immutable fields:

```text
(layer, lane, expert, tile index, tile class, projection)
```

The natural tile classes are M64, M32, and M16. A P2 operation may join only
the same expert, same layer, same projection/weight pointer, and adjacent
lanes that are both ready. It never concatenates route rows or promotes a
tile class.

This restriction matters. The first pooled PairWave service changed 174
layer-0 F32 values. Every changed token used at least one route whose tile
class changed under pooling. The lane-exact engine fixed that defect.

The final engine retains:

- direct singleton M64/M32/M16 dispatch;
- exact natural-lane P2 where qualified;
- rank-ordered owner reduction;
- W4 canonical owner sum;
- no stack, spills, local loads, or local stores in the 28 retained targets.

## Exact diagonal N2048 path

The production-qualified path is selected with
`GGML_CUDA_AW_P100_EXACT=1`. It uses:

- the existing accurate diagonal graph;
- an N2048 bounded service arena;
- CohortRail projection kernels;
- W4 owner reduction and W4 canonical ordered sum;
- one graph-local F16 activation materialization per qualified fanout;
- an aligned 128-bit F32-to-F16 converter with scalar tails;
- the exact GDN precomputed-decay producer without changing recurrence
  order;
- exact cuBLAS algorithms pinned to complete shape, dtype, layout,
  alignment, stream, architecture, CUDA, and cuBLAS signatures.

The dense selector table is:

| M | N | K | Algorithm |
| ---: | ---: | ---: | ---: |
| 8192 | 2032 | 2048 | 6 |
| 4096 | 2032 | 2048 | 5 |
| 2048 | 2032 | 4096 | 10 |
| 2048 | 2032 | 512 | 6 |
| 512 | 2032 | 2048 | 3 |
| 32 | 2032 | 2048 | 7 |
| 8192 | 512 | 2048 | 6 |
| 512 | 512 | 2048 | 8 |

Any signature mismatch falls back to the existing implementation.

## Exact service data flow

The accepted logical flow is:

1. Preserve all current-diagonal input staging.
2. Build deterministic route descriptors without changing router outputs.
3. Run exact Q8_0 gate and up projections.
4. Evaluate the existing SwiGLU expression.
5. Run the exact down projection.
6. Reduce route ranks in the original order.
7. Round the logical-owner partial at the existing BF16 boundary.
8. Return owner partials to the token home.
9. Sum logical owners in canonical order.
10. Publish the completed cell to the ordinary graph.

The final production campaign compares all intermediate service-boundary
files, not just final logits.

## Memory model

The original four live states used approximately 1,332,211,712 bytes/GPU in
major service arrays. The bounded diagonal candidate reclaims 892,000,048
bytes/GPU, approximately 0.831 GiB/GPU, while preserving byte identity.

Other measured memory boundaries remain archived:

- N512 down-only arena: 131,017,728 bytes/GPU.
- N256 gate/up plus N256 down composition: 123,087,872 bytes/GPU, but with
  an explicit service-time tax.
- Direct owner gather: approximately 254 MiB/GPU emergency capacity option,
  slower than the staged path.
- PairCache replay: modeled 955,040,768 bytes/GPU reclaimed, but its runtime
  allocation model is unresolved and therefore not qualified.

## Components that are not production

- PairFold: implemented and exact at c512, but warm pp8128 is 2685.133
  tok/s and therefore remains default-off experimental.
- StitchRail: tile-ready dispatcher/manifest study, not deployed.
- InterferenceWave/TileFrontier: positive offline replay, not a runtime.
- PairCache: predictive exact-weight cache replay, blocked by unresolved
  runtime boundaries.
- Balanced attention: documented and paused.
- HeadFold R44/N256: exact and banked, slower than the final diagonal path.
