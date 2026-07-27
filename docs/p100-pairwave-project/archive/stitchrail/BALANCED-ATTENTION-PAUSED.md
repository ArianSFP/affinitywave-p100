# Exact P100 balanced-attention epoch

Date: 2026-07-26

## Status

This work is paused and default-off.

No balanced-attention scheduler, CUDA callback, scratch allocation, selector,
or runtime switch has been added to the StitchRail source. The production
path remains the four-lane N2048 diagonal AffinityWave scheduler with
CohortRail. The general `GGML_CUDA_AW_P100_EXACT` umbrella must not activate
this mechanism unless it later passes the complete qualification sequence in
this document.

Resuming source work requires explicit confirmation because this is a large
cross-GPU scheduling change. It is not a local FlashAttention kernel edit.
This note records the mechanism and the smallest safe resume path without
claiming an integrated system gain.

## Production context

The target is the current pp8128 topology:

- four 2032-token chronological lanes;
- 40 layers with diagonal mapping
  `layer = diagonal - lane_stagger*lane`;
- default `lane_stagger=1`;
- full-attention layers 3, 7, 11, 15, 19, 23, 27, 31, 35, and 39;
- four immutable 64-expert owner GPUs;
- exact Q8_0 CohortRail expert arithmetic;
- BF16 owner partials and canonical ordered FP32 owner sum;
- exact dense selectors;
- no PairWave loader in the production diagonal run.

For a full-attention layer `L`, lane 0 reaches the layer on diagonal `L`,
lane 1 on `L+1`, lane 2 on `L+2`, and lane 3 on `L+3`. The proven balanced
layout needs all four same-layer activation shards. Therefore it cannot be
inserted by changing only one kernel launch or one tensor extent.

## Exact mechanism

### Original and balanced token mapping

The original token ownership is:

| Original lane | Global range | First chunk | Second chunk |
| ---: | --- | --- | --- |
| 0 | `[0, 2032)` | c0: base 0, 1024 tokens | c1: base 1024, 1008 tokens |
| 1 | `[2032, 4064)` | c2: base 2032, 1040 tokens | c3: base 3072, 992 tokens |
| 2 | `[4064, 6096)` | c4: base 4064, 1056 tokens | c5: base 5120, 976 tokens |
| 3 | `[6096, 8128)` | c6: base 6096, 1072 tokens | c7: base 7168, 960 tokens |

The balanced destination mapping is:

| GPU | Packed chunks | Packed offsets | Tokens |
| ---: | --- | --- | ---: |
| 0 | c0 + c7 | 0, 1024 | 1984 |
| 1 | c1 + c6 | 0, 1008 | 2080 |
| 2 | c2 + c5 | 0, 1040 | 2016 |
| 3 | c3 + c4 | 0, 992 | 2048 |

The corresponding fixed arrays are:

```text
chunk_tokens  = {1024, 1008, 1040, 992, 1056, 976, 1072, 960}
chunk_base    = {0, 1024, 2032, 3072, 4064, 5120, 6096, 7168}
source_offset = {0, 1024, 0, 1040, 0, 1056, 0, 1072}
target_device = {0, 1, 2, 3, 3, 2, 1, 0}
target_offset = {0, 0, 0, 0, 992, 1040, 1008, 1024}
```

The first endpoint in each original lane is aligned to a 256-token
FlashAttention mask-scan boundary. This alignment and a fixed count of three
K partitions are part of the exactness contract.

### Graph boundary

The safest redistribution boundary is:

```text
original lane attn_norm
    -> forward packed shuffle
    -> shared F16 cast, Q+gate, K, V
    -> Q/K normalization and RoPE
    -> chronological K/V publication
    -> exact three-partition FlashAttention
    -> sigmoid gate and attention output GEMM
    -> inverse shuffle of attn_output
    -> original lane attn_residual
```

The forward source is the F32 `attn_norm` output. Positions and the applicable
query-mask rows must use the identical chunk map so every query retains its
original absolute position and causal boundary.

Each GPU publishes its two packed K and V chunks into every destination's
canonical cache roots at the original `chunk_base`. FlashAttention must see
K and V in global chronological order, never in GPU-major or packed-query
order.

The inverse destination is the original F32 `attn_output` tensor, immediately
before `attn_residual`. This leaves the original residual input on its home
lane and guarantees that `attn_post_norm`, router arithmetic, CohortRail,
shared expert work, owner reduction, and the post-layer corridor continue to
see the production lane layout.

The balanced output projection at M2048,K4096 still needs a complete exact
selector sweep at N 1984, 2016, 2048, and 2080. The existing proof covers the
Q+gate and K/V projections, not this output GEMM. An implementation must not
advance past the arithmetic oracle until the output projection matches the
four original N2032 calls bit-for-bit.

## Source map

No source file listed here has been changed for this paused work.

### Scheduler and graph partition

`ggml/src/ggml-backend-meta.cpp`

- `ggml_backend_meta_aw_partition` identifies each cell. Its current
  `flash_attn` and `attention_state_end` markers are near lines 2866-2873.
  A resumed implementation should add stable markers for `attn_norm`,
  `attn_output`, and `attn_residual`.
- `submit_diagonal` near lines 3651-3748 submits the current
  `diagonal - lane_stagger*lane` cells.
- The existing HeadFold attention branch near lines 4480-4653 is a useful
  oracle scaffold. It already executes four same-layer attention-state
  graphs, copies K/V segments to chronological offsets, and resumes
  attention compute. The full HeadFold scheduler is not a production
  integration candidate.
- The production diagonal loop starts near line 4842. Its current attention
  corridor copies are near lines 4953-4965, pre-graph construction near
  lines 4970-4996, early K/V publication near lines 5049-5062, and the
  optional post-state resume near lines 5065-5087.
- The CUDA callback typedef and proc-address loading area is near lines
  2372-2587. A balanced epoch should be exposed as one narrow callback rather
  than placing CUDA stream and event operations directly in the meta
  scheduler.

### CUDA runtime and dense GEMMs

`ggml/src/ggml-cuda/ggml-cuda.cu`

- The exact dense selector table is near lines 2012-2105. Proven balanced
  signatures are Q+gate M8192,K2048 algorithm 5 and K/V M512,K2048
  algorithm 3 for N 1984, 2016, 2048, and 2080.
- Any selector must continue to require CUDA 12.8, cuBLAS 12.8.3, sm_60,
  exact dtypes, dimensions, strides, alignment, output layout, and the
  qualified stream. Any mismatch uses the existing fallback.
- The current corridor copy channel and callbacks are near lines 4259-4314.
  They demonstrate the required main-stream event, nonblocking copy stream,
  and destination wait pattern. The epoch needs separate multi-copy state;
  it must not reuse the corridor channel's single completion event for
  unrelated transfers.
- A new callback would also need a proc-address export near the existing
  AffinityWave exports around line 7281.

### FlashAttention partitioning

`ggml/src/ggml-cuda/fattn-common.cuh`

- The P100 parallel K-block selection is near lines 1111-1194.
- `GGML_CUDA_AW_FA_PARALLEL_BLOCKS` is currently a test-only global override
  near lines 1177-1185.
- The ordered partition combine is near lines 1274-1281.

A resumed production implementation must select three partitions only for
the complete balanced P100 D256, query4/GQA8 signature. It must not set the
global environment override for unrelated FlashAttention calls.

`ggml/src/ggml-cuda/fattn-tile.cuh` contains the retained query4 tile
implementation and its fixed tile geometry. The balanced mechanism does not
authorize query8, a different softmax tree, or a different K-tile order.

### Model graph boundaries

`src/models/qwen35moe.cpp`

- The layer graph and residual boundary are near lines 218-265.
- `build_layer_attn` is near lines 318-407. It emits `attn_norm`, the shared
  F16 projection input, Q+gate/K/V, RoPE, `attn_pregate`, `attn_gated`, and
  `attn_output`.

`src/llama-graph.cpp`

- `build_attn_mha` creates `GGML_OP_FLASH_ATTN_EXT`, applies F32 precision,
  and reshapes its output near lines 2380-2430.

### Expert arithmetic

`ggml/src/ggml-cuda/affinity-wave.cu`

- CohortRail and the live diagonal expert service remain downstream of the
  inverse shuffle.
- This mechanism requires no M64, M32, M16, planner, route, owner reduction,
  or canonical sum edit.
- PairWave-specific work remains separate and default-off.

## Required event and stream protocol

The first implementation should use one serialized scratch bank. It should
only add double buffering after the single-bank path is exact and a trace
shows a real overlap opportunity.

1. Each lane main stream records `norm_ready[lane]` after its F32
   `attn_norm` result and position/mask inputs are complete.
2. Each destination copy stream waits for the required source readiness
   events and copies its two chunks into packed scratch. Local copies use
   device-to-device transfer; remote copies use peer transfer.
3. A destination records `packed_ready[gpu]` only after both chunks,
   positions, and mask rows are present. Its main compute stream waits on
   this event.
4. The main stream runs the exact balanced dense projections, normalization,
   RoPE, and cache writes, then records `kv_ready[gpu]`.
5. Each destination K/V gather stream waits on all four `kv_ready` events.
   It copies all eight K segments and all eight V segments into that
   destination's canonical cache roots at `chunk_base`.
6. Each destination records `global_kv_ready[gpu]` after every chronological
   segment is present. The main stream waits before FlashAttention.
7. The main stream runs the exact three-partition FlashAttention, gate, and
   output projection, then records `output_ready[gpu]`.
8. Each original lane's inverse stream waits on the two source output events
   and restores its two F32 `attn_output` chunks at `source_offset`.
9. The lane records `restored_ready[lane]`. Its main stream waits before
   submitting `attn_residual` or any later node.
10. Scratch reuse waits for both the final inverse readers and all K/V peer
    readers. A later epoch must never overwrite packed Q, K, V, mask, or
    output data still referenced by an earlier stream.

Production code must not use `cudaDeviceSynchronize` or host polling between
these stages. All cross-GPU ordering must be represented by reusable events
and stream waits. Event ownership and destruction must remain on the device
that created the event.

## Existing proof and measured ceiling

The authoritative proof artifacts are in:

```text
/home/arian/llama.cpp-q36-moe/.worktrees/affinitywave-frontier-20260724/
results/qwen36-35b-moe-pp-20260721/affinitywave-frontier-20260724/
```

### FlashAttention exactness

The exact aligned, forced-three-partition comparisons are:

- `scanfold-fa-exactness-force3-1024-base0-v1.out`
- `scanfold-fa-exactness-force3-1040-base2032-v1.out`
- `scanfold-fa-exactness-force3-1056-base4064-v1.out`
- `scanfold-fa-exactness-force3-1072-base6096-v1.out`
- `scanfold-fa-aligned-nh2-force3-v1.out`

Across the four retained comparisons, 16,646,144 FP32 outputs were checked
with zero mismatches.

The measured two-KV-head chunk times were:

| Chunk | Queries | Chronological prefix | Time ms |
| ---: | ---: | ---: | ---: |
| c0 | 1024 | 1024 | 3.427 |
| c1 | 1008 | 2032 | 6.759 |
| c2 | 1040 | 3072 | 9.511 |
| c3 | 992 | 4064 | 12.842 |
| c4 | 1056 | 5120 | 16.237 |
| c5 | 976 | 6096 | 19.279 |
| c6 | 1072 | 7168 | 22.881 |
| c7 | 960 | 8128 | 23.583 |

The retained pair assignment measured:

| GPU | Pair | Time ms |
| ---: | --- | ---: |
| 0 | c0 + c7 | 27.010 |
| 1 | c1 + c6 | 29.640 |
| 2 | c2 + c5 | 28.790 |
| 3 | c3 + c4 | 29.079 |

The critical pair is 29.640 ms.

### Transport proof

The relevant artifacts are:

- `scanfold-attention-p2p-chronological-v2.out`
- `scanfold-attention-p2p-exact-unequal-v3.out`
- `scanfold-attention-p2p-probe.cu`

The retained variable-size P2P replay measured:

- forward F32 hidden shuffle: 1.673831 ms;
- chronological K/V all-gather: 1.672385 ms;
- inverse F32 hidden shuffle: 1.677956 ms;
- total communication: 5.024173 ms;
- scratch: 33,685,504 bytes/GPU;
- validation errors: zero.

The first transport probe used GPU-major K/V offsets and is invalid. Only
the chronological v2 and exact unequal v3 results are retained.

### Dense projection proof

The relevant artifacts are:

- `headfold-attention-q-exactness-probe.cu`
- `headfold-attention-q-exactness-v1.out`
- `headfold-attention-kv-exactness-v1.out`
- `HEADFOLD-AUDIT.md`

Q+gate M8192,K2048 algorithm 5 checked 66,584,576 FP32 values with zero
mismatches. Its worst balanced call measured 8.519 ms versus 9.705 ms for
selector 99.

K/V M512,K2048 algorithm 3 checked 4,161,536 values with zero mismatches and
measured 0.642 ms at the worst balanced batch.

These device oracles compared the balanced calls with four original
N2032 calls. They do not by themselves qualify the real multi-stream
schedule, output projection, valid-model logits, or the diagonal scheduler.

### Isolated ceiling

The retained isolated epoch is:

```text
29.640 ms critical attention pair
+ 5.024173 ms measured communication
= 34.664323 ms
```

The prior diagonal trace model attributed about 44.013 ms/layer to the
current full-attention critical slice. The isolated ceiling is therefore
9.348677 ms/layer, or 93.48677 ms across ten full-attention layers.

This is not a system prediction. Scheduler stalls, graph-shape work, P2P
contention, cache publication, output-GEMM selection, and displaced work can
consume all of the isolated saving.

### Rejected layouts

These controls must not be reopened without a new exact mechanism:

- equal 1016-token chunks: 5,376 of 4,161,536 FP32 outputs differed;
- a lower-imbalance split with a first chunk below 1024: 2,977 to 3,915
  mismatches depending on the lane;
- global seven-by-1024 plus 960-tail layout:
  `scanfold-fa-global-exactness-force3-v1.out` reports 6,098 mismatches.

The failures come from changed final K-tile or mask-scan geometry and changed
FP32 combine order. Small numerical errors are still exactness failures.

## Main risks

### Diagonal rendezvous

The two chunks assigned to a GPU become available on different diagonals.
GPU 0 and GPU 1 each wait for one lane-0 chunk and one lane-3 chunk. GPU 2
and GPU 3 wait for lane-1 and lane-2 chunks. A naive same-layer barrier
drains and refills the diagonal wave around every full-attention layer.

The current scheduler derives work from one formula and has no persistent
per-lane deferred-cell state. A correct integration needs explicit
`next_layer`, ready, deferred, restored, and completed state. Otherwise it
can duplicate a cell, skip a cell, or publish a corridor before its output
exists.

The prior FoldWave replay retained the balanced epoch as a useful primitive
but did not prove it as a diagonal integration. This remains the largest
performance risk.

### Auxiliary graph shapes and storage

The packed calls have N 1984, 2080, 2016, and 2048, while every production
lane graph is built for N2032. GPU 1 needs 48 more rows than its current
graph allocation. Mutating live tensor dimensions or pointers in place is
unsafe because intermediate buffer sizes and view strides were reserved for
the original graph.

A resumed implementation must allocate explicit graph-local auxiliary
descriptors and scratch for the balanced epoch. It must not globally enable
the old lane-balance split or change ordinary recurrent, expert, decode, or
append graphs.

### Positions, masks, and cache order

Packed query row numbers are not global token positions. Both position data
and query-mask rows must follow the chunk mapping. K and V must be gathered
to global chronological bases before any FA launch.

All four cache roots must contain the same canonical prefix expected by the
existing attention corridor after the epoch. Cache validation must include
every chronological prefix, not only the final attention output.

### Arithmetic selection

cuBLAS row independence does not guarantee byte identity when N changes.
The Q+gate and K/V algorithms are proven only for the recorded CUDA,
cuBLAS, sm_60, shape, stride, alignment, and stream signatures. The balanced
output projection is still unqualified.

The global three-partition diagnostic environment variable is too broad for
production. A signature mismatch must use the current implementation.

### Stream and P2P contention

Eight forward chunks, 24 remote K/V segments plus local segments, and eight
inverse chunks can contend with CohortRail input, owner partial publication,
weight prefetch, and the normal chronological corridor. A transport-only
microbenchmark does not establish live overlap.

The epoch must use separate events and copy channels. Reusing the current
single corridor completion event can introduce event overwrite or premature
scratch reuse.

## Staged resume plan

### Stage 0: large-change confirmation

Present this source and scheduling boundary before editing. Confirm that the
experiment remains default-off, does not replace the production diagonal,
and may be closed if rendezvous cost removes the isolated gain.

### Stage 1: metadata-only boundary audit

- Add stable cell indices for `attn_norm`, `attn_output`, and
  `attn_residual`.
- Dump the four full-layer boundaries and root layouts under a diagnostic
  selector.
- Run the current path unchanged and confirm identical graph order and saved
  outputs.

Rollback: remove the diagnostic markers. No runtime path changes.

### Stage 2: transport service

- Add an sm_60-only balanced-attention transport callback with a serialized
  scratch bank and separate event channels.
- Implement forward shuffle, chronological K/V gather, and inverse shuffle
  without arithmetic.
- Check every byte for all eight chunks, all four owner mappings, and repeated
  scratch reuse.

Rollback: disable or remove the callback. The normal corridor remains
unchanged.

### Stage 3: same-layer arithmetic oracle

- Reuse the existing HeadFold same-layer attention branch only as an oracle
  harness.
- Run balanced Q+gate algorithm 5 and K/V algorithm 3 under complete
  signature checks.
- Sweep and qualify the balanced M2048,K4096 output projection.
- Force exactly three FA partitions only for the balanced D256 query4/GQA8
  signature.
- Compare every intermediate projection, K/V cache root, FA output, gate,
  `attn_output`, and `attn_residual` with the four original N2032 calls.

Rollback: run the existing equal-lane oracle path. Do not promote selectors.

### Stage 4: deferred diagonal scheduler

- Keep the current production `for diagonal` loop as the complete fallback.
- Under a narrow test-only switch, stop a full-attention cell after
  `attn_norm`, register its two chunks, and continue only graph cells whose
  dependencies remain legal.
- Submit the balanced epoch only when all eight same-layer chunks are ready.
- Restore all four `attn_output` tensors, then resume each lane at
  `attn_residual`.
- Track every cell with explicit ready/deferred/completed state and assert
  exactly one execution of all 160 layer-lane cells.
- Preserve current stream ownership, expert service order, owner reduction,
  canonical sum, post work, and corridor publication.

Rollback: turn off the narrow switch and execute the original diagonal loop.

### Stage 5: qualification and retention

Advance only through:

1. resource and SASS audit;
2. transport byte oracle;
3. complete projection and FA bitwise oracle;
4. one full-layer cache and `attn_output` boundary;
5. all ten full-attention layers;
6. all 160 c512 service boundaries;
7. saved-logits SHA-256
   `47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`;
8. matched pp8128 trace;
9. five pp2048 and three pp8128 paired system runs.

Reject immediately on:

- any arithmetic, cache, route, service-boundary, or logits mismatch;
- any FA launch that does not use the retained three-partition order;
- any duplicated, skipped, or prematurely published cell;
- any regression in CohortRail, owner communication, GDN, or idle time that
  removes the attention saving;
- a non-positive paired wall median or a bootstrap 95% lower bound at or
  below zero.

## Rollback contract

A future implementation should use a narrow default-off selector such as:

```text
GGML_CUDA_AW_P100_EXACT_BALANCED_ATTN=1
```

The name is a design placeholder and does not exist in source.

The switch must guard all of the following as one unit:

- auxiliary balanced graph and scratch allocation;
- new event and copy channels;
- balanced dense selector signatures;
- signature-scoped FA partition selection;
- deferred diagonal-cell scheduling;
- forward and inverse token shuffles.

With the switch absent or zero:

- tensor allocation and graph shapes remain the current N2032 lane shapes;
- the current attention K/V corridor is used;
- the current dense selector table and FA partition heuristic are used;
- the original diagonal loop executes without the deferred-cell state;
- CohortRail and PairWave behavior are unchanged.

No public API, GGUF format, model type, owner mapping, or output layout may
depend on the experimental path. Disabling the switch must be a complete
rollback without rebuilding model data or converting weights.

## Final paused decision

The arithmetic and transport primitive is credible and bitwise proven at its
isolated boundaries. Its estimated ceiling is about 93.49 ms over ten
full-attention layers. The production diagonal composition is not proven,
and the required rendezvous can erase that ceiling.

For that reason the family is paused, unimplemented, and default-off. Resume
only with explicit confirmation, begin with the metadata and transport
stages, and retain it only after a positive exact end-to-end diagonal result.
