# AffinityWave frontier notebook

Date: 2026-07-24

## Scope

- Immutable reference: `ae2b41d682e0c18f5c7277860dd0566244413fcb`
- Detached worktree: `affinitywave-frontier-20260724`
- Production worktree and captures remain read-only.
- Accuracy gate: byte-identical or paired perplexity delta within +/-0.003.
- Q8_0 weights only. No sub-Q8 experiments.
- Target: at least 4000 tok/s pp8128, supported by probes and traces.

## Initial state

- Four Tesla P100-PCIE-16GB devices are idle apart from small desktop contexts.
- Root filesystem has 19 GB free.
- Production reference currently uses the exact T64/K32 AffinityWave kernels,
  diagonal token-lane scheduling, BF16 owner partials, F32 wire values, and
  output/gated-delta-net precapture.
- Existing accurate trace attributes the final GPU's critical path primarily to
  exact Q8 projection kernels, FP16 target SGEMMs, FlashAttention, copies, and
  gated-delta-net work. Logical identities of the FP16 SGEMMs are not yet
  captured, so the first profiler pass will add that mapping before redesign.

## Decision discipline

Every candidate must have:

1. A measured or analytically bounded target on the current critical path.
2. An isolated kernel/service benchmark when possible.
3. An end-to-end prefill measurement with the production environment.
4. Correctness and KLD/perplexity validation before retention.

Rejected variants and their evidence will stay in this notebook.

## Reference reproduction

The detached `ae2b41d68` build used CUDA 12.8, GCC 14 as nvcc's host
compiler, `sm_60`, NCCL, and FlashAttention. The first `--no-warmup` run is
not a throughput sample: it intentionally exposed that llama-bench charges
AffinityWave's two pre-capture passes to the timed evaluation in that mode.
Its internal steady pass was 2522.0 tok/s while the polluted JSON result was
737.488 tok/s. Formal controls therefore use the same warmup behavior as the
production measurement.

Valid pp8128/ub8128 samples:

- 2517.739 tok/s
- 2517.110 tok/s
- Mean: 2517.425 tok/s
- Delta from the 2517.169 production reference: +0.010%

The reference reproduction gate passed.

The allowed-precision single-GPU backend subset passed 201/201 cases for
`ADD`, `GATED_DELTA_NET`, `SSM_CONV`, and `RMS_NORM`. No sub-Q8 operation was
run. The four-GPU edge-mix service checked 16,777,216 BF16 partial values on
each GPU with zero mismatches. Its slowest-device throughput was 4.225 TF/s
with 236 M64, 12 M32, and 232 M16 tiles, making ragged tails a measured target
rather than a synthetic assumption.

## Mapped production trace

Default-off NVTX and GEMM-map instrumentation was ported without changing the
normal execution path. A pp8128 mapping capture associated all 600 target
SGEMMs with their graph node, tensor, layer, shape, and surrounding conversion
work. Coverage was 100% by duration with no ambiguous collisions.

The critical GPU had a 2811.348 ms busy span in the profiled window. Its
measured families were:

- Exact-Q8 expert projections: 902.971 ms
- Mapped ordinary SGEMM: 715.511 ms
- FlashAttention: 439.352 ms
- Copies: 285.532 ms
- Other graph work: 241.971 ms
- Gated delta net: 145.271 ms
- Owner reduction: 53.861 ms
- Conversion: 48.570 ms

Across all four GPUs, the mapped SGEMM families were recurrent QKV
1093.965 ms, recurrent attention 560.812 ms, recurrent output 557.946 ms,
full-attention attention 364.559 ms, full-attention output 186.152 ms, and
98.869 ms of smaller families. Recurrent QKV alone is about 273.5 ms per GPU,
so it meets the plan's threshold for investigating a direct QKV/conv
specialization after the Q8 engine.

The trace establishes two independent large targets. Reaching 3164 tok/s
requires about 660 ms from the unprofiled reference wall time; exact-Q8 and
mapped SGEMM together contain 1618.482 ms on the critical GPU.

## WarpWave full-K result

The first `warpwave` kernel compiled to 80 registers/thread and 16,384 bytes of
shared memory with no local-memory allocation. Three 256-thread CTAs fit per
P100 SM. Each warp owned one 16-row by 64-column tile and traversed K in strict
sequential order.

The four-GPU edge-mix check was bit-exact over 16,777,216 BF16 values per GPU,
but performance was only 2.522 TF/s on the slowest device. Gate/up took about
13.9 ms each and down about 12.1 ms, versus the reference 7.44/7.44/7.91 ms.
The route distribution required 1196 WarpWave tiles versus 480 aggregate
legacy tiles. Thus the 16-row work decomposition increased Q8 weight-tile
traffic about 2.5x while issued arithmetic stayed nearly unchanged. The extra
occupancy cannot recover that loss.

The full-K WarpWave architecture is closed on performance, not accuracy. A
split-K2 version cannot address duplicated weight traffic and is therefore not
warranted by the specified accuracy-only split-K gate. The single compact CTA
fallback will retain the legacy M64/M32/M16 reuse while reducing shared memory
and register pressure.

## Compact-Q8 fallback result

The compact single-buffer CTA kernels retained the legacy route decomposition
and compiled at 80 registers/thread. Their shared-memory footprints were
16,384 bytes for M64, 12,288 for M32, and 10,240 for M16, allowing 3, 5, and
6 resident CTAs per SM respectively. The edge-mix result was again bit-exact.

Slowest-device throughput was 4.271 TF/s. Gate/up/down were
7.553/7.554/7.934 ms, essentially the reference 7.44/7.44/7.91 ms. The added
occupancy only offset the loss of double-buffer overlap. This is about a 1%
gain and predicts nowhere near 300 ms of pp8128 removal, so the compact-Q8
fallback and the first Q8 architecture family are closed. No tile sweep is
warranted.

## DenseWave probes

The mapped production shapes show why the original 716 -> 416 ms target is
not possible with FP32 accumulation alone. The timed dense work is about
5.285 TFLOP per GPU, so the P100's 9.3-TFLOP FP32 peak gives a 568 ms
arithmetic floor before dequantization, conversion, and output traffic.
Production cuBLAS is already around 7.3-7.5 TF/s.

A standalone controlled probe used the exact 8192x2032x2048 QKV shape and
native T64 Q8 layout. The custom segmented-HFMA2 kernel evolved through:

- 16-row-token warp tile: 6.32 TF/s
- Coalesced output-channel lanes: 6.52 TF/s
- 2x8 register microtile: 6.72 TF/s
- Four-warp, 2x16 register microtile: 8.38 TF/s

The final custom kernel uses 96 registers, 12,672 bytes shared memory, and no
spills. Its K32 promotion error was relative-L2 0.002156 on the deterministic
full output. The 8.38-TF/s result is only about 12% over stable cuBLAS and
cannot remove 300 ms end to end.

The more aggressive replacement used cuBLAS FP16 accumulation for K-sliced
partials followed by deterministic FP32 promotion and summation:

| K chunk | Effective TF/s | Relative L2 |
|---:|---:|---:|
| 128 | 6.37 | 0.004849 |
| 256 | 9.17 | 0.006553 |
| 512 | 11.61 | 0.006901 |
| 1024 | 12.53 | 0.007714 |
| 2048 | 14.51 | 0.008623 |

Only K1024 and whole-K were large enough to approach the architecture gate.
Both were integrated or exercised behind default-off controls and tested:

- K1024: 2764.224 tok/s pp8128 (+9.8%). Active c512 KLD had PPL 4.067437
  versus the legacy AffinityWave 4.078325, same-top 99.216%, and mean KLD
  0.001121 against the normal reference. The incremental PPL movement
  (-0.010888) fails the +/-0.003 gate, and same-top is below 99.5%.
- Whole-K FP16: 2810.708 tok/s pp8128 (+11.7%). PPL was 4.075744,
  same-top 97.647%, and mean KLD 0.003295. It fails the user's >=99.5%
  coding-workload accuracy requirement and still misses 3164 tok/s.

Fine K chunks improve arithmetic accuracy but erase the speed advantage, in
agreement with the earlier split-K HGEMM and HFMA2 closures in project memory.
DenseWave is therefore closed: no measured speed/accuracy point meets the
large-gain and accuracy gates.

## Attention-superstage model

The mapped trace's perfect-balance device-work floor is 2639.797 ms while the
critical GPU busy union is 2811.348 ms. Even a zero-cost redesign that removes
all device imbalance can save only 171.551 ms, about 5.3% of the unprofiled
production wall time. A context-parallel attention rendezvous must additionally
pay query transfer, partial max/sum/value transfer, rank-order combination,
and lost diagonal-wave overlap. It cannot reach the required 10% modeled gain,
so the attention superstage is rejected without implementation.

## Continued frontier round: target revised to 4000 tok/s

The user revised the formal target to at least 4000 tok/s, or at most 2.032 s
for pp8128. Relative to the 2517.425 tok/s reproduction (3.229 s), this
requires about 1.197 s wall-time removal. The earlier campaign-close section
was therefore superseded; all subsequent work remains in this detached
worktree and is recorded below.

## TailWave live rejection

TailWave split the legacy M16 residual tile into M8/M4 kernels. The isolated
edge-route service suggested a useful tail win, but a production-graph trace
showed that the synthetic histogram overrepresented sparse tails.

- Legacy critical GPU: M64 615.088 ms, M16 176.020 ms, M32 114.541 ms,
  total 905.649 ms.
- TailWave critical GPU: M64 615.920 ms, M8 121.646 ms, M4 82.688 ms,
  M32 114.219 ms, total 934.473 ms.
- End-to-end pp8128: 2486 tok/s, a regression.

Evidence:

- `legacy-live-pp8128.nsys-rep` / `.sqlite`
- `tailwave-live-pp8128.nsys-rep` / `.sqlite`
- `tailwave-live-pp8128.analysis.md`

The real M16 tiles are generally close to full, so padding removal did not
repay the extra weight traffic and kernel sequencing. TailWave is closed.

## BroadWave exact-Q8 result

BroadWave changed the dominant M64 kernel from N64 to N128 while retaining the
legacy split-K2 FMA order. One 256-thread CTA shares each staged A tile across
two adjacent 64-column weight tiles.

- Resources: 122 registers/thread, 32,768 B shared, no local allocation,
  two CTAs/SM.
- Edge-route service: zero mismatches over 16,777,216 BF16 outputs per GPU;
  approximately 20.98 ms versus 23.74 ms legacy.
- End-to-end pp8128: 2572.7869 tok/s, about +2.2% over the immutable
  reproduction.
- Live critical-GPU exact-Q8 slice: 824.097 ms versus 902.971 ms reference.
- Live M64: 535.987 ms, M16: 175.944 ms, M32: 114.552 ms.

Evidence:

- `broadwave-v1-pp8128-bench-p8128-ub8128.out`
- `broadwave-live-pp8128.nsys-rep` / `.sqlite`
- `broadwave-live-pp8128.analysis.md` / `.json`

BroadWave is retained as the best accurate kernel baseline. Its measured
critical-path breakdown is exact Q8 824.097 ms, ordinary SGEMM 715.441 ms,
FlashAttention 440.133 ms, copies 289.773 ms, GDN 145.152 ms, owner reduction
53.877 ms, other graph work about 242 ms, and other AffinityWave work about
49 ms.

## FlexWave tail experiments

Exact FlexWave used a 256-thread CTA whose eight warps share one up-to-16-row
A tile across eight adjacent N64 weight tiles. It preserved the legacy
split-K2 order and checked bit-exact in the service oracle.

- Resources: 128 registers/thread, 4,096 B shared, two CTAs/SM.
- Edge-route service: approximately 19.92 ms, faster than BroadWave's
  approximately 20.98 ms.
- End-to-end pp8128: 2568.499 tok/s, slightly slower than BroadWave.

The service gain did not survive the production graph, repeating the
TailWave lesson that isolated tail-heavy routing is not a live selector.

A full-K FlexWave variant reduced resources to 80 registers and 4,096 B
shared, allowing three CTAs/SM. It remained exact on the synthetic oracle but
measured approximately 20.1 ms, no faster than the exact split-K version.
This proves split-K overhead was not the limiting tail mechanism. Both
FlexWave variants are closed.

Evidence:

- `flexwave-v1-edge-service-p2048-ub2048.out`
- `flexwave-v1-pp8128-bench-p8128-ub8128.out`
- `flexwave1-v1-edge-service-p2048-ub2048.out`

## Four-lane row-coalescing experiment

The proposed same-layer rendezvous pools each expert's rows from all four
token lanes before exact-Q8 GEMM. A service-only exact simulation used the
same input rows and identical Q8 weights, changing only descriptor grouping.

The captured edge distribution changes from 256 descriptors and
236 M64 + 12 M32 + 232 M16 tiles to 64 descriptors and
241 M64 + 16 M32 + 42 M16 tiles. Issued-row usefulness rises from 85.3333%
to 98.6513%.

Measured result:

- BroadWave service stage sum: approximately 19.8 ms.
- Coalesced service stage sum: 16.64-16.72 ms.
- Gate/up/down: approximately 5.28/5.28/5.18 ms.
- Zero mismatches over 16,777,216 BF16 outputs per GPU.
- Slowest-device effective throughput: 5.776 TF/s.

Evidence: `rendezvous-broad-edge-service-p2048-ub2048.out`.

Scaling the measured ratio against BroadWave's live 824 ms exact-Q8 slice
gives only about 130 ms gross removal. A real implementation would also
rendezvous four presently diagonal lanes at every MoE layer, pay barrier and
route-transfer costs, and lose cross-layer overlap. It cannot approach the
hundreds of milliseconds needed from a scheduler rewrite. The architecture
is closed for the present target, but the exact measurement and service
prototype remain available for a future topology with cheap same-layer
rendezvous.

## GDN chunkwise-WY investigation

The mapped QKV SGEMM is `blk.<layer>.attn_qkv.weight` times
`attn_norm-<layer>`, shape M=8192, N=2032, K=2048. It contributes about
273.5 ms/GPU; the fused token-serial GDN kernel contributes about 145 ms, so
the recurrent QKV/GDN superstage contains roughly 419 ms.

The existing production kernel
`gated_delta_net_chunked_cuda<128,false,false,2,4>` is token-serial while
keeping recurrent state resident. The model builder's
`build_delta_net_chunking` graph is a genuine chunkwise-WY formulation, but
directly enabling it exposed meta-scheduler requirements for mirrored
broadcasts, batch-preserving MUL_MAT, scalar ops, and chunk views. More
importantly, its per-64-token view ownership would move recurrent state across
AffinityWave lanes each chunk and serialize the graph. The expanded graph is
therefore the wrong integration.

The retained architectural candidate is a lane-local fused CUDA WY
superkernel with explicit lane-boundary state, not the expanded graph. The
experimental graph selector and scheduler diagnostics remain default-off.

## 16-bit lane-transport accuracy closure

BroadWave with BF16 request transport measured 2666.629 tok/s, removing about
111 ms versus F32 BroadWave. Its c512 KLD run produced PPL 4.074815 versus
the accurate AffinityWave value 4.078325, a -0.003510 movement, narrowly
outside the required +/-0.003 gate.

An IEEE-FP16 transport prototype used the same two bytes/value with finer
mantissa precision for normalized activations. It measured 2652.615 tok/s,
about 95 ms faster than F32 BroadWave, but PPL was 4.062837, approximately
-0.01549 relative to accurate AffinityWave. Both formats retained 99.608%
same-top on c512, but both fail the mandatory PPL gate and are closed.

Evidence:

- `broadwave-bf16wire-v1-bench-p8128-ub8128.out`
- `broadwave-bf16wire-kld-ppl-aw-p512-ub512.out`
- `broadwave-f16wire-v1-bench-p8128-ub8128.out`
- `broadwave-f16wire-kld-ppl-aw-p512-ub512.out`
- `broadwave-f16wire-service-service-p2048-ub2048.out`

## MegaWave exact-Q8 result

MegaWave assigned one 1024-thread CTA to an
M64xN128 tile. Two 512-thread groups preserve the legacy lower/upper K16
accumulation chains; each thread owns a 4x4 output microtile. The intended
resource result was achieved exactly: 64 registers/thread, 32 KB shared
memory, no local memory, one CTA/SM, and 32 resident warps versus
BroadWave's 16.

The service oracle was bit-exact, but the stage sum was about 24.0 ms versus
BroadWave's roughly 19.8 ms. Gate/up/down were approximately
7.60/7.60/7.86 ms. The 1024-thread barriers and reduced 4x4 microtile ILP cost
more than doubled warp occupancy could recover. Evidence:
`megawave-v1-edge-service-p2048-ub2048.out`. MegaWave is closed without a
production-graph run.

All selectors remain default-off. No production worktree/build/capture has
been written, and no commit, push, PR, or Claude-memory edit has been made.

## QueryBroadWave FlashAttention experiment

The production Pascal attention kernel is
`flash_attn_tile<256,256,4,8,false>`. It groups all eight GQA heads and four
query tokens per CTA, using 256 threads, 128 registers/thread, 29,184 B
shared memory, and two CTAs/SM. A proposed eight-query tile would halve the
number of CTAs and K/V loads while preserving the same four query
accumulators per thread.

The experimental `flash_attn_tile<256,256,8,8,false>` used 512 threads,
`nbatch_fa=32`, and `nbatch_K=64`. Cubin inspection measured 98
registers/thread, 41,216 B shared memory, no stack, and no local memory. It
therefore achieved the intended one CTA/SM and 16 resident warps, equal to
the production kernel's two CTAs x eight warps.

The generic backend-ops probe initially appeared flat, but its Nsight trace
proved that the test graph had dispatched the non-GQA
`flash_attn_tile<256,256,32,1,false>` kernel. Those probe timings are not
used as production evidence.

The production graph did dispatch `<256,256,8,8,false>`, confirmed by the
captured kernel name. It regressed pp8128 to 2447.516 tok/s from BroadWave's
2572.787 tok/s. On the critical GPU, FlashAttention grew from 440.133 ms to
532.375 ms, while exact-Q8 and SGEMM remained essentially unchanged at
826.515 ms and 715.854 ms. The profiled critical window grew from
3187.831 ms to 3352.912 ms, showing both 92.242 ms of direct attention cost
and additional lost diagonal-wave overlap.

The doubled CTA barrier scope and `nbatch_fa=32` iteration overhead cost more
than K/V reuse saves. This also proves the live kernel is not limited by
redundant K/V global traffic. QueryBroadWave is closed.

Evidence:

- `fa-query4-v2-fa-probe.out`
- `fa-query8-v1-fa-probe.out`
- `fa-query8-v1-fa-trace.nsys-rep`
- `broadwave-fa-query8-v1-bench-p8128-ub8128.out`
- `broadwave-fa-query8-live.nsys-rep` / `.sqlite`
- `broadwave-fa-query8-live.analysis.md` / `.json`

## FabWave blockwise-HFMA2 experiment

The archived Pascal split-HGEMM probe established that K16 FP16 partials
promoted into an FP32 accumulator can retain useful numerical accuracy, but
that arithmetic structure had never been tested in the compute-bound grouped
Q8 projections. FabWave applied it inside the T64/K32 engine. Activations and
dequantized Q8 values are staged as packed half2, each K32 block is evaluated
as two K16 HFMA2 partials, and those partials are promoted and accumulated in
FP32.

The first M64xN128 kernel compiled to genuine Pascal `HFMA2` instructions
with 128 registers/thread, 16,384 B shared memory, no local memory, and two
CTAs/SM. Its captured M64 kernel time fell from BroadWave's 4.459 ms/call to
3.568 ms/call, a 20.0% kernel win. Full edge-route service fell from about
20.99 ms to 18.31 ms because M32/M16 stayed on exact FP32 kernels. Gate,
up, and down measured about 5.76/5.76/5.83 ms. Scaling the M64 ratio against
the live 535.987 ms M64 slice predicts only about 107 ms removal.

A second M64xN64 retile reduced the microtile from 8x4 to 8x2. It achieved
the intended 80 registers/thread, 12,288 B shared memory, no local memory,
three CTAs/SM, and 24 active warps. Despite that occupancy increase, service
regressed to about 20.28 ms. Duplicated A staging, lost N128 reuse, and lower
per-thread ILP outweighed the additional resident CTA.

FabWave's best measured ceiling is below the 300 ms architecture gate, so it
was not advanced to a live numerical run. The N128 selector remains
default-off for future use if a topology makes the M64 slice dominant;
N64 is retained only as a measured closed variant.

Evidence:

- `fabwave-v1-edge-service-p2048-ub2048.out`
- `fabwave-v1-service-trace.nsys-rep`
- `fabwave-n64-v2-edge-service-p2048-ub2048.out`

## Layer-wise physical-lane rotation audit

BroadWave's device totals initially suggested a scheduler opportunity:
FlashAttention accounts for 76.091, 196.299, 317.252, and 440.133 ms on
physical GPUs 0 through 3. A layer-wise permutation could make those totals
look balanced without using unequal token splits.

The source map and timed trace show why that accounting is not a wall-time
model. The scheduler executes cell `(layer, lane)` on diagonal
`layer + lane`, with dependencies from both `(layer - 1, lane)` and
`(layer, lane - 1)`. Since every fourth model layer is full attention, each
steady-state diagonal contains exactly one full-attention cell. A given
attention layer then advances through lanes 0, 1, 2, and 3 on four
successive diagonals. The K/V corridor makes those four causal-prefix steps
ordered, and every diagonal rendezvouses at the global expert-owner service.

Consequently, permuting the physical GPU for a logical lane can redistribute
per-device busy totals but cannot overlap the measured 1:3:5:7 attention
staircase or remove it from the diagonal critical path. Even the optimistic
device-total calculation would reduce the maximum busy union by only about
134 ms:

- Current busy union: 2412.790 / 2467.438 / 2672.358 / 2746.762 ms.
- Replace each GPU's attention total with the 257.444 ms mean:
  2594.143 / 2528.583 / 2612.550 / 2564.073 ms.
- Optimistic maximum reduction: 134.212 ms before migration, corridor, cache
  placement, or lost-overlap costs.

Simple layer-wise rotation is therefore below the 300 ms architecture gate
and structurally cannot deliver its apparent 183 ms attention-total ceiling.
It is closed before implementation. A scheduler breakthrough must change the
logical DAG itself, for example with finer token cells plus work-conserving
dispatch, or replace the causal attention staircase with a context-parallel
superstage.

Evidence:

- `broadwave-live-pp8128.nsys-rep` / `.sqlite`
- `broadwave-live-pp8128.analysis.md` / `.json`
- `ggml/src/ggml-backend-meta.cpp`, `submit_diagonal` and corridor loops

## DenseWave prepacked-input closure

The 8.38-TF/s DenseWave prototype rereads each activation tile for every
output-channel tile, apparently issuing roughly 2 GiB of F32 activation
traffic for one 8192x2032x2048 QKV GEMM. A controlled variant preconverted
the input once to FP16 and consumed those exact cuBLAS operands directly.
The GEMM shape, T64 Q8 weights, K32 blockwise promotion, output, grid, and
resource shape were unchanged.

Measured over 12 repeats:

- F32 input with in-kernel conversion: 8.167975 ms, 8.347553 TF/s.
- Prepacked FP16 input: 7.993928 ms, 8.529299 TF/s.
- Improvement: 0.174047 ms, or 2.18%.
- Numerical result was unchanged: relative L2 0.00215612385 and maximum
  absolute difference 0.0402393341 from FP32-accumulating cuBLAS.
- Both kernels compile to 96 registers, 12,672 B shared memory, and zero
  spills.

The nominal duplicated input traffic is therefore cache-served or hidden
behind the HFMA2/promotion mainloop. Removing it does not move the kernel
toward 12-14 TF/s and cannot produce a 300 ms live contribution. The
prepacked-input hypothesis is closed; no integration or tile sweep follows.

Evidence:

- `densewave-microbench.cu`
- `densewave-microbench-v2`
- `densewave-prepacked-v2.json`

## LayerWave placement replay and closure

LayerWave was modeled as a replacement placement, not a physical relabeling
of the existing AffinityWave lanes. GPU `layer mod 4` owns the complete dense,
recurrent, and expert state for a layer and executes all four token-lane cells
for that layer. This removes the expert request/response exchange, owner
reduction, hot replicas, and same-layer recurrent/KV corridor. The remaining
mandatory edge is the 2,032x2,048 F32 hidden activation between consecutive
layers, 16.65 MB or about 1.716 ms at the measured 9.7 GB/s all-to-all rate.

An initial replay incorrectly attributed only the `diag/.../pre` NVTX ranges.
With `GGML_CUDA_AW_CORRIDOR_STATE_SPLIT`, lanes 0 through 2 submit their
FlashAttention and attention-output tail after that range. Comparing complete
device activity with the cell ranges recovered missing attention work of:

- lane 0: 15.2952 ms per full-attention cell
- lane 1: 27.2913 ms
- lane 2: 39.3804 ms
- lane 3: already contained in the attributed range

The corrected full-attention cell costs are approximately 55.13, 67.12,
79.21, and 91.47 ms for lanes 0 through 3. This correction invalidates the
earlier optimistic LayerWave number.

With all measured work restored, the simple periodic placement has per-device
work of 2181.35, 2178.61, 2185.41, and 2860.95 ms. The resource-constrained
replay is 3031.583 ms, about 2681 tok/s. Even an infinite-resource DAG replay
is 2735.788 ms, about 2971 tok/s. The last lane's causal-attention staircase
remains concentrated on one layer color, so fixed LayerWave cannot clear the
300 ms architecture gate.

A search over 100,000 balanced block-cyclic assignments was then performed.
Each four-layer block assigns one layer to each GPU, preserving ten layers per
GPU while permuting which physical GPU receives each layer position. Fixed
placement remained 3031.58 ms; simple rotations and snake assignments ranged
from about 3352 to 3550 ms. The best random assignment was 3152.686 ms, about
2578 tok/s. Moving the attention-heavy layer between GPUs introduces enough
cross-block stream conflict to outweigh its load balance. Fixed and
block-permuted LayerWave are closed.

The trace-attribution correction is retained as a required invariant for all
later placement replays: a cell model must include activity outside its
`diag/.../pre` range and reconcile back to each complete device timeline.

## HeadWave attention placement probe

HeadWave changes both placement and arithmetic. At every full-attention layer,
query heads are partitioned across all four GPUs while each device runs the
existing FlashAttention kernel on a smaller independent head group. The
proposal therefore removes arithmetic from the causal lane owner rather than
merely relabeling that owner. It requires explicit Q-head transfer, replicated
KV state for the relevant GQA group, and return or reduction of the attention
result. Those PCIe costs are not yet assumed free.

Exact production-shape backend probes used head dimensions K=V=256,
2,032 query tokens, F16 K/V, F32 precision, one KV head, and causal prefixes
of 2,032 through 8,128 tokens. Measured kernel times were:

| KV prefix | 2 query heads | 4 query heads | 8 query heads |
|-----------|---------------|---------------|---------------|
| 2,032     | 1.823 ms      | 3.455 ms      | 6.780 ms      |
| 4,064     | 3.419 ms      | 6.581 ms      | 13.011 ms     |
| 6,096     | 5.003 ms      | 9.739 ms      | 19.348 ms     |
| 8,128     | 6.578 ms      | 12.750 ms     | 25.377 ms     |

The full-prefix kernel scales almost linearly with query-head count:
25.377 -> 12.750 -> 6.578 ms for 8 -> 4 -> 2 heads. Thus four-way head
partitioning exposes real compute parallelism on Pascal and avoids the
structural failure of physical lane rotation. This is a positive arithmetic
gate only, not an end-to-end result. HeadWave advances to measured concurrent
P2P and gang-scheduling accounting; it will be closed unless that combined
model removes at least 300 ms or composes into the required 660+ ms program.

Evidence:

- `headwave-v1-fa-head-probe.out`
- `headwave-v1-fa-head-probe.err`
- `run-fa-head-probe.sh`
- `tests/test-backend-ops.cpp`

## HeadWave PCIe and full-sequence closure

A four-GPU CUDA peer-copy probe measured the actual candidate traffic rather
than scaling a single-link bandwidth number:

- 16.65 MB one-to-three hidden broadcast: 3.8087 ms
- three 8.32 MB head-result returns: 1.8940 ms
- three 16.65 MB partial-hidden returns: 3.7880 ms
- four-way 16.65 MB-per-rank hidden all-gather: 6.8024 ms
- four-way 4.16 MB-per-peer reduce-scatter traffic: 1.6300 ms
- four-way 33.29 MB F32 feature-shard all-gather: 12.9224 ms
- four-way 16.65 MB BF16 feature-shard all-gather: 6.8866 ms

The optimized per-lane HeadWave path can therefore broadcast its input and
return head outputs in about 5.70 ms, much less than the original rough
12.5 ms estimate. Sharding the Q/gate projection to M=2,048 measures
2.339 ms versus 10.163 ms at M=8,192. A K-sharded output projection
M=2,048, K=1,024 measures 1.184 ms versus 4.643 ms at K=4,096.

Despite those positive components, inserting a four-GPU gang into the
existing diagonal is structurally wrong. Each steady diagonal also contains
three recurrent cells. Running those first and then the attention gang costs
roughly `28.6 ms + HeadWave`, which regresses the short-prefix diagonals and
only helps the longest prefix. Two- and three-GPU variants leave recurrent
cells queued and do not repair the problem. HeadWave is retained only as a
component of a replacement placement.

A full 8,128-query rendezvous was also tested. With one KV head its q4
kernel takes 50.420 ms, while four 2,032-query calls at prefixes 2,032,
4,064, 6,096, and 8,128 sum to 32.527 ms. The generic kernel evaluates the
square masked domain instead of exploiting the causal triangle, so a single
full-sequence launch is 55% slower. A replacement scheduler must retain
prefix-sliced FA calls or add a triangular tile scheduler.

Replicating both KV heads on every GPU does not recover efficiency:
`nh=2,q_per_kv=2` is effectively identical to `nh=1,q_per_kv=4` at every
prefix. Full-prefix times are 12.749 and 12.761 ms respectively. The kernel
scales linearly with total query-head work, not KV-head packing.

Evidence:

- `headwave-p2p-probe.cu`
- `headwave-p2p-probe`
- `headwave-v1-headwave-p2p.out`
- `headwave-v2-fullseq-fa-head-probe.out`
- `meshwave-v1-fa-layout-fa-head-probe.out`

## CrossWave component-placement closure

CrossWave decoupled dense/token placement from expert placement. Dense,
recurrent, and attention work remained on mirrored token-lane GPUs, while
GPU `layer mod 4` owned all 256 experts for a layer. The four cells in a
steady diagonal therefore target four different expert GPUs. Each cell
sends one 16.65 MB hidden tensor to its layer owner and receives one result,
instead of fanning routed rows across four expert owners. The proposal
removes owner reduction and the global service barrier while preserving
roughly M=64 expert shapes.

The measured trace was converted into 480 non-preemptive intervals:
pre-cell compute, fixed-owner expert compute, and post-cell compute.
Dependencies include both layer and token-lane edges, measured 1.3 ms peer
latency for a full hidden transfer, and shared SM resources. A CP-SAT model
then solved placement/scheduling rather than assuming perfect balance.

Base measured arithmetic:

- Aggregate SM work after removing owner reduction and service copies:
  9,168.0 GPU-ms, or a 2,292.0 ms perfect-balance floor.
- Fixed-lane feasible schedule: 2,930.8 ms, 2,773 tok/s.
- Fixed-lane proven lower bound after 30 seconds: 2,807.0 ms.
- Free placement was highly symmetric and did not produce a better
  admissible incumbent; a one-cell-per-GPU layer permutation also converged
  back to the static assignment.

Composing the retained measured factors still misses decisively:

- DenseWave factor: 0.877
- fused-WY GDN target factor: 0.345
- FabWave exact-Q8 service factor: 0.870
- Aggregate compute: 8,025.1 GPU-ms, 2,006.3 ms balance floor
- Fixed-lane feasible schedule: 2,608.1 ms, 3,116 tok/s
- Fixed-lane lower bound: 2,492.1 ms

CrossWave is therefore closed for the 4,000 tok/s target. Its useful result
is that removing service communication is insufficient while the
cell-by-cell lane DAG remains.

Evidence:

- `crosswave-model.py`
- `crosswave-cpsat.py`
- `model-venv/`

## AtlasWave replicated-activation expert mesh

AtlasWave is the next replacement placement and differs from the previously
rejected token-slice TP:

1. A full-prompt hidden activation is replicated on all four GPUs.
2. Dense QKV weights are output-channel sharded. GDN state is channel-local;
   full attention is query-head-local. Their row-parallel output projection
   uses one panel-streamed hidden all-reduce.
3. Expert weights remain expert-axis sharded, 64 complete experts per GPU.
   Because each GPU already has every token activation, it gathers all rows
   for its local experts without request P2P. Pooling all four token lanes
   raises the mean expert M while preserving complete N=512/K=512 expert
   matrices.
4. Each GPU reduces its local expert routes into a partial full-prompt
   hidden tensor; one panel-streamed all-reduce restores the replicated
   activation.

This preserves the productive expert M dimension that old row-split TP
destroyed. It also avoids the K=128 down projection exposed by a pure
feature-sharded expert mesh. Direct microbenchmarks show why that matters:
M=2,048,N=256,K=128 reaches only 3.32 TF/s, versus 4.21 TF/s at K=512.

The dense feature-shard shape M=2,048,N=8,128,K=2,048 measures 6.731 TF/s
with cuBLAS and 7.411 TF/s with prepacked DenseWave. This is approximately
the current dense rate rather than an arithmetic win, but it avoids a major
placement regression. Prefix-sliced head-local FA remains required because
the full-square launch is slower.

### Coalesced FabWave arithmetic probe

The service benchmark gained a diagnostic-only
`GGML_CUDA_AW_BENCH_COALESCE=1` switch so row pooling can be composed with a
selected arithmetic engine without changing live scheduler behavior.
Four-lane coalescing plus FabWave measured:

- descriptors: 256 -> 64
- useful issued-row fraction: 85.333% -> 98.651%
- stage sum: approximately 19.8 ms BroadWave -> 13.83-13.99 ms
- gate/up/down: approximately 4.32/4.32/4.26 ms
- elapsed effective throughput: 6.879-7.099 TF/s across four GPUs

This is the first measured placement-plus-arithmetic composition at the
original approximately 7 TF/s service reopening threshold. It is a
diagnostic pass, not an accuracy pass. FabWave's blockwise HFMA2 arithmetic
fails the byte-exact BF16 owner-partial oracle; the earlier FabWave timing
also ran with checking disabled. It cannot advance to an accurate backend
until a live KLD/PPL gate passes or exact accumulation is restored.

The first live c512 KLD/PPL gate also fails:

- baseline PPL: 4.082694
- FabWave PPL: 4.076708
- PPL delta: -0.005986 (outside the required +/-0.003)
- mean KLD: 0.000889
- maximum KLD: 0.026041
- same-top token fraction: 99.216% (below the required 99.5%)

The approximately 7 TF/s result is therefore not an admissible backend as
implemented. The next bounded arithmetic experiment is to halve the HFMA2
accumulation interval from 16 scalar K terms to 8 before promotion to FP32.
This directly attacks the measured numerical error without changing weight
precision, routing, placement, or output reduction. It must pass both the
accuracy gate and retain a large fraction of the coalesced service gain.

K8 promotion was compiled as a default-off `GGML_CUDA_AW_FAB_K8=1`
diagnostic. `cuobjdump` reports the same resources as K16: 128 registers and
16 KB shared memory. On the coalesced production route distribution it
measures 5.845-6.159 effective TF/s, with stage sums of 16.28-16.40 ms. This
is approximately 15% slower than K16 but still above the 5.5 TF/s
continuation gate.

Its live c512 gate fails:

- K8 PPL: 4.071006
- baseline PPL: 4.082694
- PPL delta: -0.011688
- mean KLD: 0.000826
- maximum KLD: 0.026511
- same-top token fraction: 99.608%

The same-top gate recovers, but the PPL gate gets materially worse and mean
KLD barely moves. Promotion cadence is therefore closed as the primary
accuracy repair. This points to FP16 operand formation rather than half
accumulator depth. The next mechanism keeps the Q8 integer exactly
representable in FP16 and applies its original half block scale in FP32
after each promoted partial. If FP16 activation conversion remains the
limiter, the planned extension represents each activation as FP16 high and
FP16 residual rails.

### Post-scale integer HFMA2

`GGML_CUDA_AW_FAB_QSCALE=1` stores the signed Q8 integer directly in the
half2 B tile. Every Q8 value is exactly representable in FP16. The original
half block scale is retained separately and applied to each promoted
partial in FP32, removing the `half(q*scale)` rounding from FabWave.

With K8 promotion, the coalesced route distribution retains 5.895-6.062
effective TF/s and 16.42-16.50 ms stage sums. Its live c512 result is:

- post-scale K8 PPL: 4.082115
- baseline PPL: 4.082694
- PPL delta: -0.000579 (passes)
- mean KLD: 0.000899
- maximum KLD: 0.023151
- same-top token fraction: 99.216% (fails)

The PPL bias is almost completely removed without a material throughput
cost. Weight-value rounding was therefore responsible for that bias.
However, the unchanged same-top rate shows activation rounding and/or half
partial accumulation still causes decision flips. Post-scale is retained as
the base for a two-rail activation experiment: `a_hi=half(a)` and
`a_lo=half(a-float(a_hi))`, accumulated sequentially against the exact Q8
integer before FP32 scaling.

### Full two-rail activation closure

The high/low activation prototype uses 128 registers, 24,832 bytes shared
memory, and 32 bytes stack per thread. It runs the two rails sequentially so
the half accumulator array is not doubled. Despite preserving two CTAs/SM,
the second HFMA2 stream dominates:

- effective throughput: 3.628-3.688 TF/s
- stage sum: 27.58-27.59 ms
- gate/up/down: approximately 8.96/8.95/8.71 ms

This misses the 5.5 TF/s microbenchmark continuation gate and is slower than
the exact coalesced path. Full two-rail correction is closed without a live
KLD run. Any residual correction must be selective enough to preserve the
fast K16 arithmetic base. Before designing that selector, the unmeasured
K16 plus post-scale composition will determine whether removing weight
rounding is already sufficient at the fastest promotion interval.

K16 plus post-scale preserves the fast arithmetic:

- effective throughput: 6.778-7.214 TF/s
- stage sum: 13.92-13.99 ms
- gate/up/down: approximately 4.35/4.35/4.27 ms

It still fails the live gate:

- K16 post-scale PPL: 4.069142
- baseline PPL: 4.082694
- PPL delta: -0.013552
- mean KLD: 0.000785
- maximum KLD: 0.019124
- same-top token fraction: 99.216%

The two fast one-rail variants and the accurate-bias K8 variant do not meet
both accuracy requirements; the full correction loses the arithmetic gain.
Half-precision promotion is therefore closed as a standalone breakthrough.
The retained AtlasWave row pooling will next be composed with an exact FP32
kernel that targets the measured occupancy ceiling: a smaller output tile,
compact Q8 staging, no 32 KB CTA allocation, and a <=85-register target for
three CTAs/SM.

### Projection-selective exactness

Because gate, up, and down are separate launches, a mixed arithmetic backend
can retain fast integer-HFMA2 on selected projections and dispatch the
existing bit-exact BroadWave M64 kernel on the numerically sensitive
projection. This is controlled diagnostically by
`GGML_CUDA_AW_FAB_EXACT=down|gateup`.

Down-exact plus K16 post-scale gate/up measures:

- effective throughput: 6.514-6.624 TF/s
- stage sum: 14.88-14.90 ms
- gate/up/down: approximately 4.37/4.37/5.19 ms

Its live c512 result is:

- PPL: 4.065101
- PPL delta: -0.017594 (fails)
- mean/max KLD: 0.000782/0.026953
- same-top: 99.608% (passes)

Restoring the down projection exactly fixes the same-top gate but makes the
PPL bias more negative. This localizes most systematic bias to approximate
gate/up; approximate down had partially cancelled it. The complementary
gate/up-exact test is therefore warranted and should isolate the smaller
opposite-signed down error while retaining fast K16 arithmetic on down.

Gate/up-exact plus K16 post-scale down measures 6.093-6.132 TF/s, but fails
both live gates: PPL delta -0.008765 and same-top 99.216%.

The final evidence-based composition used K8 post-scale gate/up and exact
BroadWave down. It measures 5.773-6.115 TF/s with a 16.49-16.60 ms stage
sum. Its live c512 result meets the campaign's earlier statistical limits:

- PPL: 4.082931
- PPL delta: +0.000237
- mean/max KLD: 0.000758/0.019712
- same-top: 99.608%

The user subsequently clarified that no accuracy sacrifice is acceptable.
This candidate is not byte-identical, so it is rejected despite passing the
previous PPL and same-top thresholds. All HFMA2 expert paths are now closed
for advancement. They remain default-off diagnostic evidence only.
Subsequent breakthrough work must preserve the established FP32 FMA chains
and byte-exact oracle.

Evidence:

- `meshwave-coalesced-fab-v1-service-p2048-ub2048.err`
- `meshwave-coalesced-fab-v2-service-p2048-ub2048.out`
- `fabwave-kld-v1-ppl-aw-p512-ub512.out`
- `meshwave-coalesced-fab-k8-v1-service-p2048-ub2048.out`
- `fabwave-k8-kld-v1-ppl-aw-p512-ub512.out`
- `meshwave-coalesced-fab-k8-qscale-v1-service-p2048-ub2048.out`
- `fabwave-k8-qscale-kld-v1-ppl-aw-p512-ub512.out`
- `meshwave-coalesced-fab-dual-v1-service-p2048-ub2048.out`
- `meshwave-coalesced-fab-k16-qscale-v1-service-p2048-ub2048.out`
- `fabwave-k16-qscale-kld-v1-ppl-aw-p512-ub512.out`
- `meshwave-coalesced-fab-k16-qscale-downexact-v1-service-p2048-ub2048.out`
- `fabwave-k16-qscale-downexact-kld-v1-ppl-aw-p512-ub512.out`
- `meshwave-coalesced-fab-k16-qscale-gateupexact-v1-service-p2048-ub2048.out`
- `fabwave-k16-qscale-gateupexact-kld-v1-ppl-aw-p512-ub512.out`
- `meshwave-coalesced-fab-k8-qscale-downexact-v1-service-p2048-ub2048.out`
- `fabwave-k8-qscale-downexact-kld-v1-ppl-aw-p512-ub512.out`
- `densewave-microbench-v2`
- `ggml/src/ggml-cuda/affinity-wave.cu`

## Exact AtlasWave pooling and lower bound

The AtlasWave expert-placement primitive has now been measured with the
existing exact BroadWave FP32 accumulator. The service diagnostic can pool
one, two, or four token cells into each expert owner without changing the
Q8 weights or arithmetic. Every run used checking and reported zero
mismatches over 16,777,216 BF16 outputs on every GPU.

Measured on the captured production route distribution:

- one-cell pool: 21.00 ms stage, 85.333% useful issued rows, 4.759 effective
  TF/s on the slowest GPU
- two-cell pool: 17.84 ms stage, 94.815% useful issued rows, 5.44-5.73
  effective TF/s
- four-cell pool: 16.58 ms stage, 98.651% useful issued rows, 6.219 effective
  TF/s

The four-cell result is an exact approximately 31% effective-throughput
gain. Unlike the previously closed row-coalescing proposal, AtlasWave does
not add a rendezvous solely to improve padding. Its replicated input
placement makes every route for the 64 locally owned experts already
available on the owner GPU, so the pooling is a natural consequence of the
replacement architecture.

An executable lower-bound model was built directly from
`broadwave-live-pp8128.analysis.json`, including the measured non-expert
families and the previously omitted attention tails. Before communication,
with current exact dense and GDN arithmetic:

- one-cell pool: 2,268.968 ms, 3,582 tok/s
- two-cell pool: 2,143.801 ms, 3,791 tok/s
- four-cell pool: 2,093.575 ms, 3,882 tok/s

The target is 2,032 ms for 4,000 tok/s. Full four-cell pooling is therefore
mandatory, but it still misses by 61.575 ms before paying communication.
One- and two-cell token panels sacrifice too much exact expert efficiency
and are closed as the primary AtlasWave schedule.

With the previously proposed, still unmeasured GDN factor of 0.345, the
four-cell compute floor becomes 1,998.669 ms, or 4,066.7 tok/s. That leaves
only 33.331 ms for the gross two full-hidden all-reduces per layer. At the
current provisional 10.3 ms per all-reduce estimate, 95.95% of
communication would need to overlap. This is not yet a feasible schedule;
it is a quantitative requirement. Exact GDN acceleration and feature-panel
communication must both be measured before AtlasWave can be retained.

Evidence:

- `atlas-exact-pool1-v1-service-p2048-ub2048.out`
- `atlas-exact-pool2-v1-service-p2048-ub2048.out`
- `atlas-exact-pool4-v1-service-p2048-ub2048.out`
- `atlaswave-exact-model.py`
- `atlaswave-exact-model-v1.json`
- `atlaswave-exact-gdn-target-model-v1.json`

## Expanded WY graph outside AffinityWave

A standard four-GPU tensor-split control at c512 measured 369.515 ms
(1,385.599 tok/s). Enabling the existing expanded chunkwise-WY graph did
not reach execution. Graph allocation aborted at
`ggml-backend-meta.cpp:675` because the inferred split axis remained
unknown. The loader still activates the expert-placement repack callbacks,
and the expanded recurrence introduces a split-state combination that the
old metadata layer cannot represent.

This is not a WY performance or accuracy result. It reproduces the same
pre-execution metadata failure seen in earlier graph attempts and must not
be cited as evidence against chunkwise-WY arithmetic. The next valid probe
must isolate a single channel-local GDN tensor, or implement it inside the
AtlasWave placement where recurrent state and QKV channels never cross
devices. No reassociated recurrence will be retained without an explicit
output/reference accuracy gate.

Evidence:

- `wy-base-control-v1-bench-base-p512-ub512.out`
- `wy-base-expanded-v1-bench-base-p512-ub512.out`
- `wy-base-expanded-v1-bench-base-p512-ub512.err`

## Exact GDN subwarp reduction probe

The exact `gated_delta_net_chunked_cuda<128,false,false,2,4>` target measured
2,495.21 us for H=32, T=1,024. Its loop SASS completes all recurrence work
for the first of two columns before beginning the second. Each column pays
two five-step warp XOR reductions.

A default-off `GGML_CUDA_AW_GDN_SUBWARP=1` prototype mapped each column to a
16-lane logical subwarp. It reconstructed the existing reduction tree
exactly: each logical lane accumulated the original lane and lane+16
strands separately, added them in the same order as the original first XOR
step, and then executed offsets 8, 4, 2, and 1. Per-state update order and
FP32 operations were retained. The CUDA backend GDN correctness suite
passed.

The mechanism is decisively slower:

- existing exact c2/w4: 2,495.21 us
- exact subwarp16: 4,215.10 us
- subwarp resource use: 62 registers, versus 48 for c2/w4

Although it halves cross-lane reduction work, each half-warp must issue its
own q/k loads and hold two row strands. The additional load instructions
and register dependency graph dominate. Logical subwarp reduction is
closed.

The SASS audit exposes a narrower exact opportunity that is not another
tile sweep: c2/w4 already owns two independent columns per warp, but
serializes their complete dependency chains. A paired-column mainloop can
interleave the two existing FP32 chains, including their reductions,
without reassociating either chain. This is the next measured kernel.

The paired-column kernel passed the CUDA GDN suite but measured 2,829.58 us
and used 56 registers. Interleaving independent reduction chains does not
hide enough latency to repay the larger live register and scheduling set;
it is 13.4% slower than c2/w4 and is closed.

Precomputing the scalar decay once per head/token, rather than redundantly
executing `expf` in all 64 column warps, is exact because the same FP32
result is stored and reloaded. `GGML_CUDA_AW_GDN_PREEXP=1` retains the
48-register recurrence kernel, passes the CUDA suite, and measures
2,415.61 us including the precompute kernel. This is a real 3.2% GDN gain,
but only about 5 ms on the live 145 ms slice. It is documented for later
fusion into the upstream alpha/gate producer and is not advanced as a
frontier mechanism.

## Layer-local chunkwise-WY measurement

The earlier "base" WY probe was not actually free of split metadata:
expert-parallel and EPLB variables remained enabled. A new true layer-split
control places every recurrent layer and its full graph on one GPU, avoiding
all tensor-axis inference.

The existing expanded chunkwise-WY graph now executes, but loses at both
measured lengths:

- c512 fused control: 1,408.376 ms, 363.539 tok/s
- c512 expanded WY: 1,483.816 ms, 345.056 tok/s
- pp8128/ub2048 fused control: 16,681.575 ms, 487.244 tok/s
- pp8128/ub2048 expanded WY: 17,774.409 ms, 457.287 tok/s

The target-length graph is 6.55% slower. Its many materialized tensors,
small triangular operations, and per-chunk graph launches erase the
parallel GEMM benefit on P100. Since it is slower, reassociation accuracy
testing cannot make it admissible. The existing expanded graph is closed.
A fused chunk engine would be a distinct architecture and needs its own
microbenchmark.

Evidence:

- `wy-layer-control-v1-bench-layer-p512-ub512.out`
- `wy-layer-expanded-v1-bench-layer-p512-ub512.out`
- `wy-layer-control-pp8128-ub2048-v1-bench-layer-p8128-ub2048.out`
- `wy-layer-expanded-pp8128-ub2048-v1-bench-layer-p8128-ub2048.out`

## RelayWave placement hypothesis

AtlasWave's full replicated hidden state requires two full-hidden
all-reduces per layer: one after a feature-sharded dense stage and one after
expert owners. RelayWave retains only the measured component that matters,
full four-lane pooling at expert owners, while changing the boundary
placement:

1. Expert owners consume a replicated normalized activation and execute all
   routes for their 64 experts with the exact four-cell pool.
2. Owner partials are reduced-scattered by token lane. Each token owner
   receives one complete quarter-prompt hidden activation.
3. Dense, recurrent, and attention work runs token-local with complete
   dense weights, as in the accurate AffinityWave path. Lane ownership can
   rotate by layer to distribute the causal-attention staircase.
4. Token lanes are all-gathered only at the next expert rendezvous.

Reduce-scatter plus all-gather moves the same bytes as one ring all-reduce,
half the gross communication of AtlasWave, while preserving complete
expert K/N shapes and the measured 98.651% useful rows. The architecture is
bulk-synchronous only at the expert pool; token-owner panels can resume
dense work as their reduce-scatter panels complete. Its next gate is a
deterministic F32 four-GPU reduce-scatter/all-gather probe at the exact
8,128 x 2,048 activation shape.

## RelayWave F32 communication probe

A four-GPU probe now measures the exact RelayWave boundary operation. Its
custom path uses deterministic F32 additions in fixed GPU-rank order and
validates every output element against the corresponding fixed-order host
expression. The exact 8,128 x 2,048 tensor measured:

- custom reduce-scatter: 6.644 ms
- custom all-gather: 7.923 ms
- custom combined: 15.580 ms
- NCCL all-reduce: 10.367 ms
- NCCL reduce-scatter plus all-gather: 11.006 ms
- validation errors: zero

A second full-size run measured 15.075 ms custom combined and 10.317 ms
NCCL all-reduce. The custom copy-engine path is slower in isolation, but it
is the relevant mechanism because earlier probes showed peer copies can
overlap SM work, while the tested NCCL kernels competed for SM issue slots.

Feature-panel results were:

| hidden columns | custom RS (ms) | custom AG (ms) | combined (ms) | NCCL AR (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.269 | 0.262 | 0.507 | 0.337 |
| 128 | 0.528 | 0.532 | 0.972 | 0.658 |
| 256 | 0.953 | 0.983 | 1.944 | 1.320 |
| 512 | 1.966 | 1.972 | 3.944 | 2.622 |
| 1,024 | 3.573 | 3.857 | 7.498 | 5.175 |
| 2,048 | 6.849 | 7.947 | 15.075 | 10.317 |

All panel sizes reported zero validation errors. Communication is linear
enough that panels do not reduce gross transfer time; their purpose is to
expose producer/consumer overlap and reduce the final uncovered tail.

Evidence:

- `atlaswave-allreduce-probe.cu`
- `run-atlaswave-allreduce-probe.sh`
- `relaywave-allreduce-f32-v1.out`
- `relaywave-allreduce-f32-panels-v1.out`

## RelayWave dense output-panel timing

The existing DenseWave microbenchmark was used only to time independent
cuBLAS output-column panels for the production recurrent-output shape
N=2,032, K=4,096. Results were:

| output columns | time per call (ms) | calls for 2,048 columns | total (ms) |
| ---: | ---: | ---: | ---: |
| 2,048 | 5.210 | 1 | 5.210 |
| 512 | 1.138 | 4 | 4.552 |
| 256 | 0.597 | 8 | 4.772 |
| 128 | 0.378 | 16 | 6.050 |
| 64 | 0.182 | 32 | 5.827 |

Four 512-column calls are 12.6% faster than the monolithic call while also
creating four producer events for communication. A simple producer/AG
pipeline estimate is about 9.0 ms instead of 13.2 ms for serial monolithic
compute plus full all-gather.

This is not yet an admissible accuracy result. Independent M sizes can make
cuBLAS select different kernels or K-reduction schedules. The existing
microbenchmark compares each call against its own high-precision reference,
not panelized output against the immutable monolithic call. RelayWave must
count bitwise mismatches between those two device results before output
panelization can be retained. If they differ, the mechanism is rejected
unless the production algorithm can be forced identically for both layouts.

Evidence:

- `relaywave-dense-output-panels-v1.out`

The actual four-GPU producer/communication pipeline was then measured at
the exact shape with four complete dense-weight replicas. Each token owner
computes four 512-column panels and publishes a panel immediately to all
four expert-owner GPUs through peer-copy streams. Results:

- monolithic GEMM: 5.186 ms
- four panel GEMMs: 5.394 ms
- monolithic all-gather alone: 7.945 ms
- panel all-gather alone: 8.344 ms
- monolithic GEMM/all-gather pipeline: 10.506 ms
- four-panel pipeline: 7.749 ms
- measured stage speedup: 1.356x

The timing is a substantial placement result, but the default cuBLAS form
fails the exactness gate. Of 4,161,536 local FP32 outputs per four GPUs,
1,435,104 differed bitwise from the monolithic call. The gathered check
found the expected fourfold 5,740,416 mismatches. Output columns are
mathematically independent, so this is internal cuBLAS kernel/reduction
selection caused by the changed M shape, not a cross-panel reduction.
Nevertheless, this implementation is rejected unless an explicit Pascal
cuBLAS algorithm reproduces the production result bit-for-bit.

Evidence:

- `relaywave-dense-ag-probe.cu`
- `run-relaywave-dense-ag-probe.sh`
- `relaywave-dense-ag-v1.out`

That condition was subsequently satisfied. An exhaustive exact-shape audit
tested legacy cuBLAS algorithms -1 and 0 through 23, tensor-op selector 99,
and selectors 100 through 115 for packed and strided panels of 1,024, 512,
256, 128, and 64 output columns. Pascal algorithm 3 is the best exact
four-panel result:

- 512-column packed panels, algorithm 3: 4.506 ms
- bitwise mismatches versus monolithic production selector 99: zero
- 512-column strided panels, algorithm 3: 4.509 ms, zero mismatches

Algorithms 2, 4, 5, and 6 are also bitwise exact for 512-column panels but
are slower. The default panel selector caused the earlier mismatches.
Algorithm 3 preserves the production result while exposing panel events.

The full four-GPU pipeline was rerun with production math mode, monolithic
selector 99, and panel selector 3. A 30-repeat measurement reported:

- monolithic GEMM: 5.159 ms
- four exact panel GEMMs: 4.600 ms
- monolithic all-gather alone: 7.468 ms
- panel all-gather alone: 7.905 ms
- monolithic pipeline: 10.387 ms
- exact panel pipeline: 7.486 ms
- pipeline speedup: 1.388x
- local bitwise mismatches: zero
- gathered bitwise mismatches: zero

This is the first retained exact RelayWave arithmetic/placement composition.
At one occurrence per 40 primary layers, the directly measured stage delta
is about 116 ms before integration effects.

Evidence:

- `relaywave-cublas-exactness-probe.cu`
- `run-relaywave-cublas-exactness-probe.sh`
- `relaywave-cublas-exactness-v1.out`
- `relaywave-dense-ag-exact-v2.out`

### Exact selector audit across all live dense shapes

The exactness probe was generalized and run across every target SGEMM shape
mapped from the production graph. It compared each candidate directly to
selector 99, the `CUBLAS_GEMM_DEFAULT_TENSOR_OP` used by ggml-cuda.

- M=8,192, N=2,032, K=2,048:
  - selector 99 monolithic: 9.277 ms
  - two 4,096-column panels, algorithm 5: 8.300 ms
  - speedup: 1.118x; bitwise mismatches: zero
- M=4,096, N=2,032, K=2,048:
  - selector 99 monolithic: 5.197 ms
  - two 2,048-column panels, algorithm 6: 4.304 ms
  - speedup: 1.208x; bitwise mismatches: zero
- M=2,048, N=2,032, K=4,096:
  - selector 99 monolithic: approximately 4.725 ms in the selector audit
  - four 512-column panels, algorithm 3: 4.506 ms
  - speedup: 1.049x; bitwise mismatches: zero
- M=2,048, N=2,032, K=512:
  - selector 99 monolithic: 0.623 ms
  - algorithm 3 monolithic: 0.572 ms
  - speedup: 1.089x; bitwise mismatches: zero

The M=8,192 and M=4,096 gains are material because those families account
for approximately 1,458.5 and 560.8 aggregate GPU-ms respectively. On a
perfectly balanced four-GPU placement, the measured factors remove about
38 and 24 ms of wall time. The K=512 selector removes about 2 ms. These are
retained exact arithmetic components, although the replacement placement
still needs a communication schedule that does not add more than they save.

Evidence:

- `relaywave-cublas-exactness-m8192-k2048-v1.out`
- `relaywave-cublas-exactness-m4096-k2048-v1.out`
- `relaywave-cublas-exactness-m2048-k512-v1.out`

## Exact feature-streamed expert continuation

To preserve full four-lane expert pooling while exposing all-gather
overlap, the BroadWave M64/N128 kernel gained a diagnostic continuation
mode. It preserves its two independent FP32 accumulator chains across
feature panels. Each chain is stored and reloaded as FP32, then the existing
final `acc0 + acc1` is executed after the last panel. No FMA or final
addition is reassociated. M32 and M16 tails remain full-K launches after the
last panel.

The continuation kernel compiles at 123 registers and 32,768 bytes shared
memory, versus 122-124 registers and the same shared memory for the original
kernel. On the exact pool-four distribution:

- original stage: approximately 16.58 ms
- two K panels: 17.22-17.30 ms stage sum; 11.11-11.15 ms gate/up
- four K panels: 17.96-18.00 ms stage sum; 11.85-11.87 ms gate/up
- every device checked 16,777,216 BF16 owner outputs with zero mismatches

The exact continuation overhead is therefore only about 0.69 ms for two
panels or 1.39 ms for four. Arithmetic continuation itself is retained.

The decisive combined test then performed a real four-source peer
all-gather into panel-major buffers before each continuation launch, with
cross-device events and consumer protection:

- two panels: approximately 26.2-28.4 ms elapsed; gate/up plus copies
  approximately 19.8-20.7 ms
- four panels: approximately 26.3-27.6 ms elapsed; gate/up plus copies
  approximately 20.1-21.0 ms
- all devices still reported zero output mismatches

Panel granularity does not repair the result. The measured copy/Q8
composition is essentially serialized and adds about 9 ms to the expert
stage. This invalidates the optimistic assumption that the 7.9 ms
all-gather can hide under BroadWave gate/up merely because copy engines are
independent. On this workload the PCIe/memory path contends with the Q8
mainloop. Four-way feature replication into expert owners is closed as the
primary RelayWave boundary. The exact continuation primitive remains useful
for a lower-fanout placement.

Evidence:

- `relay-kpanel2-exact-v1-service-p2048-ub2048.out`
- `relay-kpanel4-exact-v1-service-p2048-ub2048.out`
- `relay-kpanel2-copy-exact-v1-service-p2048-ub2048.out`
- `relay-kpanel4-copy-exact-v1-service-p2048-ub2048.out`

## Route concentration

The recorded code and Wiki route histograms contain 2,560 layer samples and
83,886,080 routed token-expert assignments each. Across layers:

- code: the hottest 16 experts carry 40.799% of routes; hottest 32 carry
  55.408%
- Wiki: the hottest 16 experts carry 40.894% of routes; hottest 32 carry
  57.471%

Hot-expert replication could therefore affect a majority of requests with
32 replicas per layer, but aggregate histograms do not reveal per-token
destination fanout or whether smaller replica batches lose the measured
four-cell pooling efficiency. It remains a placement hypothesis, not a
modeled gain.

## Accuracy constraint

The user requires no accuracy sacrifice. From this point, approximate
HFMA2 arithmetic, lower-precision intermediates not already present in the
immutable reference, reassociated reductions, relaxed comparison gates,
and all sub-Q8 weights are rejected. A placement or scheduling change must
first reproduce the immutable operation bit-for-bit at its local boundary.
Only then will the assembled path advance to KLD-pair and perplexity
validation.

## Token-level CacheRelay capture

The live service gained a default-off binary route instrument:

```text
GGML_CUDA_AW_ROUTE_DUMP=/absolute/path
```

Each record contains `(layer, home GPU, token count, route count)` followed
by the exact top-8 int32 expert IDs. The production graph and arithmetic are
unchanged. The instrument synchronizes and copies only router IDs, so its
throughput is diagnostic and is not used as a performance result.

The pp8128 capture contains four complete 40-layer x four-lane passes at
2,032 tokens/lane. The first three passes are bitwise identical and belong
to graph setup/pre-capture. The final measured evaluation is different, as
expected from llama-bench's measured prompt, and is the pass used below.

- file: `cache-relay-routes-pp8128-v1.bin`
- bytes: 41,628,160
- SHA-256:
  `0dd89bf538822716dbed425a6aadbe1b90f452de0a867859e9ca3ad4a1c618e4`
- final pass records: 160
- invalid expert IDs: zero
- live model footprint observed during the run: approximately
  10.31-10.38 GiB/GPU
- observed free memory: approximately 5.89-5.96 GiB/GPU

The exact final-pass owner-group fanout is 3.6164 nonempty logical owners per
lane token. Current AffinityWave transfers all three remote owner slots,
including zeros. Merely suppressing empty remote groups lowers this to about
2.71 remote groups/token.

Evidence:

- `cache-relay-route-dump-v1-bench-p8128-ub8128.out`
- `cache-relay-route-dump-v1-bench-p8128-ub8128.err`
- `cache-relay-routes-pp8128-v1.bin`
- `cache-relay-model.py`

## Exact owner-group CacheRelay placement

Naively moving individual hot routes is not exact. The immutable backend
forms one FP32 partial per logical owner by walking route ranks 0 through 7,
rounds that owner partial to BF16, then sums logical owners in a fixed order.
Separating a hot route from a mixed hot/cold owner group would reassociate
that reduction or require returning full FP32 route vectors.

The retained CacheRelay unit is therefore a complete token/owner group:

1. Logical owner identity never changes.
2. The four 64-expert logical groups may be permuted among physical GPUs,
   but the partial remains tagged with its immutable logical owner.
3. A remote token/owner group moves to its lane GPU only when that GPU has
   every expert weight used by that group.
4. A mixed group remains wholly on its primary GPU, including any hot
   routes, so the rank walk and single BF16 owner rounding remain unchanged.
5. Empty remote groups are represented by the same exact BF16 zero without
   transport.

Projection rows remain mathematically independent. Four-lane pooling was
already measured with zero mismatches across 16,777,216 BF16 outputs/device,
but the new cached/primary split still requires a dedicated bitwise boundary
probe before implementation.

One expert's three exact Q8_0 projections occupy 3,342,336 bytes/layer.
The first token model uses a deterministic route-frequency cache at each
`(layer, lane GPU)` and enumerates all 24 physical permutations of the four
logical owner groups. It models the exact M64/M32/M16 planner and charges:

- 0.998314 us per issued row, calibrated from the measured 16.58 ms
  pool-four service result;
- 1.273402 us per remote token/owner pair, from one F32 input plus one BF16
  owner partial at the measured 9.7 GB/s four-way PCIe rate.

The Q8 and PCIe terms are added because the real feature-panel experiment
showed that they contend/serialize on this path.

| extra experts/layer/GPU | replica GiB/GPU | cached routes | local owner groups | remote groups/token | padding | modeled Q8+PCIe |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000 | 0.000% | 25.187% | 2.706 | 97.117% | 1,245.5 ms |
| 8 | 0.996 | 17.935% | 48.691% | 1.856 | 96.756% | 1,076.0 ms |
| 16 | 1.992 | 31.062% | 61.401% | 1.396 | 96.414% | 967.2 ms |
| 24 | 2.988 | 40.315% | 70.076% | 1.082 | 96.046% | 893.7 ms |
| 32 | 3.984 | 47.489% | 76.454% | 0.852 | 95.733% | 844.4 ms |
| 36 | 4.482 | 50.612% | 79.117% | 0.755 | 95.548% | 824.2 ms |
| 40 | 4.980 | 53.261% | 81.485% | 0.670 | 95.336% | 808.6 ms |
| 44 | 5.479 | 55.565% | 83.558% | 0.595 | 95.193% | 793.3 ms |

This is the first placement mechanism in the frontier round that changes
both communication fanout and physical Q8 load placement by hundreds of
modeled milliseconds while preserving the immutable reduction semantics.
The apparent Q8 gain despite slightly worse global padding comes from
offloading hot complete-owner groups from the slowest primary GPU onto the
token-lane GPUs.

The absolute Q8 model is conservative (937.5 ms at R=0 versus the live
824.1 ms projection slice), so only relative deltas should be carried into
the end-to-end budget. At R=40 the model removes about 207 ms of its Q8 term
and 229 ms of its communication term. This composes with the retained exact
cuBLAS panel/selector mechanisms, but it is still not enough by itself for
4000 tok/s.

Next gates:

1. Replace route-frequency selection with an exact conjunction/hypergraph
   optimizer, because a group is useful only when every member is cached.
2. Allocate the cache under a global per-GPU byte budget rather than the
   same count in every layer.
3. Build a sparse complete-owner-group scatter/service/return probe at the
   captured distribution.
4. Require bitwise equality for every owner partial and final MoE output
   before any live integration.

Machine-readable evidence:

- `cache-relay-model-frequency-v1.json`
- `cache-relay-model-frequency-v1.md`

### Exact hypergraph cache solve

The fixed-R=40 cache was then solved as the actual conjunction problem.
For every `(layer, lane GPU, candidate primary logical group)`, CP-SAT uses
one expert-residency Boolean and one owner-group coverage Boolean per unique
route pattern. A group is covered if and only if all of its expert Booleans
are selected. The route-frequency placement is supplied only as a feasible
hint. Eight solver workers and a three-second per-option limit were used;
all 160 caches selected by the final physical placement were certified
`OPTIMAL`.

The integer objective weights each covered group by the measured PCIe pair
cost plus its route count times the measured issued-row cost. All 24 physical
logical-group permutations are then evaluated with the exact tile and
directional-pair model.

| R=40 selector | cached routes | local owner groups | remote groups/token | padding | Q8 model | PCIe model | combined |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frequency | 53.261% | 81.485% | 0.670 | 95.336% | 730.0 ms | 78.7 ms | 808.6 ms |
| exact hypergraph | 54.098% | 81.317% | 0.676 | 95.358% | 711.2 ms | 90.0 ms | 801.2 ms |

The solver moves slightly more route work to caches and improves physical
Q8 balance, while accepting a small communication increase. The net gain
over frequency selection is 7.4 ms in the mechanism model. More importantly,
the exact solve confirms that the approximately 35% stage reduction is not
an artifact of a naive hot-frequency assumption.

Evidence:

- `cache-relay-model-hypergraph-r40-v1.json`
- `cache-relay-model-hypergraph-r40-v1.md`

### VRAM constraint and streamed-cache pivot

The user rejected the R=40 persistent design because its 4.98 GiB/GPU
footprint would consume capacity needed for KV cache, safety headroom, and
larger models. R=40 remains useful only as a measured architectural ceiling.
It is not a deployable backend candidate. No persistent multi-GiB expert
replication will be implemented.

The retained low-memory hypothesis is StreamCacheRelay:

1. Keep the existing single exact-Q8 primary copy of every expert.
2. Before a layer/cell becomes ready, peer-copy only that cell's selected
   remote experts into a small T64 scratch cache.
3. Compute only complete immutable logical-owner groups locally, as in the
   exact CacheRelay rule above.
4. Reuse or overwrite the cache after the cell retires. Double-buffer only
   enough layer/cell windows to expose copy/compute overlap.

Because T64 is an exact byte permutation of the same Q8_0 weights, the
staging operation itself cannot change accuracy. One R=8 layer cache is
26,738,688 bytes/GPU. A double buffer is approximately 51 MiB/GPU; even a
four-cell window is approximately 102 MiB/GPU. R=16 doubles those numbers.

The first frequency model gives the optimistic benefit before weight
transfer:

- R=8: 1,245.5 -> 1,076.0 ms, 169.5 ms removed;
- R=16: 1,245.5 -> 967.2 ms, 278.4 ms removed.

Raw weight traffic per GPU/layer is approximately 26.7 MB at R=8 and
53.5 MB at R=16. At the measured 9.7 GB/s all-to-all rate, an entirely
unhidden transfer costs about 2.76 or 5.51 ms/layer respectively. Therefore
the streamed architecture is not assumed to win. Its next gate is a live
peer-copy probe with the exact T64 byte counts, source/destination pattern,
and concurrent dense/Q8 work. Persistent cache allocation and global
multi-GiB placement optimization are closed unless the user changes the
capacity constraint.

## StreamCacheRelay weight-streaming gate

The ephemeral cache probe copies the real exact-T64 byte count for every
selected expert on all four GPUs while a production-shaped SGEMM runs on
the SMs. One complete expert's gate, up, and down weights occupy 3,342,336
bytes. Every case verified the first and last byte of every destination
region and reported zero validation errors.

Balanced source traffic:

| cache | active scratch/GPU | copy | dense | concurrent | exposed copy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| R4, M2048 | 13.37 MB | 1.849 ms | 4.850 ms | 4.658 ms | -0.192 ms |
| R8, M2048 | 26.74 MB | 3.623 ms | 4.649 ms | 4.657 ms | 0.008 ms |
| R16, M2048 | 53.48 MB | 6.906 ms | 4.657 ms | 8.272 ms | 3.615 ms |
| R24, M2048 | 80.22 MB | 10.922 ms | 4.651 ms | 11.018 ms | 6.367 ms |
| R8, M8192 | 26.74 MB | 4.148 ms | 9.153 ms | 9.119 ms | -0.034 ms |
| R16, M8192 | 53.48 MB | 8.371 ms | 9.118 ms | 9.125 ms | 0.007 ms |

The negative exposed values are timing noise, not superlinear overlap.
Captured worst-source skew was also replayed:

| cache | source expert counts | dense shape | copy | dense | concurrent | exposed |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| R8 | 1, 6, 8, 17 | M2048 | 5.127 ms | 4.995 ms | 5.130 ms | 0.135 ms |
| R8 | 1, 6, 8, 17 | M8192 | 5.125 ms | 9.114 ms | 9.137 ms | 0.023 ms |
| R16 | 14, 14, 6, 30 | M2048 | 8.547 ms | 4.649 ms | 8.839 ms | 4.190 ms |
| R16 | 14, 14, 6, 30 | M8192 | 9.404 ms | 9.113 ms | 9.140 ms | 0.027 ms |

R16 is retained as the primary low-memory boundary only when its static
next-layer weights are prefetched under the live long QKV projection.
R8 is the fallback for shorter overlap corridors. R16 requires 53,477,376
bytes/GPU active or 106,954,752 bytes/GPU double-buffered, not persistent
multi-GiB replication. R24 is closed.

Evidence:

- `streamcache-weight-probe.cu`
- `run-streamcache-weight-probe.sh`
- `streamcache-r4-dense-m2048-v2.out`
- `streamcache-r8-dense-m2048-v2.out`
- `streamcache-r16-dense-m2048-v2.out`
- `streamcache-r24-dense-m2048-v2.out`
- `streamcache-r8-dense-m8192-v1.out`
- `streamcache-r16-dense-m8192-v1.out`
- `streamcache-r8-skew17-m2048-v1.out`
- `streamcache-r8-skew17-m8192-v1.out`
- `streamcache-r16-skew30-m2048-v1.out`
- `streamcache-r16-skew30-m8192-v1.out`

## StreamCacheRelay owner-partial exactness

The R16 placement manifest was replayed for the complete final route pass:
40 layers, four homes/layer, 2,032 tokens/home. The probe computes a
synthetic but route-varying eight-rank FP32 owner partial, performs exactly
one BF16 rounding per immutable logical owner, physically assembles primary,
cached, and remote groups over P2P, and then applies the production cyclic
BF16 owner sum.

The full-width result checked:

- 1,175,769 present token/owner groups;
- 293,777 primary groups;
- 428,155 cached groups;
- 453,837 remote groups;
- 2,663,383,040 owner values;
- 665,845,760 final values;
- zero assignment errors;
- zero owner mismatches;
- zero final mismatches.

The first full-width run falsely reported 551,655,901 owner mismatches and
339,808,644 final mismatches. This was a probe race: the next record reused
remote scratch before the preceding home stream finished assembly and
comparison. Adding the missing home-stream synchronization changed a
one-layer smoke from 18,955,827 owner mismatches to zero and the complete
run to zero. The failed v1 artifacts are retained so the harness correction
is auditable; they are not an architecture failure.

Evidence:

- `make-streamcache-owner-manifest.py`
- `streamcache-owner-r16-manifest-v1.bin`
- `streamcache-owner-partial-probe.cu`
- `streamcache-owner-r16-all-c256-v1.out`
- `streamcache-owner-r16-all-c2048-v1.out`
- `streamcache-owner-r16-all-c2048-v2.out`

## ScanFold proposal audit

The proposal's general principle is sound: attention, recurrence, and MoE
need not share one placement. [MoE Parallel
Folding](https://arxiv.org/abs/2504.14960) supports that principle, but its
training/H100 results do not establish the timing or exactness of this
Pascal inference design. Each ScanFold component was therefore gated
against this rig.

The live `NOTES-codex.md` remains the canonical experiment record. It is not
frozen because the user explicitly requested that every new test remain
documented. A focused snapshot is also stored in `SCANFOLD-AUDIT.md`.

### CP4 Gated DeltaNet is closed as proposed

The model has 32 value heads and 128x128 recurrent state. A rank summary
containing both `(M, S)` is exactly 4 MiB in FP32. The
[Flash Linear Attention CP
description](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/cp/README.md)
confirms this summary form and the FP32 transition chain.

The current exact recurrent kernel for one 2,032-token shard measured
5.46481 ms. That is already slower than the proposal's complete 3.5 ms/layer
gate before summary construction, a 4 MiB exchange, scan, merge, or halo
traffic. The existing expanded WY graph is 6.55% slower at target length.
More importantly, composing chunk transforms reassociates the FP32
recurrence, so it is not bitwise equivalent merely because the affine
formula is exact over real arithmetic.

This closes the proposed CP4 affine scan. A new fused recurrence algorithm
could reopen only after an isolated local kernel is at least about 3x faster
and independently passes the accuracy gate. The adapted placement keeps
the current chronological GDN corridor.

Evidence:

- `run-scanfold-gdn-floor-probe.sh`
- `scanfold-gdn-floor-t2032-v1.out`

### Exact balanced attention epoch

The proposed equal 1,016-token chunks initially looked promising. One KV
group measured 1.871, 3.465, 5.029, 6.558, 8.156, 9.714, 11.252, and
12.744 ms for chronological prefixes 1,016 through 8,128. Pairing
`0+7, 1+6, 2+5, 3+4` gave a 14.743 ms critical pair. Exact chronological
transport measured 5.044 ms total.

That equal split is not bitwise: 5,376 of 4,161,536 FP32 outputs differ,
with maximum absolute difference 3.93e-5. The cause is not approximation in
the attention formula. The smaller calls change both the final K tile and
the launcher's chosen number of parallel K partitions, which changes the
FP32 combine order.

A default-off diagnostic selector now holds the production partition count
at three:

```text
GGML_CUDA_AW_FA_PARALLEL_BLOCKS=3
```

Combining it with 256-key-aligned first endpoints gives these within-lane
splits:

| original lane | first chunk | second chunk | first global endpoint |
| ---: | ---: | ---: | ---: |
| 0 | 1,024 | 1,008 | 1,024 |
| 1 | 1,040 | 992 | 3,072 |
| 2 | 1,056 | 976 | 5,120 |
| 3 | 1,072 | 960 | 7,168 |

All four comparisons passed bitwise: 16,646,144 FP32 values checked with
zero mismatches. A lower-imbalance split below 1,024 queries failed by
2,977-3,915 values because it no longer enters the same mask-scan geometry.
A global seven-by-1,024 plus 960-tail layout also failed: 6,098 of
16,646,144 values differed, maximum absolute difference 1.01e-5, around an
old lane boundary. Both are closed despite their small errors.

The exact two-KV-head chunk times were:

| chunk | queries | prefix | time |
| ---: | ---: | ---: | ---: |
| 0 | 1,024 | 1,024 | 3.427 ms |
| 1 | 1,008 | 2,032 | 6.759 ms |
| 2 | 1,040 | 3,072 | 9.511 ms |
| 3 | 992 | 4,064 | 12.842 ms |
| 4 | 1,056 | 5,120 | 16.237 ms |
| 5 | 976 | 6,096 | 19.279 ms |
| 6 | 1,072 | 7,168 | 22.881 ms |
| 7 | 960 | 8,128 | 23.583 ms |

Exhaustive perfect matching confirms that the original extreme pairing is
the minimum-critical-time assignment:

- GPU 0: chunks 0+7, 27.010 ms, 1,984 tokens;
- GPU 1: chunks 1+6, 29.640 ms, 2,080 tokens;
- GPU 2: chunks 2+5, 28.790 ms, 2,016 tokens;
- GPU 3: chunks 3+4, 29.079 ms, 2,048 tokens.

The real variable-size P2P replay measured:

- forward hidden shuffle: 1.673831 ms;
- chronological K/V all-gather: 1.672385 ms;
- inverse hidden shuffle: 1.677956 ms;
- total communication: 5.024173 ms;
- scratch: 33,685,504 bytes/GPU;
- validation errors: zero.

The retained exact epoch is therefore 29.640 + 5.024 = 34.664 ms before the
small 2.36% worst-rank QKV/MoE token-count imbalance. It clears the
proposal's 35 ms isolated gate and predicts roughly 90 ms total removal
across ten attention layers relative to the approximately 44 ms/layer live
critical slice. It is a valid supporting mechanism, not a path to 4,000
tok/s by itself.

The first transport probe placed gathered K/V in GPU-major order. That is
not a valid causal layout. The chronological v2 corrected the offsets before
any timing was retained. The incorrect v1 artifact is preserved.

Evidence:

- `run-scanfold-fa-probe.sh`
- `run-scanfold-fa-exactness-probe.sh`
- `run-scanfold-fa-global-exactness-probe.sh`
- `run-scanfold-fa-aligned-probe.sh`
- `run-scanfold-attention-p2p-probe.sh`
- `scanfold-fa-exactness-v1.out`
- `scanfold-fa-exactness-force3-1024-base0-v1.out`
- `scanfold-fa-exactness-force3-1040-base2032-v1.out`
- `scanfold-fa-exactness-force3-1056-base4064-v1.out`
- `scanfold-fa-exactness-force3-1072-base6096-v1.out`
- `scanfold-fa-global-exactness-force3-v1.out`
- `scanfold-fa-aligned-nh2-force3-v1.out`
- `scanfold-attention-p2p-chronological-v2.out`
- `scanfold-attention-p2p-exact-unequal-v3.out`

### Same-layer MoE adaptation

The proposal's full FP32 hidden all-gather/local-expert/reduce-scatter is
not retained. The earlier exact feature-panel experiment measured
approximately 26-28 ms and exposed about 9 ms of copy/Q8 contention, far
above the proposed 3 ms communication gate. Megatron-Core's dispatcher
documents the general all-gather/local-expert/reduce-scatter structure, but
its optimized NVLS route is Hopper-specific and does not transfer to four
PCIe P100s.

The retained adaptation is R16 StreamCacheRelay plus immutable complete
owner groups. It uses about 53.5 MB/GPU active scratch, has passed the
full-width bitwise boundary, and reduces the mechanism model from 1,245.5
to 967.2 ms, a gross 278.4 ms. Its realized scheduler saving still requires
a dependency replay because weight prefetch, service pooling, and lane
barriers can overlap or contend.

The proposal's low-register three-CTA Q8 kernel is not reopened; that kernel
was already measured and closed. GuardWire BF16 or protected-column
transport is also closed because it changes arithmetic and prior BF16 wire
tests failed the accuracy gate.

### Adapted architecture retained for replay

Full ScanFold is rejected because its GDN foundation fails both timing and
bitwise gates. The retained module-specific composition is:

1. keep the exact current AffinityWave chronological GDN corridor;
2. enter the exact 256-key-aligned balanced attention epoch only at the ten
   attention layers;
3. use R16 ephemeral exact-T64 StreamCacheRelay and complete-owner-group
   placement for MoE;
4. preserve the exact dense panel/selector mechanisms already retained;
5. account for copy-engine contention and barriers in a dependency replay
   before system integration.

This composition has measured components sufficient to make approximately
3,000 tok/s plausible. It does not yet provide the roughly 1.127 seconds
needed for 4,000 tok/s from the 2,572.8 tok/s accurate BroadWave result.
Faster exact expert arithmetic remains mandatory.

## FoldWave dependency replay

The static AffinityWave dependency replay reproduces the BroadWave profile
window within 0.726935 ms:

- measured: 3,187.830587 ms;
- replayed: 3,188.557522 ms;
- relative error: 0.0228%.

The replay keeps the base critical-path scheduling order fixed across
scenarios. This prevents a changed greedy tie-break from masquerading as a
mechanism gain.

### R16 is not a 278 ms gain in the live diagonal

The earlier R16 result of 1,245.5 -> 967.2 ms was a same-layer pooling
model. It assumed four token shards at one layer could merge their rows
before exact-Q8 tiling. A live AffinityWave diagonal instead contains four
cells from four different layers, so local caches do not merge those rows.

Measured-route replay at the actual diagonal service unit gives:

| quantity | R0 | R16 | change |
| --- | ---: | ---: | ---: |
| exact-Q8 model | 866.920 ms | 852.241 ms | -14.679 ms |
| owner communication | 322.646 ms | 180.086 ms | -142.560 ms |

At the individual-cell level, R16 increases issued rows from 994,544 to
1,551,760, or 56.0%, because cached groups are split away from the primary
owner pool. Pooling the four different-layer cells on each diagonal restores
occupancy, but the resulting exact-Q8 saving is still only 14.679 ms.

The same-layer 278 ms mechanism therefore requires a per-layer rendezvous.
The CP4 GDN scan failed its timing and bitwise gates, so that rendezvous is
not composable with the exact chronological recurrence.

### Composed results

| scenario | wall | rate | interpretation |
| --- | ---: | ---: | --- |
| diagonal R16 | 3,135.860 ms | 2,591.95 tok/s | barrier-free placement |
| plus exact dense selectors | 3,064.393 ms | 2,652.40 tok/s | measured selectors |
| central exact composition | 2,971.971 ms | 2,734.89 tok/s | retained ceiling |
| short-overlap conservative | 3,231.974 ms | 2,514.87 tok/s | regression |
| same-layer pool, free barrier | 2,915.638 ms | 2,787.73 tok/s | invalid control |
| invalid pool plus attention | 2,823.216 ms | 2,878.99 tok/s | invalid control |
| all measured ceilings | 2,707.216 ms | 3,002.35 tok/s | noncomposable bound |

The central composition uses barrier-free diagonal R16, the exact dense
selectors, the full 93.490 ms measured balanced-attention ceiling, and the
long-corridor streaming exposure. It still misses:

- 3,000 tok/s by 262.637 ms;
- 3,500 tok/s by 649.685 ms;
- 4,000 tok/s by 939.971 ms.

The final row is deliberately invalid: it applies same-layer pooling without
its barrier and also credits the RelayWave panel ceiling despite the required
full hidden all-gather and measured copy/Q8 contention. Even that bound only
barely reaches 3,000 tok/s and misses 4,000 tok/s by 675.216 ms.

The prior statement that the adapted composition made 3,000 tok/s plausible
is superseded by this calibrated replay. ScanFold closes as a complete
architecture. Exact balanced attention remains a valid supporting primitive,
and R16 remains a low-memory placement boundary, but neither justifies
integration before exact expert arithmetic changes its ceiling.

Evidence:

- `foldwave-replay.py`
- `foldwave-replay-v1.json`
- `FOLDWAVE-REPLAY.md`

## ExpandWave exact FP32 weight control

The next arithmetic hypothesis was that BroadWave's native Q8 conversion and
scale multiplies were consuming enough Pascal issue bandwidth to hide a large
exact speedup. An existing N64 FP32-weight path was first measured as a quick
control:

| path | down kernel | expansion |
| --- | ---: | ---: |
| legacy exact Q8 | 6.294762 ms | 0 |
| legacy FP32 weight | 7.051438 ms | 0.296892 ms amortized |

That result was not sufficient to close the idea because the old FP32 path
changes the output tile geometry and pads M32/M16 tails as M64.

### Fair N128 implementation

The diagnostic ExpandWave kernel retains BroadWave's M64xN128 tile,
split-K2 partition, per-output FMA order, final partial add, 32 KiB shared
memory, and two-CTA/SM occupancy. Only its weight staging changes: one
64-expert down-projection set is expanded from exact T64 Q8_0 into row-major
FP32, and M64 tiles consume those values. M32/M16 tails stay on the existing
exact Q8 kernels.

Resource and static SASS counts:

| kernel | registers | shared | FFMA | FMUL | signed I2F | global loads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BroadWave Q8 | 122 | 32 KiB | 1,024 | 32 | 24 | 7 |
| ExpandWave FP32 | 123 | 32 KiB | 1,024 | 0 | 0 | 19 |

Both compile without local/stack spills and retain two CTAs/SM.

The matched four-cell coalesced edge-route result is:

| quantity | BroadWave | ExpandWave | change |
| --- | ---: | ---: | ---: |
| down | 5.164076 ms | 5.035814 ms | -0.128262 ms (-2.48%) |
| instrumented service | 16.657062 ms | 16.505006 ms | -0.152056 ms |
| FP32 expansion | 0 | 1.095088 ms | +1.095088 ms |
| extra allocation | 0 | 268,438,528 bytes/GPU | +256 MiB class |
| output mismatches | 0 | 0 | bitwise |

Including expansion, the mechanism regresses by approximately 0.967 ms.
Even if expansion were free and the full 2.48% kernel gain applied to every
operation in the 824.097 ms live Q8 slice, the ceiling would be only about
20.5 ms. This fails the 300 ms architecture gate by more than an order of
magnitude.

ExpandWave is closed. No gate/up expansion, 512 MiB double buffer, or
768 MiB three-projection cache will be built. The experiment establishes
that Q8 decoding is not BroadWave's dominant bottleneck; the 1,024 ordered
FP32 FFMAs remain after conversion disappears.

Evidence:

- `EXPANDWAVE-AUDIT.md`
- `expandwave-broadwave-control-v2-service-p2048-ub2048.out`
- `expandwave-n128-f32down-v1-service-p2048-ub2048.out`
- `expandwave-legacy-q8-v1-service-p2048-ub2048.out`
- `expandwave-legacy-f32down-v1-service-p2048-ub2048.out`
- `expandwave-sass-summary-v1.txt`

## RailWave exact split-rail engine

BroadWave's exact result is already expressed as two independent FP32
accumulator rails followed by one final add. RailWave tested whether those
rails could become separate device-scheduled CTAs:

1. rail 0 consumes K offsets 0-15 of every Q8 block;
2. rail 1 consumes offsets 16-31;
3. each rail preserves its complete original FMA sequence;
4. rail 0 writes the output and rail 1 writes an FP32 partial;
5. a separate kernel performs the original rail0 + rail1 final add.

The two rails use disjoint activation and weight values, so the split does
not duplicate FMA or Q8 traffic. It reduces each CTA to 80 registers and
16 KiB shared memory, allowing three CTAs/SM instead of BroadWave's two.
Each rail has 512 static FFMAs, retaining 1,024 per pair. The full-width
service oracle passed with zero mismatches on all GPUs.

The matched result is negative:

| quantity | BroadWave | RailWave | change |
| --- | ---: | ---: | ---: |
| instrumented service | 16.814186 ms | 19.970924 ms | +18.77% |
| gate | 5.283967 ms | 5.908104 ms | +11.81% |
| up | 5.279710 ms | 5.903179 ms | +11.81% |
| down | 5.179555 ms | 7.090537 ms | +36.89% |
| device allocation | 688,088,708 B | 822,306,436 B | +128 MiB |
| mismatches | 0 | 0 | bitwise |

The paired Nsight traces settle whether only the partial was at fault:

- BroadWave M64/N128 compute: 4.773927 ms/call;
- RailWave paired-rail compute: 4.958832 ms/call;
- gate/up final add: 0.434648 ms/call;
- down final add: 1.514674 ms/call.

RailWave compute is already 3.87% slower before the add. Duplicated CTA
descriptor, address, scale, loop, and control work costs more than the extra
occupancy saves. A counter-based direct handoff could remove partial traffic
but cannot rescue the losing compute kernel, and atomic accumulation would
not preserve the required rail order.

RailWave is closed. No flag handoff, grid sweep, or tail port is justified.

Evidence:

- `RAILWAVE-AUDIT.md`
- `railwave-v1-coalesced-service-p2048-ub2048.out`
- `railwave-v1-coalesced-trace.nsys-rep`
- `railwave-v1-coalesced-trace.sqlite`
- `broadwave-v2-coalesced-trace.nsys-rep`
- `broadwave-v2-coalesced-trace.sqlite`
- `railwave-sass-summary-v1.txt`

## HeadFold exact replacement placement

HeadFold is the first replacement placement in this campaign with a
measured, exact route through 3,000 tok/s. It keeps hidden activations in
four persistent 2032-token shards, but temporarily assigns eight complete
Gated DeltaNet heads to each GPU at recurrent layers.

The original H8/T8128 two-column recurrence took 10.35341 ms. The existing
one-column kernel improved it to 7.50614 ms. Four H8/T2032 launches improve
it to 7.25270 ms and are bitwise identical to the monolithic launch over the
complete 33,816,576-byte output/state image. Both deterministic dumps have
SHA-256
`2bd7407fef38e21b52cd4ff9a81f2a37e4e43c19811090c917711684a3a160dc`.
The first cross-process dump comparison was invalid because the stock test
harness seeds from `std::random_device`; those artifacts are retained but
not used.

The layout-faithful naive transport corridor was 10.39144 ms/layer, with
zero errors in staged, pitched, and direct peer-gather implementations.
Pipelining changes that result:

- exact FP32 hidden ring alone: 4.03777 ms;
- four exact QKV head-panel GEMMs: 9.25414 ms;
- ring plus QKV panels: 9.28147 ms, only 0.02734 ms exposed;
- segmented recurrence: 7.43165 ms in the standalone composition;
- recurrence plus pitched output redistribution: 9.67703 ms, 2.24537 ms
  exposed.

The QKV panel result matches monolithic selector 99 over 66,584,576 FP32
values. The recurrence comparison covers 33,816,576 values, and the output
redistribution covers 33,292,288 values. Every mismatch count is zero.

The calibrated bulk-synchronous replay predicts:

| scenario | wall | rate |
| --- | ---: | ---: |
| HeadFold R0 central | 2,923.824 ms | 2,779.92 tok/s |
| HeadFold ephemeral-R16 central | 2,645.432 ms | 3,072.47 tok/s |
| HeadFold ephemeral-R16 conservative | 2,744.306 ms | 2,961.77 tok/s |
| P100 FP32 expert peak bound | 2,340.915 ms | 3,472.15 tok/s |
| P100 FP16 expert peak bound | 2,090.021 ms | 3,888.96 tok/s |
| zero-cost expert-Q8 bound | 1,841.797 ms | 4,413.08 tok/s |

The central 3,500 target requires 9.66 effective TF/s in the exact expert
slice. The 4,000 target requires 24.40 effective TF/s if nothing else
changes, above even the P100 FP16 peak. Faster exact experts and a lower
non-Q8 floor are both mandatory.

Persistent R16 replication is rejected. The retained boundary is
58,261,504 bytes/GPU of QKV ring scratch plus at most 106,954,752 bytes/GPU
for double-buffered ephemeral exact-T64 weights, or 165,216,256 bytes/GPU
conservatively. No multi-GiB cache is retained.

Full analysis and evidence are in `HEADFOLD-AUDIT.md` and
`headfold-replay-v1.json`.

## TemporalQueue and bounded N128 service audit

The no-replication temporal placement proposal is fully modeled and its
canonical FP32/BF16 chain is measured bitwise exact. Compute-only expert
placement removes 261 ms of Q8 imbalance but creates enough canonical-chain
traffic to regress under serialization. Communication-aware placement saves
74.7 ms with layer barriers, then only about 39 ms after the retained
four-layer queue averages existing physical-owner skew. Arbitrary physical
expert splitting and capacity-constrained token homes are closed as primary
architectures.

The original 508-token/final-tail queue is now closed. Actual route replay
shows 30.33% to 55.84% of tokens complete only in the final publication and
the chronological ready prefix is normally zero until then. Flushing tails
per publication raises calibrated Q8 work from 937.457 ms to 1,504.156 ms.
Two chronological 4,064-token panels limit that tax to 43.404 ms.

An SM-only CP-SAT schedule for the two-panel adaptation reaches 2,638.150 ms
with two live panels, 2,631.140 ms with four, and 2,575.298 ms with unlimited
buffering. Every run proves a lower bound near 2,454.6 ms before copies and
events. GPU 2 alone has 2,411.723 ms of measured assigned work. Thus neither
deeper buffering nor the scheduler rewrite can reach 4,000 tok/s at current
arithmetic rates.

The surviving architecture keeps current owner placement and replaces the
1.24 GiB/GPU major live materializations with the measured 93.20 MiB bounded
arena and materialization-free N128 service. A two-panel cross-layer queue is
deferred until faster exact arithmetic and fixed-floor fusion make the
dependency replay predict below 2.00 s.

Full solver, exactness, memory, and target accounting are in
`TEMPORAL-QUEUE-AUDIT.md`, `temporal-queue-replay-v1.json`, and
`temporal-panel-current-*-v1.json`.

## DoubleWave exact barrier-removal experiment

DoubleWave tested a 512-thread M64xN128 CTA that divides 64 rows between two
cohorts and uses 48 KiB to double-buffer both A and fully dequantized B. It
preserves the complete BroadWave split-K2 FMA sequence while reducing the
K32 loop from two CTA barriers to one.

The kernel compiles at 106 registers/thread with zero spills and passes the
16,777,216-value-per-GPU BF16 service oracle. It is nevertheless about 20%
slower: projections take 6.32-6.37 ms instead of roughly 5.15-5.27 ms.
CTA-scaled SASS counts expose 81,920 shared-load instructions versus 49,152
for BroadWave. Both row cohorts reread B, so the 66.7% shared-load increase
dominates the barrier reduction. DoubleWave is closed.

Full evidence is in `DOUBLEWAVE-AUDIT.md`.

## Exact materialization-free N-panel service

Down projection panels from N2048 through N128 were implemented with
immediate rank-ordered FP32 reduction and the existing BF16 owner boundary.
All variants pass the full four-GPU bitwise oracle. N128 regresses down plus
reduction from 5.6815 to 8.1645 ms because M32/M16 lose output-column
parallelism. N512 retains 6.1417 ms while removing 96 MiB of route-output
storage in the isolated service.

A full-route N512 panel raises the bounded scratch design to 131,017,728
bytes/GPU, 3,200,000 bytes under the 128 MiB limit. It is the retained
low-memory panel size.

Same-layer synchronized owner copies fail the hardware gate. Copying the
exact remote BF16 panels exposes 8.991 ms with simultaneous streams and
15.791 ms with rotated single-stream sends, rather than at most 3 ms. The
PCIe switch is saturated. Same-layer row pooling is closed; N512 can only be
used inside the existing diagonal overlap until a production trace proves a
speed benefit.

Full evidence is in `NPANEL-SERVICE-AUDIT.md`.

## Complete exact bounded-memory service

The proposed gate/up/SwiGLU materialization removal has now been tested two
ways. A fused 512-thread BroadWave CTA is bitwise exact but raises
gate/up/SwiGLU from 10.7107 to 12.6207 ms. An unchanged BroadWave
output-column traversal is better: two N256 panels take 11.5828 ms, while
four N128 panels take 14.0576 ms. All checks cover 16,777,216 owner values
per GPU with zero mismatches.

The best complete composition is N256 gate/up plus N256 down/reduce. It
measures 18.4640 ms versus the 16.6480 ms control. Phase-reusing the HeadFold
ring yields a 123,087,872-byte arena, 11,129,856 bytes below 128 MiB and
about 1.126 GiB/GPU smaller than the current four live states.

This is retained as the low-memory boundary, not as a speed optimization.
The measured 1.8160 ms/service arithmetic tax must either be hidden by the
diagonal wave or paid only when the added KV capacity is required.

Full evidence is in `MATERIALIZATION-FREE-SERVICE-AUDIT.md`.

## VectorWave exact shared-load experiment

VectorWave remapped each BroadWave warp from eight rows x four strided
columns/thread to two logical half-warps with four rows x eight contiguous
columns/thread. It retained the exact two K16 FP32 accumulator rails and
compiled at the same 122 registers and 32 KiB, with no spills.

The intended SASS change occurred: static LDS instructions fell from 192 to
96 and total instructions from 1,956 to 1,734, while the 1,024 FFMAs were
unchanged. The full owner oracle passed with zero mismatches.

Performance nevertheless regressed from 16.6480 to 18.0292 ms instrumented
service. The vector B loads create the shared-bank transactions that the
four scalar conflict-free BroadWave loads express explicitly. Lower
instruction count is not lower shared-memory service demand. VectorWave and
shared-layout swizzles based on the same 128-value/warp lower bound are
closed.

Full evidence is in `VECTORWAVE-AUDIT.md`.

## Exact route-scatter family

Five alternatives to the 64-descriptor deterministic route scan were
implemented and measured. The single-pass atomic form changes 1,366 bytes
across all 16 layer-0 BF16 owner-partial dumps, because unordered allocation
changes route tile membership and a small number of rounding outcomes. It
fails the exact component gate.

Stable warp, segmented-ballot, stable-radix, and segmented eight-lane forms
all reproduce the 16 owner partials byte-for-byte. Nsight Systems shows why
they do not win. After normalizing two pre-capture waves plus one measured
wave across four GPUs, the existing pp2048 count/build/scatter path costs
only 14.352 ms on one critical GPU. Stable8 costs 24.366 ms and segment8
costs 27.486 ms. The current 64 CTAs turn the repeated route reads into
cached, massively parallel work; reducing reads starves or serializes
Pascal.

Matched pp8128 confirms the regression: 3,156.324 ms and 2,575.147 tok/s
for deterministic versus 3,194.271 ms and 2,544.556 tok/s for stable8.
Route scatter and serial-plan fusion are closed as architectural
contributors.

Full exactness, timeline normalization, and evidence paths are in
`ROUTE-SCATTER-AUDIT.md`.

## RecurFill exact recurrent-corridor experiment

RecurFill tested whether already-required dense `z` and output projections
could fill the exact HeadFold GDN corridor using high- and low-priority
nonblocking streams with no persistent weights or large cache.

The proposed resource model was incorrect. Nsight Systems and `cuobjdump`
show that cuBLAS algorithm 10 uses `maxwell_sgemm_fp16_128x64_tn` with 120
registers/thread, 128 threads, and about 13 KiB shared memory. The exact H8
GDN kernel uses 40 registers/thread and no shared memory. The kernels overlap
physically but contend heavily: GDN publications that take about 0.495 ms
alone extend as high as 1.841 ms under dense load.

A bounded exact cuBLAS selector sweep found algorithm 8 to be the best
co-scheduling choice. It uses a 64-thread `maxwell_sgemm_fp16_64x64_tn`
kernel with 124 registers/thread and about 9 KiB shared memory. In 100-repeat
confirmation it completes the combined corridor in 15.2401 ms, versus
16.2056 ms for the fastest current sequential algorithm-10 dense work plus
GDN. This is only 0.9655 ms/layer, or 28.96 ms across all 30 recurrent
layers, before adding normalization, gates, events, peer copies, slot
pressure, or downstream dependencies.

All tested selectors reproduce 46,301,184 FP32 reference values bitwise.
RecurFill is exact and memory-safe, but its favorable independent-work upper
bound is far below the 300 ms architecture gate. Production integration and
the one/two/four-slot sweep are closed unless a future low-register dense
kernel changes the measured coexistence result.

Full methods, selector table, resource evidence, and artifact paths are in
`RECURFILL-AUDIT.md`.

## HalfPipe exact BroadWave staging

HalfPipe divided BroadWave's 16-KiB expanded-B stage into two K16 halves and
assigned each half to four whole producer warps. It preserves the M64xN128
geometry, 32-KiB shared allocation, two ordered FP32 accumulator rails, and
final `acc0 + acc1` boundary without adding any VRAM.

The full-barrier `halfpipe_sync` diagnostic is bitwise exact over all
16,777,216 owner outputs. It compiles at 124 registers/thread for F32 input
and 128 for BF16, with 32 KiB shared and no spills, so it retains two
CTAs/SM. Matched traces improve the M64 median from 4.6267 to 4.5154 ms
(2.41%). The complete isolated service improves only 0.1901 ms because the
M32/M16 tails are unchanged.

Two full pp8128 runs average 3,147.205 ms and 2,582.610 tok/s, versus
3,157.830 ms and 2,573.919 tok/s for matched BroadWave. The exact
end-to-end saving is 10.626 ms (0.34%).

The warp-asynchronous `halfpipe_bar` variant uses four named PTX barriers to
separate half-free and half-filled states. It remains bitwise exact and
spill-free, but regresses critical isolated service to 17.8876 ms versus
16.5666 ms for the safe form. Barrier phase and half-CTA scheduler pressure
exceed the recovered overlap.

Retain `halfpipe_sync` default-off as a small exact optimization. Close
`halfpipe_bar` and further staging refinements as architectural paths. The
10.626 ms credit does not make the bounded-memory composition pass its
1.98-second integration gate.

Full implementation, exactness, resource, SASS, trace, and pp8128 evidence
are in `HALFPIPE-AUDIT.md`.

## Exact direct peer owner-gather

The staged owner-output path was replaced experimentally with a width-4
kernel that reads all four BF16 owner partials through peer UVA, reproduces
the original rotated FP32 accumulation and per-add BF16 boundary, and writes
the home output directly. The standalone primitive improves staged
copy-plus-sum from 4.137333 to 2.469728 ms and checks 199,753,728 FP32 values
with zero bit mismatches. An independent live pp2048 in-run oracle also
passes.

Production overlap reverses the result. A clean pp8128 HalfPipe control takes
3,143.375892 ms (2,585.755 tok/s); direct owner-gather takes 3,264.180001 ms
(2,490.059 tok/s), a 120.804109 ms regression. Matched traces show memcpy
falling by 106.367 ms but owner-sum SM work growing by 109.831 ms. Critical
GPU busy union rises 78.569 ms, idle rises 29.724 ms, and the
perfect-balance work floor rises 70.053 ms.

The mode skips exactly 266,338,304 bytes, or 254 MiB/GPU, of four-owner
receive storage at pp8128. Retain it default-off only as an exact emergency
capacity option. Panel-ready event partitioning is closed because it cannot
remove the measured work-floor regression and the entire remaining
transport opportunity is below the 300 ms architecture gate.

Full implementation, exactness, memory accounting, matched trace, and
artifact paths are in `DIRECT-OWNER-AUDIT.md`.

## Pause handoff - 2026-07-24 22:59 UTC

Experimentation is paused at the user's request. No GPU workload is active
under this worktree and the four-GPU flock is released. The latest complete
decision boundary is:

- RecurFill closed: favorable independent-work ceiling is only 28.96 ms.
- `halfpipe_sync` retained default-off: exact +10.626 ms pp8128.
- `halfpipe_bar` closed: exact but slower.
- Direct owner-gather retained only for 254 MiB/GPU capacity relief: exact
  but 120.804 ms slower at pp8128.
- The bounded N256 materialization-free service remains the preferred
  capacity design at 123,087,872 bytes/GPU, with a measured 1.8160 ms/service
  arithmetic tax.
- No tested composition predicts the required <=2.032 s for 4,000 tok/s, so
  no invasive architecture integration has been started.

The production worktree was rechecked at pause. It remains on
`affinitywave-production-20260723` at
`ae2b41d682e0c18f5c7277860dd0566244413fcb`; its pre-existing modified files
and untracked captures were not touched.

## Epoch GroupWave replay - 2026-07-25

The bounded diagonal N2048 + HalfPipe + exact dense-selector candidate
averages 2927.554595 ms, or 2776.379 tok/s, and reclaims about
0.831 GiB/GPU. The user accepted this as the approximate 1 GiB/GPU memory
target.

A measured zero-copy helper-attention upper bound removes only 98.120 ms
and cannot reach 3000 tok/s. The corrected current-placement two-panel
model likewise lands at 2926.794 ms once both dispatch and combine traffic
are included.

An exact epoch optimizer over permutations of four indivisible 64-expert
logical owner groups changes the result. It reduces modeled critical Q8
work from 825.745 to 715.004 ms without replicas or split reduction
boundaries. Composed with the already bitwise-exact balanced attention
primitive and corrected two-panel service, all epoch solves are optimal at
2558.737 ms, or 3176.567 tok/s. This clears the implementation gate with
150.596 ms margin to 3000 tok/s.

Full inputs, invalid-model warnings, exact artifact names, and the staged
implementation contract are in `EPOCH-GROUPWAVE-AUDIT.md`.

Stage 1 implementation is complete. The composed EPLB map and explicit
logical-owner remap reproduce all four layer-0 N2048 service outputs
byte-for-byte at pp512. See `epoch-groupwave-diagonal-dump-v1/` and the
updated audit for hashes.

Stage 1 alone does not accelerate the old diagonal. At pp2048 it takes
924.597690 ms versus 917.667416 ms for current placement, a 6.930274 ms
regression. No pp8128 diagonal run is warranted. This confirms the placement
must be composed with the same-layer two-panel scheduler modeled by the
optimizer.

## PairWave - 2026-07-25

The PairWave replay and all initial component gates are recorded in
`PAIRWAVE-AUDIT.md`. The central N512 PairFold estimate is 2694.407 ms, or
3016.620 tok/s, with about 1.12 GiB/GPU reclaimed. Pair-return traffic adds
only 0.010266 ms in a matched exact probe, and resident exact tile work
exceeds final dispatch latency by at least 2.2217x on all 80 panels.

The default-off manifest and one-copy loader now pass a full serialized
model load. Every GPU owns 20 layers and registers 60 T64 expert tensors;
inactive layer slices are zero-sized. Router EPLB arithmetic remains
unchanged. The next gate is a pair-local pooled service that emits four
separate canonical BF16 slots.

The pooled service is now replaced by a lane-exact 256-descriptor queue.
Pooling changed 174 layer-0 values, and all 137 affected tokens correlate
with measured M16/M32/M64 class transitions. The lane-exact path passes all
160 service boundaries and complete c512 saved logits byte-for-byte.

Tile-preserving 2xM32 and 2xM16 weight-sharing kernels are exact but slower.
At pp2048, plain PairWave is 1555.162/1554.078 ms, 2xM32 is
1593.355/1592.923 ms, and 2xM16 is 1631.850/1633.353 ms. PairWave retains
plain lane-exact arithmetic. Full evidence and resource counts are in
`PAIRWAVE-AUDIT.md`.

The corrected lane-exact replay is `pairwave-replay-v3.json`. The old
3016.620 tok/s estimate concatenated route rows from two lanes before tiling
and is invalid. With original lane tile boundaries, PairFold N512 at 25%
copy exposure is 2782.101 ms, or 2921.533 tok/s. Its zero-copy bound is
2747.179 ms, or 2958.672 tok/s. Even the zero-copy, zero-panel-tax N2048
bound is 2737.309 ms, or 2969.340 tok/s, 27.975 ms above the 3000 tok/s
target. Full scheduler integration is gated off.

The plain lane-exact pp2048 Nsight trace contains wave intervals of
1820.714, 1596.165, and 1574.159 ms. After warm-up, aggregate exact-Q8
device work is stable at 1263.042 and 1263.144 ms. The timed pass divides
into 425.419 ms M64, 226.888 ms M32, and 610.837 ms M16. M32 plus M16
therefore accounts for 66.3% of exact-Q8 device work, but both exact shared-B
bundle experiments already regress. The next retained placement study is
bounded ephemeral exact-T64 replication inside each active pair. It may move
only complete original lane tiles and must charge exact canonical-result
transport plus measured weight-copy exposure.

## RelayWave exact helper and work-floor audit

`relaywave-bound-v2.json` evaluates the suggested barrierless scheduler and
projection-only gate/up/SwiGLU helper against the immutable-lane route plan.
The helper arithmetic is exact: it returns 512-wide F32 middle rows and
leaves down projection plus canonical BF16 reduction on the original owner.

The complete measured SM workload is 9,713.358 GPU-ms. Even perfect
four-device balance is 2,428.340 ms, or 3,347.14 tok/s. This misses the
proposal's 2,250 ms pre-code gate by 178.340 ms and 3,500 tok/s by
106.054 ms before any copies or dependencies. Deleting all measured service
overhead still leaves 2,327.474 ms, 5.189 ms slower than 3,500. RelayWave as
a 3,500-tok/s architecture is closed by work conservation.

The helper itself is efficient but bounded. R4 recovers 73.626 ms, or
96.62%, of the complete 76.203 ms Q8 load-imbalance ceiling. R16 recovers
75.887 ms, or 99.58%, and uses 71,303,168 bytes/GPU of double-buffered
gate/up weights. Including N512 scratch still reclaims 1.052 GiB/GPU. Its
246.6 MB of middle traffic has a 25.420 ms ideal wire time before exposure.

This also closes full R4-R44 expert replication. It can add at most 2.578 ms
beyond the smaller R4 helper, but requires four times wider F32 result
returns or exact masked tile duplication. The masked R44 replay duplicates
20,165 of 67,377 tiles and raises Q8 critical work to 2,176.702 ms. R4
projection help is retained only for composition after a mechanism removes
at least 300 ms of aggregate SM work.

### RelayWave proof-of-concept estimate and pause

The panelized diagonal service already uses CUDA events and returns without a
device-wide synchronization. Its fixed-order defect is more specific: every
active service group is placed on all four main streams before any cell's
post graph is submitted. A meaningful RelayWave proof must submit a cell's
post work between service groups and retain enough input slots to avoid
losing the existing copy-ahead overlap.

The realistic exact pp8128 estimate is 3000-3100 tok/s, centered near
3050 tok/s. A clean upside case is 3150-3200 tok/s. The measured
2428.340 ms perfect-balance SM-work floor excludes 3500 tok/s without faster
arithmetic. The proposed proof acceptance gate is at most 2700 ms
(at least 3010 tok/s), byte-identical output, and bounded service memory.
Start without helpers; R4 is the only rational helper follow-up.

The user paused this work before implementation or GPU testing so a separate
arithmetic effort can use the cards. This inspection round made no source
changes and launched no GPU job. The coordination lock was confirmed free and
no AffinityWave benchmark, perplexity, or Nsight process was present.

## CohortRail implementation checkpoint

CohortRail is implemented behind
`GGML_CUDA_AW_Q8_ENGINE=cohortrail` and remains default-off. The device queue
builder performs stable exact-pointer grouping into P4, P3, P2, and singleton
queues. The M16 P2/P3/P4 kernels use 64-thread consumer groups, named
producer-consumer barriers, double-buffered B, and the prescribed A buffering.
Both 32- and 64-thread producer variants compile; 64 threads is retained after
the 32-thread synthetic route was 5.8% slower.

The final resource audit reports zero stack/local allocation. F32 producer-64
register/shared-memory use is P2 94/20 KiB, P3 103/28 KiB, and P4 95/32 KiB;
the BF16 variants use 87, 95, and 95 registers. P4 SASS retains 512 FFMA and
32 FADD instructions, named BAR.ARV/BAR.SYNC instructions, and no HFMA or
FP64 arithmetic. The full benchmark, llama-bench, perplexity, and backend-op
targets build successfully.

Synthetic active-four testing forms 58 P4 cohorts and no M16 fallback. All
16,777,216 checked outputs per GPU are exact. The P4 trace measures
424.616 ms versus 647.521 ms for four equivalent plain M16 tiles, or 1.525x.
Synthetic active-two testing forms 58 P2 cohorts and no fallback. All
8,388,608 checked outputs per GPU are exact. The P2 trace measures
239.120 ms versus 335.113 ms, or 1.401x. Replacing the M32 reduction with a
unioned slab reached only a 4.2% best case and regressed 1.5% in the P2 route,
so the M32 experiment was removed and the retained exact kernel restored.

The production service gate passes completely. Layer 0 and all 160 c512
service boundaries are byte-identical to the accepted PairWave dumps. The
saved logits are byte-identical to the retained file and have SHA-256
`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`.

The first final pp2048 production trace is positive but below the continuation
gates. The PairWave baseline/candidate windows are 1648.104/1596.349 ms.
Critical-GPU exact-Q8 work falls from 319.326 to 294.259 ms, or 7.85%, below
the required 10%. Real-route M16 falls from 153.186 ms to 128.058 ms including
empty P3/P4 launches, or 1.196x; excluding those launches it is 1.207x, below
the required 1.25x. On GPU 3, the candidate divides into P2 82.723 ms,
singleton 44.231 ms, empty P4 0.607 ms, and empty P3 0.497 ms over 240
projection calls.

The natural resume point is one bounded captured-route tuning pass. First test
the already-compiled 32-thread producer on the production route. If it does
not improve at least 2%, retain 64 threads. A final low-risk candidate is to
avoid PairWave's known-empty P3/P4 launches and test two-CTA/SM grids for its
sparse P2 and singleton queues. Do not start the five pp2048 plus three pp8128
paired campaign unless both the 1.25x M16 and 10% exact-Q8 continuation gates
clear. At this checkpoint the four-GPU lock is free and no GPU job is active.

## CohortRail diminishing-returns checkpoint - 2026-07-25

The user authorized continued exact arithmetic work beyond the original
continuation gates. The retained production-route result now reduces
critical-GPU exact-Q8 work from the frozen PairWave 319.326 ms to 252.286 ms,
or 20.99%. The traced pp2048 window falls from 1648.104 to 1498.138 ms, or
9.10%. M16 falls from 153.186 ms to 90.085 ms (62.148 ms P2 plus 27.937 ms
singleton), or 41.19% and 1.700x. M64 falls from 106.157 to 102.135 ms.
M32 remains the retained exact implementation at 60.066 ms.

The final retained additions in this pass are:

- M64 CursorPipe keeps per-stage input and weight cursors, uses three aligned
  `LDG.E.CI.128` Q8 packet loads, and replaces the per-work signed software
  division by shift/mask mapping for the model's power-of-two N128 group
  counts. The last change alone reduces M64 from 103.196 to 102.135 ms.
- Singleton M16 uses per-stage input and weight cursors.
- Named barriers use register IDs for buffer and subgroup selection. Generic
  P2 SASS falls from 25 to 13 `BAR.SYNC`, 30 to 16 `MEMBAR.CTA`, 70 to 31
  NOP, and 14,400 to 13,568 text bytes while preserving 512 FFMA.
- The combined singleton-cursor/register-barrier trace reduces P2 from
  64.413 to 62.224 ms and singleton from 29.800 to 27.935 ms. Exact-Q8 work
  falls from 258.865 to 253.376 ms.

Final retained resources are spill-free: generic P2 F32/BF16 use 96
registers and 20 KiB shared memory; singleton uses 95/96 registers and 8 KiB;
M64 uses 122/122 registers and 32 KiB. All have zero stack and local memory.
Active1 and active2 checks report zero mismatches on all four GPUs. A BF16
active2 run also reports zero mismatches. Earlier service-boundary and saved
logits qualification remains byte-identical, with SHA-256
`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`.

Closed experiments in this pass:

- Generic P2 stage cursors: exact but only 64.796 to 64.762 ms and one extra
  register.
- Fixed-shape P2: exact but 64.822 to 66.739 ms.
- Sequential, warp-quad, and fixed-shape M32 variants: exact but slower or
  neutral. The production M32 results were 68.854, 59.879, and 60.054 ms
  versus 59.980 ms retained.
- Singleton running-output cursors: one form cut about 400 static SASS
  instructions but regressed 27.937 to 28.258 ms; the predicated form was
  neutral at 27.905 ms and added a register. Both are fully reverted.
- M64 opposite-half Q8 prefetch: exact for F32 and BF16 but raises registers
  from 122 to 126/128 and regresses 102.135 to 106.825 ms. It is fully
  reverted.

This is the arithmetic diminishing-returns boundary. The retained source has
been rebuilt successfully and the GPU lock is free. Before final
qualification, remove the still-compiled rejected selector machinery for
M32 fixed/warp-quad/rail-local, P2 fixed, and producer32, then repeat the
resource audit, edge-value harness, c512 saved-logits SHA, and final paired
system runs.

## LateFold exact scheduling boundary - 2026-07-26

The corrected immutable-tile scheduler was extended with a measured-tax
stage-ablation search. Full four-panel execution is closed: charging the
already measured exact 4x508 dense and recurrent segmentation taxes raises
the full transform to 2295.137 ms, slower than its 2278.701 ms source.

The only retained subset, LateFold, splits recurrent and attention output
work into four exact 508-token calls. Recurrent precompute, recurrent core,
attention input work, shared experts, and every natural CohortRail/Q8 tile
remain monolithic. The exact output selector tax is 0.168208 ms/cell and
0.5 us is charged per extra panel action. GroupComplete still waits for the
latest original supplier of each token and preserves route-rank and
canonical BF16 order.

| Capture | wave | total | rate |
| --- | ---: | ---: | ---: |
| train | 2262.523 ms | 2270.423 ms | 3579.950 tok/s |
| heldout | 2261.317 ms | 2269.217 ms | 3581.852 tok/s |

These are CPU dependency replays, not production measurements. Capacity
c24 and c32 improve train by only 0.018 ms over c16, so c16 is retained.
The compute busy fractions are 94.06%, 96.31%, 96.91%, and 96.18% on the
train capture, equivalent to 3.835 active GPUs. The busiest fixed resource
leaves 69.807 ms of ordering slack; perfect work redistribution leaves only
93.571 ms. Scheduling is now near saturation.

The transformed memory timeline includes input panels, exact output tiles,
gate/up middles, canonical BF16 partials, receive panels, and local late
activations. With a separate 16 MiB descriptor/event reserve, train needs
140.432 MiB/GPU and heldout needs 156.400 MiB/GPU. The retained heldout
bound reclaims 0.934 GiB against the 1,166,661,424-byte legacy arena.

LateFold aggregate compute is 8675.807 GPU-ms. Perfect redistribution is
2168.952 ms wave, or 3733.83 tok/s after the fixed tail. A 4000 tok/s
composition must still remove at least 579.407 aggregate GPU-ms. The primary
record and artifact inventory are in
`INTERFERENCEWAVE-TILEFRONTIER-AUDIT.md`.

## LateFold arithmetic composition search - 2026-07-26

OutputFold and attention output K-continuation were tested as exact-order
placement companions. OutputFold halves packet-return traffic but is
41.491 ms slower than LateFold. The initial K-continuation placement is
21.335 ms slower. A robust 39-change joint packet/state placement improves
the continuation graph by 3.236 ms but remains 18.099 ms slower than
LateFold. Neither mechanism is retained.

The complete LateFold DAG was then replayed across exact-Q8 speed factors.
The fixed-order train path reaches 3902.21 tok/s at 1.4x, 3954.44 at 1.5x,
3999.68 at 1.6x, 4057.55 at 1.75x, and 4143.34 at 2.0x. A 10,000-move
order search makes the 1.6x train result 4001.73 tok/s, but leaves only
0.879 ms of integration margin. A practical Q8-only gate is at least 1.75x,
preferably 2.0x.

At 1.6x Q8, the critical chain shifts to 616.388 ms recurrent precompute and
448.980 ms final dense work. Combined end-to-end stage requirements are
stronger: 1.2x Q8 plus 1.20x dense stages predicts 4071.44 tok/s robust,
while 1.4x Q8 plus 1.10x dense stages predicts 4088.46 tok/s and 43.97 ms
of target margin. Dense-stage factors scale the complete calibrated tasks,
not only GEMM, and must not be applied directly to an isolated kernel
speedup.

All 32 measured-tax panelization subsets were rerun at 1.6x Q8. The existing
recurrent-late plus attention-late split remains best. A hypothetical
monolithic-throughput partial-publication interface for every dense and
recurrent stage improves only another 13.528 ms. Detailed tables, scripts,
and machine-readable artifacts are in
`INTERFERENCEWAVE-TILEFRONTIER-AUDIT.md`.
