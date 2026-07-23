# AffinityWave architecture

## Problem statement

The target is exact-Q8_0 prompt processing for Qwen3.6-35B-A3B on four Tesla
P100 PCIe 16 GB GPUs. P100 has no integer tensor cores and the model combines
40 MoE layers with attention and recurrent gated-delta-net state. The
production-line baseline supplied to the Phase-0 model was about 1697 tok/s at
pp8192.

The preceding optimization ladder established several facts:

| Step | Approximate pp8192 result | Main lesson |
|---|---:|---|
| Layer-split baseline | 539 tok/s | Host expert sorting and poor parallelism dominated |
| Stable counting sort | 659 tok/s | Removing host algorithmic work mattered |
| Expert parallelism | 952 tok/s | Four-device expert ownership was necessary |
| Per-device submission workers | 1217 tok/s | Serial host submission was costly on the ordinary EP path |
| True four-way expert split | 1460-1488 tok/s | Experts had accidentally been split over only two devices |
| Pinned route buffers | 1470-1521 tok/s | Pageable route transfers were a smaller remaining cost |
| Phase-0 input baseline | 1697 tok/s | Starting point for the AffinityWave feasibility model |

The remaining production path still executes layer by layer and pays collective
and scheduling costs around every MoE layer. AffinityWave asks whether token
chunks can move through different layers concurrently while all GPUs provide a
shared expert service.

## 40-layer by 4-lane wavefront

An 8128 or 8192 token prompt is split into four contiguous token lanes. Lane
`c` has a fixed home GPU. Cell `(c, l)` represents lane `c` executing layer
`l`.

A cell may execute after:

1. the previous layer for the same lane has completed; and
2. the same layer for the previous lane has published the recurrent or KV
   state needed by this lane.

The 40 by 4 grid therefore has 43 diagonals. The three extra fill/drain
diagonals create a structural 7.5% scheduling cost before any transport or
imbalance is considered.

The checkpoint's full-wave path exists only under benchmark controls in the
meta backend. It partitions graphs into 40 cells, creates four token lanes,
submits the active cells for each diagonal, invokes the expert service, and
passes state corridors between adjacent lanes. It is not wired into normal
request execution.

## Weight placement

The proposed placement is:

- replicate the approximately 2.313 GiB dense trunk on every GPU;
- assign one primary owner to each of 256 routed experts per layer, with
  exactly 64 primaries per GPU;
- optionally replicate the 16 hottest experts per layer on every GPU,
  costing approximately 1.494 GiB/GPU;
- keep embedding and output tensors sharded.

The included placement was generated from aggregate per-expert counts. It is a
proxy, not a valid production calibration, because it lacks token-level route
co-occurrence. The retained service schedule does not use hot-expert
replication.

## Owner-routed expert service

For every active cell:

1. the home GPU produces token activations, top-8 expert IDs, and route
   weights;
2. requests are packed once for each owner needed by those routes;
3. each owner pools work from the active cells and evaluates gate, up, SwiGLU,
   and down;
4. each owner reduces its routes into one BF16 partial per token;
5. the home GPU sums owner partials in fixed owner order.

The synthetic Phase-1 shape has four active cells, 2048 tokens per cell, 256
work descriptors per GPU, and 16,384 routed rows per GPU.

## Exact Q8_0 T64/K32 layout

The original Q8_0 expert tensors are repacked at load time into a tile-major
layout:

- 64 output rows per tile;
- K staged in groups of 32;
- original Q8_0 quantization values and scales are preserved;
- no sub-Q8 format is used.

The retained service uses:

- M64 projection tiles;
- split-K2 for M64;
- M32 and M16 paths for short tails;
- BF16 request wire in the checkpointed wave benchmark;
- BF16 owner partials with fixed-order aggregation.

The isolated service check compares every produced BF16 owner partial with a
deterministic reference. It checked 16,777,216 values per GPU with zero
mismatches on all four GPUs.

## Dense and state execution

The meta backend contains benchmark-only support for:

- contiguous token-axis partitioning across four lanes;
- replicated dense weights;
- graph partitioning into pre-expert, routed expert, shared expert, and post
  regions;
- GDN recurrent-state and convolution-halo corridors;
- cumulative attention K/V corridors;
- two untimed pre-capture passes on P100;
- a diagonal scheduler with configurable lane staggering and service grouping.

The retained configuration uses equal token quarters, lane stagger 1, GDN
chunk columns 2 with 4 warps, and singleton expert groups (`1111`).

## Source map

| File | Role |
|---|---|
| `ggml/src/ggml-cuda/affinity-wave.cu` | Manifest validation, repack, native-Q8 service, checks, timing |
| `ggml/src/ggml-cuda/affinity-wave.cuh` | CUDA backend interface and live-cell descriptors |
| `examples/affinity-wave/affinity-wave-bench.cpp` | Isolated four-GPU service benchmark |
| `ggml/src/ggml-backend-meta.cpp` | Token lanes, graph partitioning, diagonal benchmark scheduler |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | Experimental chunked GDN prefill kernel |
| `ggml/src/ggml-cuda/moe-gemm.cu` | Q8 service and plan kernels used by the prototype |
| `ggml/src/ggml-cuda/mmvq.cu` | T64-compatible decode-side support |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Proc-address registration and repack integration |
| `src/llama-model.cpp` | Dense-vs-expert placement behavior for wave mode |
| `src/llama-model-loader.cpp` | Loader hooks for repacked expert tensors |
| `src/llama-graph.cpp` | Graph construction integration points |

## Configuration families

The implementation is deliberately opt-in. Important controls include:

| Control | Purpose |
|---|---|
| `GGML_CUDA_AFFINITY_WAVE=1` | Enables strict AffinityWave validation |
| `GGML_CUDA_AW_MAP=<path>` | Placement manifest |
| `GGML_CUDA_AW_WIRE=f32|bf16` | Request wire |
| `GGML_CUDA_AW_CHECK=1` | Synthetic exactness check |
| `GGML_CUDA_AW_Q8_LAYOUT=native|t64k32` | Expert-weight layout |
| `GGML_CUDA_AW_M64_SPLIT=1|2` | M64 split-K choice |
| `GGML_CUDA_AW_HOME=chunk|layer` | Benchmark ownership mode |
| `GGML_CUDA_AW_WAVE_DENSE=1` | Replicated-dense benchmark graph |
| `GGML_CUDA_AW_WAVE_TOKEN_SPLIT=1` | Four token lanes |
| `GGML_CUDA_AW_WAVE_DENSE_BENCH=service` | Output-withheld service benchmark |
| `GGML_CUDA_AW_LANE_STAGGER=1` | Retained diagonal spacing |
| `GGML_CUDA_AW_PRECAPTURE=1` | Two untimed capture passes |
| `GGML_CUDA_AW_GROUP_PATTERN=1111` | Singleton active-cell groups |
| `GGML_CUDA_AW_GDN_CHUNKED=2` | Retained GDN chunk width |
| `GGML_CUDA_AW_GDN_WARPS=4` | Retained GDN launch geometry |

Several other controls exercise rejected or incomplete paths. They are listed
in [EXPERIMENTS.md](EXPERIMENTS.md) and must not be enabled casually.

## Dependency and correctness invariants

- One throughput-bound expert sequence at a time per P100.
- State copies use CUDA events and peer DMA; consumers wait before use.
- Every owner processes its groups in deterministic order.
- Owner partials are summed in fixed GPU rank order.
- No sub-Q8 weights, expert skipping, or approximate routing.
- Unsupported hardware or invalid manifests fail loudly in explicit mode.
- Normal execution uses EP4 unless the benchmark-only wave path is requested.

## Unimplemented production work

- return usable logits from the complete 40 by 4 execution;
- validate full-wave logits with byte identity or KLD pairs;
- validate the chunked GDN kernel mathematically and with perplexity;
- append-prefill and arbitrary prompt lengths;
- normalize GDN and attention state into the ordinary decode layout;
- handle multiple sequences and the server request lifecycle;
- replace aggregate routing proxies with token-level calibration;
- demonstrate a material performance margin after correctness costs.
