# AffinityWave OpenAI-style pipeline and production-logit completion

Date: 2026-07-23

## Scope

- Source checkpoint:
  `9d3983b89952c7fc1c6aa38fc1a7bd3182992382`.
- Worktree:
  `/home/arian/llama.cpp-q36-moe/.worktrees/affinitywave-openai-sass`.
- Build:
  `/home/arian/llama.cpp-q36-moe/.worktrees/build-p100-openai-sass`.
- Model:
  `Qwen3.6-35B-A3B-Q8_0.gguf`.
- Q8_0 weights only. No sub-Q8 experiment was used.
- The immutable AffinityWave source and its retained 2827.6 tok/s result were
  not modified.

The retained 2827.6 tok/s number is a service-only run with output withheld.
It remains the primary AffinityWave service baseline, but it is not an
end-to-end production comparator. This campaign enables the final gather,
normalization, and output head and validates actual logits.

## OpenAI source review

The reviewed OpenAI GEMM kernels are hand-scheduled Maxwell/Pascal Maxas SASS.
The most relevant kernel, `hgemm_32x64x32_NN.sass`, uses FP16 storage with
FP32 accumulation and explicitly overlaps independent load/address work with
long FFMA groups. `gen_kernels.py` expands and assembles architecture-specific
kernels; `openai_gemm.py` selects datatype/transposition variants and
autotunes shapes against cuBLAS. The benchmark list is dominated by dense
training GEMMs rather than Q8 MoE routing shapes.

Blocksparse uses fixed 32x32 sparse blocks, lookup tables, grouped update
kernels, and explicit SM60/SM61 builds. Its static block topology is not a
direct representation of token-dependent top-k expert routing. The useful
transfer was its persistent/grouped work scheduling and explicit control of
reduction order, not the block-sparse data structure itself.

No OpenAI SASS or source was copied. The implementation applies the scheduling
ideas to the existing AffinityWave CUDA kernels, avoiding a Maxas dependency
and preserving the current CUDA 12.8 build.

## Implementation

### Q8 pipeline

- Added `GGML_CUDA_AW_Q8_KERNEL=cuda|interleave`.
- For the M64 Q8 kernel, one quarter of the next B tile's Q8 dequantization
  and shared-memory stores is issued after each four current-tile K steps.
- For the M32 Q8 kernel, one quarter is issued after each two K steps.
- Next-stage A stores remain at the stage boundary to avoid a longer live
  range. M16 stays on the retained kernel.
- The retained `cuda` kernel remains the default when AffinityWave is not
  enabled.

### Production graph

- Localized lane allocations and token-relative `out_ids` while preserving
  the final suffix/output split.
- Added final lane gather so the normal result normalization and output head
  execute and return real logits.
- Corrected convolution and GDN corridor state view offsets/strides after
  lane localization.
- Added explicit CUDA stream event joins for recurrent state transfer.
- Routed live MoE work through deterministic descriptor construction. This
  replaced an atomic route scatter whose row order varied at layer 1.
- Matched the accepted expert aggregation semantics: owner partials are
  rounded to BF16 and combined by a custom kernel in the same rank order as
  the accuracy-passed NCCL result.
- Added exact graph pre-capture/reuse, persistent state snapshots, and early
  convolution/GDN corridor state transfer.
- Added a guarded monotonic causal-mask recurrence for the AffinityWave
  single-sequence production graph.
- Kept opt-in activation, corridor, route, local-partial, and service-output
  diagnostics used for first-divergence bisection.

## Differential diagnosis

The two-prefix corridor oracle showed that lane 0 and the transferred lane 1
recurrent state were stable after fixing localized cache views. Layer-boundary
hashing then found the first remaining nondeterminism at layer 1. The
attention/GDN output, attention residual, and shared expert matched; only
lane 3 owner 3's pre-reduction expert partial differed.

Replacing the atomic route-row allocation with a deterministic canonical scan
made repeated layer dumps and full-logit metrics identical. The remaining PPL
bias came from the custom sequential owner sum. Replacing it with the normal
BF16 NCCL reduction semantics moved the matched-graph result inside the
accuracy gate.

Rejected reduction variants:

| Variant | PPL delta vs matched `ub128` base | Decision |
| --- | ---: | --- |
| NCCL singleton all-reduce | -0.002294 | production default |
| NCCL reduce-to-root | +0.007289 | rejected |
| grouped four-cell all-reduce | -0.003757 | rejected |
| grouped two-cell all-reduce | +0.005849 | rejected |

## Correctness and accuracy

### Synthetic service arithmetic

| Routing | Effective TF/s | Tile mix M64/M32/M16 | Result |
| --- | ---: | --- | --- |
| uniform | 5.528385 | 256/0/0 | 0/16,777,216 mismatches |
| edge-mix | 4.339570 | 236/12/232 | 0/16,777,216 mismatches |

The edge-mix harness returns status 1 only because its generic throughput
threshold is 5.5 TF/s. Its arithmetic comparison passed exactly.

### Formal real-logit gate

The production graph uses four 128-token lanes, so the formal oracle is the
normal graph at `c512/ub128`. This controls the numerical effect of normal
ubatch partitioning.

| Metric | AffinityWave | Normal `ub128` | Result |
| --- | ---: | ---: | --- |
| PPL | 4.078325 | 4.080618 | delta -0.002294, PASS |
| KLD | 0.000880 | - | PASS |
| RMS probability delta | 0.748% | - | - |
| Same top token | 99.608% | - | - |

The result was reproduced in one synchronized diagnostic run and two
asynchronous production runs with identical reported metrics. The final run
completed with status 0 and real output enabled.

As a secondary observation, comparison with a normal `c512/ub512` file gives
PPL 4.078325 vs 4.082694, or -0.004369. This is not the formal gate because
the normal graph itself shifts by -0.002066 between `ub512` and `ub128`
(KLD 0.000718). The matched `ub128` comparison above is the required
KLD-pair methodology.

### Backend regression

- Focused CUDA model-path suite:
  2,172/2,172 passed (`ADD`, `GET_ROWS`, `MUL_MAT`, `MUL_MAT_ID`,
  `GATED_DELTA_NET`, `SSM_CONV`, and `RMS_NORM`).
- The final allowed-precision subset passed 221/221 after the production
  cleanup.
- The broader CUDA backend sweep emitted 24,321 cases with no failure or CUDA
  error before its 600-second guard expired in the large FlashAttention
  parameter matrix. This is recorded as an incomplete sweep, not as a pass.
- Release build completed for `llama-perplexity`, `llama-bench`,
  `llama-affinity-wave-bench`, and `test-backend-ops`.
- `git diff --check` passed.

## End-to-end performance

All runs used the production `env -i` stack, CPU affinity 0-11, four P100s,
the `.xsession-errors` watchdog, Q8_0 weights, and output enabled.

| Graph | Batch | Samples (tok/s) | Mean (tok/s) |
| --- | --- | --- | ---: |
| AffinityWave final binary | pp8128/ub8128 | 2518.216, 2516.122 | 2517.169 |
| Normal exact-checkpoint comparator | pp8128/ub4096 | 1378.306, 1395.181 | 1386.743 |

The normal graph cannot fit `ub8128` on this configuration; AffinityWave fits
it because lane-local allocations cap each device's token dimension.

The 2827.6 tok/s retained service-only/output-withheld result is intentionally
not mixed into this comparison.

## Production configuration

The validated settings are:

```text
GGML_CUDA_AFFINITY_WAVE=1
GGML_CUDA_AW_Q8_LAYOUT=t64k32
GGML_CUDA_AW_Q8_KERNEL=interleave
GGML_CUDA_AW_M64_SPLIT=2
GGML_CUDA_AW_WAVE_DENSE=1
GGML_CUDA_AW_WAVE_TOKEN_SPLIT=1
GGML_CUDA_AW_WAVE_OUTPUT=1
GGML_CUDA_AW_WIRE=f32
GGML_CUDA_AW_PARTIAL=bf16
GGML_CUDA_AW_NCCL_SUM=0
GGML_CUDA_AW_NCCL_ORDER_SUM=1
GGML_CUDA_AW_GROUP_CELLS=1
GGML_CUDA_AW_GDN_CHUNKED=2
GGML_CUDA_AW_PRECAPTURE=1
GGML_CUDA_AW_PRECAPTURE_OUTPUT=1
GGML_CUDA_AW_CORRIDOR_EARLY=all
GGML_CUDA_AW_CORRIDOR_STATE_SPLIT=1
```

Debug synchronization and diagnostic dumps are disabled by default.
