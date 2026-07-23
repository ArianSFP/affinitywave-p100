# AffinityWave status and evidence

## Honest status

AffinityWave is a buildable private research prototype. It is not a production
backend and has not demonstrated correct end-to-end model output.

The checkpoint proves three narrower things:

1. an exact-Q8_0 cross-layer expert service can be built for four P100 GPUs;
2. deterministic BF16 owner partials can be reproduced exactly on all GPUs;
3. a benchmark-only 40-layer by 4-lane service schedule can be driven at about
   2828 projected tokens/s for pp8128.

It does not prove that a user request can be processed at that rate or that the
resulting logits are correct.

## Banked performance

Retained configuration:

| Component | Value |
|---|---|
| Expert layout | Q8_0 T64/K32 |
| Projection path | M64 split-K2, M32/M16 tails |
| Request and owner wire | BF16 |
| Owner reduction | fixed rank order |
| Token lanes | four equal contiguous quarters |
| Dense weights | replicated in benchmark path |
| GDN | chunk columns 2, 4 warps |
| Lane stagger | 1 |
| Service groups | `1111` |
| Pre-capture | two untimed passes |

Clean pp8128 confirmations:

| Run | Elapsed | Projected rate |
|---|---:|---:|
| Confirmation 1 | 2874.222 ms | 2827.9 tok/s |
| Confirmation 2 | 2874.839 ms | 2827.3 tok/s |
| Mean | 2874.531 ms | 2827.6 tok/s |

The spread is 0.617 ms, or 0.0215%. Each run intentionally returns status 1
after the timed line because output is withheld.

## Exactness evidence

The clean retained exactness rerun checked:

- 16,777,216 BF16 values per GPU;
- 67,108,864 values total;
- zero mismatches on all four GPUs;
- expected BF16 bits: 48020;
- observed BF16 bits: 48020.

The four wall-derived complete-service rates were approximately 5.128 to 5.285
effective TFLOP/s/GPU. The earlier exact retained winner had a 5.237 TFLOP/s
minimum.

This evidence covers the synthetic expert service only. It does not cover:

- dense-layer numerical behavior under the wave schedule;
- attention or GDN state corridors;
- the chunked GDN math;
- shared-expert composition in a complete model output;
- logits, perplexity, or KLD for the full wave.

## Transport evidence

An independent copy-engine probe on the same topology measured 11.96 GB/s per
GPU under full GEMM load. Packing and fixed owner reduction together cost less
than 0.8 ms in the original Phase-1 service. Native-Q8 gate, up, and down
projections dominate the service time.

## Feasibility verdict

The registered Phase-0 gate required a pessimistic pp8192 estimate no slower
than 2.05 seconds. Results:

| Model | pp8192 time | Verdict |
|---|---:|---|
| Optimistic fixed-work plus perfect expert balance | 2.1122 s | fail |
| Resource-aware at 5.5 TFLOP/s/GPU | 2.1810 s | fail |
| Resource-aware at 6.5 TFLOP/s/GPU | 2.0551 s | fail |
| Sensitivity at 7.0 TFLOP/s/GPU | 2.0056 s | first pass |

The prototype was built only because the user explicitly overrode this NO-GO
to obtain a service springboard.

## Identity and reproducibility

| Item | SHA-256 or revision |
|---|---|
| Parent | `05dcabf9a97b8a91bec5621d2b4c5ce1ce9b2ca8` |
| Checkpoint | `9d3983b89952c7fc1c6aa38fc1a7bd3182992382` |
| Tree | `acbecf68f512212b95015540e3774864bf935aed` |
| Model | `c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131` |
| `llama-bench` | `eb949e2cf086b0e6dab6fd018418f2f2ec356a1585c035ff06e93c439e8a26c8` |
| `llama-affinity-wave-bench` | `0f49ab934b6a80e5109b43dbe4276ccdf4cf4a8f96893a1aa344ee2109a093a8` |
| `libggml-cuda.so.0.15.3` | `d38b3a4f9fea749a0edfdab6ded9d88135bc3e8ec87291a2b5b1802aaf82e223` |

Build environment:

- CUDA 12.8.61;
- driver 580.159.03;
- CMake 4.2.3;
- GCC/G++ 14;
- Release build;
- `CMAKE_CUDA_ARCHITECTURES=60`;
- four Tesla P100-PCIE-16GB GPUs on one PCIe switch.

## Remaining gap

At pp8128, 3500 tok/s requires about 2322.3 ms. The banked service benchmark is
about 552.2 ms slower. Even if that gap were closed, correctness machinery and
production integration would add costs not present in the current benchmark.

The original reopening condition remains appropriate:

1. obtain token-level route traces;
2. demonstrate about 7.0 effective TFLOP/s/GPU for the complete expert service,
   including packing;
3. retain at least 10 GB/s peer transport;
4. rebuild the simulation using exact token-owner traffic;
5. only then invest in end-to-end state and output validation.

## Publication boundary

This private branch includes the immutable checkpoint and documentation. Later
default-off experimental source changes are not folded into the checkpoint.
Their measured outcomes are documented in [EXPERIMENTS.md](EXPERIMENTS.md).

No upstream pull request is intended.
