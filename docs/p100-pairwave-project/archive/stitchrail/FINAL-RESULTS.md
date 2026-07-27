# Exact P100 diagonal GEMM program: final result

Date: 2026-07-26

## Outcome

The cleaned default-off P100 exact path passes the complete diagonal-prefill
accuracy, resource, and paired-performance campaign. The retained production
switch is `GGML_CUDA_AW_P100_EXACT=1`; CohortRail remains the exact-Q8 engine.

The final clean binary is:

- `libggml-cuda.so` SHA-256:
  `44d0db6d838953a57fe5dc18d9045f1d3274f933c94d7e5494816a0d254403b7`;
- tracked source-diff SHA-256:
  `82d5d9826e31e7f238b9d33fca0483ac08094121d66df38f86e15cfb864c624f`;
- detached source base:
  `ae2b41d682e0c18f5c7277860dd0566244413fcb`;
- target:
  CUDA 12.8, cuBLAS 12.8.3, `sm_60`, four Tesla P100 PCIe 16 GB GPUs.

No commit, push, or PR was made.

## Retained work

The diagonal N2048 production path retains:

- a stable parallel 64-descriptor planner and direct singleton CohortRail
  dispatch, with empty diagonal cohort launches and redundant panel2048
  descriptor work removed;
- W4 exact owner reduction and W4 canonical ordered owner sum, with the W2
  kernels retained as controls;
- one graph-local F16 activation materialization per qualified fanout;
- an aligned P100 F32-to-F16 converter using 128-bit loads and packed stores,
  with the original scalar conversion for tails;
- the exact GDN precomputed-decay producer without changing recurrence order;
- exact dense cuBLAS selectors pinned to the complete qualified signature.

The final dense selector table is:

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

Every selector additionally requires the qualified CUDA/cuBLAS versions,
compute capability, dtypes, dimensions, strides, layout, alignment, and
stream context. A mismatch falls back to the existing implementation.

PairWave-specific work is preserved separately: the parallel planner,
variable-count rank-ordered reducer, and W2/W4 canonical peer sums passed the
same 640 service-boundary and saved-logits oracle. The original PairWave
kernels remain available as fallbacks.

## Exactness

The final `stitchrail-final-prefill-v2` campaign compared the umbrella OFF
and ON from scratch:

- 640 OFF and 640 ON c512 service-boundary files were emitted;
- relative layouts were identical;
- all 640 corresponding files matched byte-for-byte;
- OFF and ON saved logits matched byte-for-byte;
- both saved-logits SHA-256 values were
  `47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`.

An independent post-campaign comparison also covered the 1,920 route and
partial diagnostics outside the formal gate. All 2,560 files and
2,015,313,920 bytes in each arm have identical layouts and contents.

## Paired performance

The campaign randomized arm order with seed `exact-p100-20260726`.

| Prompt | Pairs | OFF median | ON median | Paired median saved | Paired wall gain | Bootstrap 95% lower |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 5 | 849.235 ms | 816.850 ms | 32.706 ms | 3.851% | 3.655% |
| 8128 | 3 | 2857.911 ms | 2790.956 ms | 67.309 ms | 2.355% | 2.292% |

All eight individual pairs favored the retained path. At pp8128, median
throughput is 2912.264 tok/s ON versus 2844.036 tok/s OFF. The paired median
throughput gain is 2.412%.

Relative to the frozen pre-program three-run mean of 2867.465 ms and
2834.559 tok/s, the final pp8128 median saves 76.509 ms and improves
throughput by 2.741%. Relative to the earlier approximate 2750 tok/s
production reference, the measured final median is 5.900% faster.

## Static audit

All 28 retained target kernels have zero stack, local memory, spill loads,
and spill stores in both linked-resource and fresh ptxas reports. Their
qualified FFMA, FADD, and barrier schedules are unchanged. The retained
targets contain no HFMA, HMMA, FP64 arithmetic or conversion, LDL, or STL.

The full imported AffinityWave translation unit still contains dormant
historical fallback kernels with nonzero stack or spills. The zero-stack
claim applies to the 28 retained production targets, not those dormant
variants.

## Closed and paused scope

- Expert M64/M32/M16 arithmetic is paused at diminishing returns.
- M512+M512 stacking was exact but flat.
- M32+M32 stacking was exact and 1.672x isolated, but its realistic
  pp8128 ceiling is below the 5 ms continuation threshold.
- The exact router GemmEx winner projected only about 1.63 ms pp8128 and was
  not integrated.
- Decode, append-MMVQ, and MTP optimization were removed from this campaign
  at the user's direction.
- Balanced attention is intentionally unimplemented and documented in
  `BALANCED-ATTENTION-PAUSED.md` for a later session.

## Evidence and release

- paired campaign:
  `p100-exact-campaigns/stitchrail-final-prefill-v2/summary.md`;
- exactness report:
  `p100-exact-campaigns/stitchrail-final-prefill-v2/exactness.tsv`;
- resource and instruction archive:
  `p100-exact-final-static-20260726/STATIC-AUDIT.md`;
- full engineering notebook:
  `NOTES-codex.md`.

Three non-failing audit caveats are retained:

- the campaign began at 01:29:56 UTC, while its campaign-specific
  coordination entry is stamped 01:31 UTC; every leg still demonstrably
  acquired the harness lock and used the watchdog;
- the campaign metadata does not embed the binary and source-diff hashes, so
  its provenance depends on the adjacent static archive, matching mtimes,
  and unchanged current hashes;
- both logits arms encountered the same nonfatal NCCL initialization failure
  and used the internal AllReduce fallback. Both completed successfully and
  remained byte-identical.

At 2026-07-26 02:05 UTC the four-GPU lock was acquirable, no llama or
Nsight process remained, and `/home/arian/.xsession-errors` was unchanged at
22,762 bytes. NVML could not initialize because the installed 580.173
library does not match the loaded 580.159.03 kernel driver. As a fallback,
device-owner inspection found only `nautilus` and `gnome-text-edit` graphics
handles and no project CUDA workload. The GPUs are released.
