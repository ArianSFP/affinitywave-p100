# StitchRail notebook

## Scope

Implement the approved exact two-diagonal StitchRail scheduler on the retained
CohortRail arithmetic base. The path remains default-off, preserves immutable
lane tile classes and canonical reduction order, and must leave at least
0.8 GiB/GPU reclaimed.

## Imported arithmetic base

Detached base is `ae2b41d682e0c18f5c7277860dd0566244413fcb`.
The tracked frontier diff was imported without committing. Its retained
CohortRail trace reports 252.286 ms critical-GPU exact-Q8 work and a
1498.138 ms pp2048 window versus the frozen PairWave 319.326 ms and
1648.104 ms baseline. Active1, F32 active2, BF16 active2, all 160 service
boundaries, and saved logits SHA-256
`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`
were already qualified in the source worktree.

The imported source still contains rejected CohortRail selectors for
producer32, fixed P2, and alternate M32 kernels. Remove these before the
arithmetic-base qualification and retain only the measured production path.

## Retained-path cleanup and build

Removed the runtime selectors and production instantiations for producer32,
fixed P2, and fixed/warp-quad/rail-local M32. The retained path now hardwires
producer64, the generic register-ID P2 kernel, the pointer-cursor singleton,
and the original exact M32 kernel. M64 retains its cursor and power-of-two
work mapping.

The isolated CUDA sm_60 Release build completed for the affinity-wave
benchmark, llama-bench, llama-perplexity, and test-backend-ops targets using
GCC 14 as nvcc's host compiler. Static resource inspection reports no local
allocation for the retained production M64 and M32 specializations; M64 uses
122 registers and 32 KiB shared memory, while M32 uses 143-144 registers and
24 KiB shared memory.

## Frozen CohortRail arithmetic base

The cleaned binary passes active1 F32, active2 F32, and active2 BF16 with zero
mismatches on every GPU. Active1 checks 4,194,304 values/GPU and retains
59/3/58 M64/M32/M16 tiles. Both active2 runs check 8,388,608 values/GPU and
retain 118/6/116 tiles.

The c512 all-layer dump contains 640 files. Every service input, route-ID row,
route-weight row, and reduced F32 output matches the frozen CohortRail
reference byte-for-byte for all 160 cells. The 126,647,308-byte saved-logits
file has SHA-256
`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`.
An independent audit additionally compared every route and partial
diagnostic: all 2,560 files and 2,015,313,920 bytes per arm match.

Three new pp2048 production traces measure:

- r1: 1495.555 ms window, 252.231 ms critical exact-Q8;
- r2: 1498.992 ms window, 253.928 ms critical exact-Q8;
- r3: 1512.829 ms window, 252.268 ms critical exact-Q8.

The means are 1502.459 ms and 252.809 ms. Maximum deviations from the means
are 0.690% for the window and 0.443% for exact-Q8. This repeat evidence
selects the retained M64 shift/cursor, register-ID P2, pointer-cursor
singleton, and original M32 path rather than a newer timestamp.

## Diagonal wall calibration

Three locked and watched pp8128 runs of the cleaned CohortRail engine on the
actual immutable-owner `panel2048` diagonal path measured:

- r1: 2867.518963 ms, 2834.506 tok/s;
- r2: 2866.422474 ms, 2835.590 tok/s;
- r3: 2868.454128 ms, 2833.582 tok/s.

The mean is 2867.465188 ms / 2834.559 tok/s and the maximum deviation is
0.0364%. The replay predicted 2866.920401 ms before this wall anchor, an
error of only 0.544788 ms.

Two excluded harness probes established configuration boundaries. A PairWave
manifest with an 8128-token service panel completed in 4739.465 ms but is not
the required panel2048 topology. The same pair-local placement with
`panel2048` exhausted graph-reserve memory because it places 128 expert
slices on two GPUs. Neither result is used by the replay. The valid diagonal
runs use four immutable 64-expert owners, `DIAGONAL_SERVICE=panel2048`, exact
dense selectors, and no PairWave loader.

## Exact replay and implementation gate

`stitchrail-replay.py` parses the last complete pp8128 route pass into every
immutable `(layer, lane, owner, expert, tile index, class, rows, compact row)`
descriptor. It models one SM stream/GPU, serialized endpoint input and
publication copies, four lane input slots, two scratch generations, the
current/next diagonal DAG, measured M64/M32/M16/P2 CTA waves, and the required
priority order:

1. completed return/sum/post;
2. pre work;
3. down work;
4. gate/up work.

Equally prioritized ready tasks use downstream critical-path rank. Gate/up
and down are separate tasks so the priority order is not hidden inside an
indivisible cell service.

The calibrated pp8128 replay is:

| scenario | wall ms | tok/s | decision |
| --- | ---: | ---: | --- |
| accepted HalfPipe whole diagonal | 2927.555 | 2776.379 | reference |
| CohortRail whole diagonal | 2867.465 | 2834.559 | measured base |
| StitchRail ready dispatcher | 3419.690 | 2376.821 | reject |
| plus M64/M32 gate/up stitching | 3342.733 | 2431.544 | reject |
| plus adjacent-lane M16 P2 | 3342.733 | 2431.544 | reject, 0 ms |

The required scheduler gate is 2700 ms. The phase-correct ready dispatcher
misses it by 719.690 ms and is also 552.225 ms slower than the qualified
CohortRail base. M64/M32 stitching saves 76.957 ms within that rejected
schedule but remains 475.268 ms slower than the base, so it is not enabled.
No compatible ready adjacent-lane P2 pair occurs; all 22,553 M16 descriptors
remain on the exact singleton path.

An exploratory indivisible-cell model reaches 2691.347 ms, but it cannot
enforce pre-before-down-before-gate/up and therefore is not an implementation
candidate. This is the source of the apparent near-gate result seen during
replay development.

The generated `stitchrail-v1.manifest.json` is explicitly non-deployable:
`runtime_implementation_enabled=false` and all three components are disabled.
Its manifest SHA-256 is
`23bf11bb9a1e8ba0508c1c939a4519d2fbdc4a619591af13f5140466a3d3a514`;
the cost SHA-256 is
`d32a3c25fd21561fb3a3cfa791c0d6c7dc8088eddceac91005695484bd5c43ca`.
The 640-file candidate/reference boundary root is
`511ec6ac3baa55a8640cba3bab04edca47a5916257112606ce016432605e4563`.

Manifest validation accounts for 34,467 M64, 10,357 M32, and 22,553 M16
tiles, enforces at most two live diagonals, and confirms zero enabled
incremental bytes. The proposed 16,646,144-byte shadow slot is below the
33,006,589-byte cap, but is not allocated. The accepted approximately
0.831 GiB/GPU reclaim therefore remains unchanged.

Per the plan's pre-code gate, no StitchRail runtime scheduler, callback,
stream, or tile-queue source changes were made. Consequently pp2048
StitchRail tracing, pp8128 promotion measurements, and paired KLD are not
applicable. The already qualified CohortRail cleanup is the fastest positive
configuration retained by this iteration.

## Exact P100 GEMM program baseline

Implementation resumed at 2026-07-25 22:57 UTC from detached HEAD
`ae2b41d682e0c18f5c7277860dd0566244413fcb`. Before the new coordination
entry, the complete tracked binary diff had SHA-256
`0f3dbda0414e811702b4c7da2ba9d6d5ccfb5ba8bcd0fbb7e9417a08bdde9bcb`.

The retained pre-change binaries are:

- `llama-bench`: `2d95cb9b5b360fd4ebb3376e719c2c7c707fca16cdb556f11efb9f7de327460d`
- `llama-affinity-wave-bench`: `5dee6f30b05ebf45fa29c0374a7e3b0984ce839e41a99acee6f6ea1ac869b6f0`
- `libggml-cuda.so`: `d8dd8f0458bb32e2012cd27fd59e22daa6648ac03335e2a4aa8f4745ac69b383`

The Q8_0 GGUF is 37,801,097,504 bytes with the previously recorded SHA-256
`c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131`.
The isolated build uses CUDA 12.8.61, sm_60, Release mode, and Unix Makefiles.
The three diagonal pp8128 wall runs above remain the production wall
comparator. A fresh CohortRail pp8128 mapped trace is required before final
component attribution because the existing complete pp8128 trace predates
CohortRail arithmetic.

### Pre-change diagonal mapped trace

`p100-exact-baseline-diagonal-trace-pp8128.nsys-rep` captures the archived
binary with the new umbrella disabled. The selected window is 2,897.116 ms;
GPU 3 is critical with 2,437.253 ms busy and 459.863 ms idle. The trace has
zero local-memory allocation in every listed custom production kernel.

Critical-GPU component times are:

- exact Q8 projection: 730.317 ms;
- FlashAttention: 439.044 ms;
- GDN: 144.706 ms;
- F32-to-F16 conversion: 48.555 ms;
- owner reduction: 37.979 ms;
- serial `aw_live_build_plan`: 19.988 ms;
- owner sum: 9.725 ms.

The exact-Q8 detail is M64 502.887 ms, M32 114.823 ms, and singleton M16
106.924 ms. Although diagonal groups cannot form cohorts, they additionally
pay 3.130 ms in cohort construction plus 5.683 ms in empty P2/P3/P4 kernel
launches. The redundant one-panel descriptor builder costs another 0.877 ms.
This directly validates the structural singleton and panel2048 launch-pruning
work, with a measured 9.690 ms critical ceiling before touching arithmetic.

The trace contains all 3,120 logical dense ranges, but this CUDA/cuBLAS build
does not expose the old target SGEMM names to the analyzer. Kernel-name
attribution still confirms 63.773 ms in the 100 M512 GEMMs, 6.375 ms in the
60 M32 GEMMs, 13.355 ms in router SGEMM, and 48.555 ms across 310 conversion
launches. Selector qualification must therefore use explicit logical ranges
and projection-output comparison rather than the analyzer's target-name
acceptance flag.

### First P100-exact implementation build

The default-off umbrella now covers the current diagonal topology only:
stable parallel 64-descriptor planning, direct CohortRail singleton dispatch,
the panel2048 descriptor bypass, W4 owner reduction, W4 canonical owner sum,
shared graph-local F16 activations, and the proven GDN pre-exp path. The W2
canonical sum remains a runtime control. The existing exact dense selectors
now also require the CUDA 12.8/cuBLAS 12.8.3 compile and runtime versions,
CC 6.0, full contiguous rows, the qualified stream, and 16-byte alignment.
This initial build also contained an unqualified Q8_0 MMVQ N=5..8
rows4/read-only-load candidate. It was removed before the final prefill-only
build and is not part of the retained result.

The combined build completed successfully for `llama-bench`,
`llama-affinity-wave-bench`, `llama-perplexity`, and `test-backend-ops`.
The candidate `libggml-cuda.so` SHA-256 is
`70348d06a5c02557edfa89d9c01fabd1683725e93313ca1ce56ba90437bdb0a8`.
The historical resource and targeted SASS reports from this build are
`p100-exact-resource-final.txt` and `p100-exact-target.sass`; they are not
the final cleaned-binary audit.

New-kernel resources are:

- parallel planner: 32 registers, 1,792 bytes shared, zero stack/local;
- W4 owner reducer: 39 registers, 64 bytes shared, zero stack/local;
- W4/W2 canonical sums: 31-32 registers, zero shared/stack/local;
- GDN pre-exp producer: 8 registers, zero shared/stack/local;
- initial, later-removed Q8_0 MMVQ N=5/6/7/8:
  103/96/111/109 registers, zero stack/local.

The targeted SASS has exactly 32 FP32 FFMAs in the W4 owner reducer and 48
FP32 FADDs across the four W2/W4 canonical-sum instantiations. These kernels
contain no HFMA, FP64 arithmetic, stack, spills, or local memory. Runtime
exactness and performance gates are not yet claimed.

### Initial c512 exactness

The first graph-reserve attempt exposed a deliberately conservative CUDA
capability check that rejected Q8_0 GEMMs consuming an explicit F16 RHS. The
meta split already produced one correct local F16 slice per GPU. A strict
`MUL_MAT`, sm_60, Q8_0, K2048, contiguous F16, F32-output, global-N>=512
exception now admits only the intended shared-conversion path.

With that correction, the device-side parallel-plan comparator completed at
c512 with zero descriptor, offset, cursor, queue-count, or stable-order
mismatches. The candidate boundary run generated all 640 retained service
oracle files plus 1,920 additional route/partial diagnostics. Every retained
file exists and is byte-identical: 0 missing, 0 mismatches. The shared
candidate/reference manifest root is
`f9901807fd87d0471f24655e4f649255ef6695d2f43abd265e2f5025ee7a227d`.
Saved-logits qualification is next.

The candidate saved-logits file has SHA-256
`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`
and is byte-identical to the retained CohortRail logits file. The initial
arithmetic/service gate is therefore complete.

### First performance screen

The combined umbrella is positive in one paired screen at each prompt:

| prompt | OFF ms | ON ms | saved ms | wall gain | OFF tok/s | ON tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 858.540 | 828.137 | 30.402 | 3.54% | 2385.45 | 2473.02 |
| 8128 | 2869.424 | 2806.323 | 63.101 | 2.20% | 2832.62 | 2896.32 |

These are screening pairs, not the final statistical campaign. A mapped ON
trace is next to attribute the retained gain and identify the next candidate.

### First candidate mapped trace

`p100-exact-candidate-trace-pp8128.nsys-rep` completed under the production
watchdog. Relative to the mapped umbrella-OFF baseline, the selected window
fell from 2,897.116 to 2,833.258 ms (-63.858 ms) and critical-GPU busy time
fell from 2,437.253 to 2,370.101 ms (-67.152 ms). The perfect-balance
device-work floor fell by 66.642 ms.

Critical-GPU deltas are:

| family | OFF ms | ON ms | delta ms |
| --- | ---: | ---: | ---: |
| F32-to-F16 conversion | 48.555 | 13.645 | -34.910 |
| owner reduction | 37.979 | 22.403 | -15.576 |
| owner sum | 9.725 | 5.650 | -4.075 |
| other AffinityWave planning/dispatch | 52.877 | 30.671 | -22.206 |
| exact Q8 projection | 730.317 | 725.582 | -4.735 |
| GDN | 144.706 | 140.819 | -3.887 |
| dequantization | 9.756 | 9.698 | -0.058 |
| FlashAttention | 439.044 | 440.273 | +1.229 |
| memcpy | 286.586 | 291.648 | +5.062 |
| other graph work | 883.117 | 891.581 | +8.464 |

The summed family deltas exceed the wall improvement because formerly
overlapped kernels become exposed as the critical path shortens. The close
agreement among the 63.858 ms mapped-window reduction, 67.152 ms
critical-GPU busy reduction, and the 63.101 ms unprofiled screen confirms
that the combined candidate removes real work rather than relying on a
favorable overlap accident.

### Vector conversion and remaining N2032 selectors

The aligned sm_60 F32-to-F16 converter uses one `LDG.E.CI.128`, four
`F2F.F16.F32` operations, and two packed 32-bit stores per hot-path thread.
It compiles to 30 registers with zero shared memory, stack, local memory, or
spills. The scalar tail uses the original conversion expression. Its first
c512 saved-logits file is byte-identical to the retained candidate and keeps
the required SHA-256.

The two previously unqualified live N2032 dense signatures were swept over
DEFAULT, algorithms 0-23, and 99-115 in three fresh 30-repeat processes.
Every accepted candidate was compared bitwise with selector 99 over the
complete output:

| M | N | K | selector 99 median ms | exact winner | winner median ms | isolated gain | projected critical saving |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 2032 | 2048 | 0.641500 | 3 | 0.572068 | 10.82% | 6.94 ms / 100 calls |
| 32 | 2032 | 2048 | 0.107221 | 7 | 0.101628 | 5.22% | 0.34 ms / 60 calls |

The M512 screen clears the 5 ms continuation threshold. Both selectors are
now provisionally pinned under the existing exact CUDA 12.8.3, sm_60, dtype,
shape, stride, current-stream, contiguous-layout, and 16-byte-alignment
signature. They still require the complete real-schedule oracle and mapped
trace.

### PairWave preservation path

The default-off umbrella now also has an isolated PairWave implementation:
the same stable 256-thread descriptor planner, exact rank-ordered metadata and
fused two-logical-group reducer, and host-resolved W4/W2 canonical peer sum.
The original PairWave kernels remain the fallback and diagonal dispatch,
events, ownership, and transport are unchanged.

The new PairWave planner uses 32 registers and 7,168 bytes shared memory; its
checker uses 40 registers. The W4/W2 reducer uses 32 registers and 136 bytes
shared memory, and the peer-sum variants use 19-37 registers. All have zero
stack/local memory/spills. The rotated W4 sum has four packed 64-bit BF16
loads followed by exactly 16 FP32 additions; W2 has four packed loads and
eight additions. The reducer has packed 128-bit route loads and FP32 FFMA,
with no HFMA or FP64 arithmetic. Runtime PairWave oracles are pending.

The final compact reducer lowers W4/W2 register use to 28/27 and static code
to 252/228 instructions. W4 has eight loop-body FFMAs and W2 has four; both
have one CTA barrier and zero local/stack memory. This replaces nvcc's
initial, invalid-range expansion of the encoded count loop without changing
any valid 0-8 route metadata semantics.

The final PairWave c512 run enabled the device serial-plan comparator and
completed all 40 layers. Its 640 service-boundary files match the retained
CohortRail oracle byte-for-byte with zero missing or different files. Its
saved logits also match the retained file byte-for-byte and retain SHA-256
`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`.
The PairWave preservation path therefore passes the full service gate.

### Router selector closure

The F32 router was probed with the production `cublasSgemm(T,N)` result as
the immutable reference. An equivalent F32 `cublasGemmEx` sweep covered
DEFAULT, 0-23, and 99-115 in three fresh 30-repeat processes. Algorithm 102
is the fastest exact candidate at a 0.324252 ms process median versus
0.364941 ms for `cublasSgemm`, an isolated 11.15% improvement. With only 40
critical-GPU router calls, this projects about 1.63 ms pp8128, below the 5 ms
small-family continuation threshold. No router production change is
retained, and the custom fused router/top-8 axis remains closed.

### Final combined c512 diagonal oracle

The combined final diagonal binary completed the device serial-plan
comparator and emitted 2,560 dump artifacts. Selecting the established 640
service boundaries yields zero missing files and zero byte mismatches against
the retained CohortRail oracle. The other 1,920 route/partial files remain
diagnostic-only.

### Final pp8128 screen before trace

The final binary's pp8128 umbrella ON/OFF screen is:

| state | wall ms | throughput tok/s |
| --- | ---: | ---: |
| ON | 2,789.724 | 2,913.550 |
| OFF | 2,855.388 | 2,846.549 |

The same-binary umbrella delta is 65.664 ms (2.30% wall, 2.35% throughput);
the exact dense selector additions are active in both arms. Relative to the
frozen three-run pre-program mean of 2,867.465 ms and 2,834.559 tok/s, the
final ON screen saves 77.741 ms and improves throughput by 2.79%. Relative to
the earlier approximately 2,750 tok/s production reference, it is 5.95%
faster. This remains a screen, not the final randomized paired campaign.

### Final mapped trace

`p100-exact-final-trace-pp8128.nsys-rep` has a 2,814.809 ms mapped window and
2,357.843 ms critical-GPU busy time. Versus the first integrated candidate,
the final vector and selector pass removes another 18.449 ms of mapped wall
and 12.258 ms of critical-GPU busy work.

The direct critical-GPU kernel deltas are:

- 80 contiguous F32-to-F16 conversions: 9.058 -> 3.862 ms, saving 5.196 ms;
- 100 M512 plus 60 M32 dense GEMMs: 69.560 -> 57.774 + 6.070 ms, saving
  5.716 ms in this capture;
- all other large kernel families are statistically flat: exact Q8 725.927
  ms, FlashAttention 440.238 ms, and GDN 141.014 ms.

Across the complete four-GPU trace, the vector converter replaces 108.627 ms
of scalar contiguous conversion with 46.034 ms, a 57.62% reduction. The real
M512/M32 schedule confirms the isolated selector result and the new M32
kernel maps all 60 calls without collision.

The final pp2048 screen is 825.799 ms / 2,480.023 tok/s ON versus 858.181
ms / 2,386.443 tok/s OFF. The umbrella saves 32.382 ms (3.77% wall, 3.92%
throughput) and remains consistent with the initial pp2048 screen.

### N512 dense selector screen

All six N512 shapes were swept over DEFAULT, algorithms 0-23, and 99-115 in
three fresh 30-repeat processes. Four exact winners clear the 2% family
threshold:

| M | N | K | selector 99 median ms | exact winner | winner median ms | isolated gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 512 | 2048 | 2.360246 | 6 | 2.176798 | 7.77% |
| 4096 | 512 | 2048 | 1.303070 | 10 | 1.097628 | 15.77% |
| 2048 | 512 | 4096 | 1.163050 | 10 | 1.091867 | 6.12% |
| 512 | 512 | 2048 | 0.192077 | 8 | 0.178684 | 6.97% |

M2048,K512 improves only 0.26% and M32 improves only 0.25%; both remain on
the existing fallback. A complete c2048 production-logits file is being
saved before the four winners are provisionally pinned.

### N512 valid-model qualification

The combined four-selector c2048 file failed byte identity despite every
synthetic selector probe being exact for its generated input. Each mapping
was therefore isolated in a complete watched c2048 run:

| mapping | complete logits SHA-256 | decision |
| --- | --- | --- |
| M8192,K2048 algorithm 6 | `c36a03873dfa73f23210874061f393fc35abf442d18abdd0600cdb28cd1278ae` | retain |
| M4096,K2048 algorithm 10 | `652eedd639a1c1aeb0a59776e23f392d4196de77e34cd67f659620f6cb1572a4` | reject |
| M2048,K4096 algorithm 10 | `7ea1533e8034108478e44690674cfb6063e94a204ae5a1801b11f00d8c203c01` | reject |
| M512,K2048 algorithm 8 | `c36a03873dfa73f23210874061f393fc35abf442d18abdd0600cdb28cd1278ae` | retain |

The baseline SHA is `c36a0387...8ae`. The rejected mappings and the
test-only selector mask were removed. The cleaned binary has SHA-256
`5556dafbf69aa94baf879cad39a95ddeb616e88be513cbadb4a14ce36c884e6b`.
Its combined 485 MiB c2048 logits file is byte-identical to the baseline and
has the same SHA.

The retained pair improves the live pp2048 screen from the prior umbrella-ON
825.799 ms / 2,480.023 tok/s to 821.080 ms / 2,494.276 tok/s, saving
4.719 ms and adding 0.57% throughput. These selectors do not enter the
N2032 pp8128 path.

### Stacked projection screen

Three fresh 30-repeat processes tested the two allowed stacked families and
compared both complete output regions bitwise:

- shared gate+up, M512+M512 -> M1024: median speedup 0.9973x; closed;
- recurrent alpha+beta, M32+M32 -> M64 algorithm 7: median 0.206361 ms
  separate versus 0.123453 ms stacked, 1.6716x and 0.082908 ms saved per
  pair.

The M32 result is provisionally interesting, but its pp8128 critical ceiling
is only about 2.5 ms before constructing the transient combined weight and
output views. Production integration remains conditional on proving that
preparation does not erase the isolated gain.

Production mapping closes the M32 stack under the program's continuation
rule. The 30 recurrent alpha/beta pairs account for 6.077 ms of balanced
critical SGEMM work, so the isolated result can save at most 2.441 ms before
preparation. A stacked M64 output interleaves each token's 32 alpha and 32
beta values. The existing downstream reshape, softplus, and sigmoid paths
require two contiguous matrices, so a graph-only integration adds one F16
weight concat and two F32 materializations per pair. Even a backend-specific
combined dequant and vectorized output split projects only 2.1-2.4 ms saved.
The exact probe is preserved, but no production machinery is added for a
family below the 5 ms continuation threshold.

### Decode and MTP scope

The user removed decode, append-MMVQ, and MTP optimization from this
campaign. The unqualified Q8_0 N=5-8 rows4/read-only-load candidate was
introduced after the frozen frontier import, so it and its helper were
removed completely. `mmvq.cu` and `vecdotq.cuh` are byte-identical to the
frontier baseline again. No decode or MTP result is claimed, and the final
campaign covers diagonal prefill only. The cleaned targets rebuild
successfully and the resulting `libggml-cuda.so` SHA-256 is
`57ebc9f4eb5810a58721148baaa55493df35d4aac39ceb918861a29b0c193c24`.

The device-side diagonal and PairWave serial-plan comparators were used to
qualify both parallel planners with zero mismatches. Their kernels, host
launches, synchronization, and `GGML_CUDA_AW_P100_EXACT_PLAN_CHECK`
plumbing were then removed before the final build. Both parallel planners
and their original serial fallback implementations remain intact.

### Final cleaned static audit

The post-cleanup binary SHA-256 is
`44d0db6d838953a57fe5dc18d9045f1d3274f933c94d7e5494816a0d254403b7`
and its tracked source-diff SHA-256 is
`82d5d9826e31e7f238b9d33fca0483ac08094121d66df38f86e15cfb864c624f`.
The complete archive is `p100-exact-final-static-20260726`.

All 28 retained target kernels have zero stack, local memory, spill loads,
and spill stores in both the linked resource table and fresh ptxas records.
Their FFMA, FADD, and barrier counts match the qualified schedules, with no
HFMA, HMMA, FP64 arithmetic or conversion, LDL, or STL. The imported
translation unit still compiles dormant historical fallback variants with
nonzero stack or spills; those are not selected by the retained production
path, so the zero-stack claim applies specifically to the 28 retained target
kernels. See `p100-exact-final-static-20260726/STATIC-AUDIT.md`.

### Final randomized campaign and release

The replacement clean-build `stitchrail-final-prefill-v2` campaign completed
at 2026-07-26 02:02:58 UTC. It regenerated both c512 service dumps and saved
logits before any timing run. Both arms contain 640 service files with
identical layouts and byte contents. Their saved logits are byte-identical
and both have the required SHA-256
`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`.

All five randomized pp2048 pairs and all three randomized pp8128 pairs favor
the retained path:

| prompt | pairs | OFF median ms | ON median ms | paired median saved | paired wall gain | bootstrap 95% lower | paired throughput gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | 5 | 849.235 | 816.850 | 32.706 ms | 3.851% | 3.655% | 4.006% |
| 8128 | 3 | 2857.911 | 2790.956 | 67.309 ms | 2.355% | 2.292% | 2.412% |

Median pp8128 throughput is 2912.264 tok/s ON versus 2844.036 tok/s OFF.
Relative to the frozen 2834.559 tok/s CohortRail mean it is 2.741% faster;
relative to the earlier approximate 2750 tok/s production reference it is
5.900% faster. The complete terminal record is `FINAL-RESULTS.md`.

Balanced attention remains paused, unimplemented, and default-off. Its
mechanism, proof artifacts, risks, staged resume plan, and rollback are
archived in `BALANCED-ATTENTION-PAUSED.md`, whose SHA-256 is
`1c1821716adca9465139cdfa7df534de5d147d3b59d62cb5c0343651a425aaf1`.
PairWave remains a separate preservation path: its planner, variable-count
ordered reducer, W2/W4 peer sums, original fallbacks, cubin symbols, 640-file
oracle, and exact logits artifact are retained. Decode, append-MMVQ, and MTP
optimization were not retained or run.

The final binary still has SHA-256
`44d0db6d838953a57fe5dc18d9045f1d3274f933c94d7e5494816a0d254403b7`.
`git diff --check` and all retained shell-script syntax checks pass. At
2026-07-26 02:05 UTC the four-GPU lock is free, no llama or Nsight process is
running, and `.xsession-errors` remains 22,762 bytes. NVML currently reports
a 580.173 userspace versus 580.159.03 kernel-driver mismatch; fallback device
inspection finds only desktop graphics handles and no project CUDA workload.
The worktree and GPUs are released.

The final evidence has three non-failing caveats. The campaign began at
01:29:56 UTC but its campaign-specific coordination entry is stamped 01:31
UTC; the harness lock and watchdog are nevertheless present on every leg.
The campaign metadata does not embed binary or source-diff hashes, so
provenance uses the adjacent static archive, matching mtimes, and unchanged
current hashes. Finally, both logits arms report the same nonfatal NCCL
initialization failure and use internal AllReduce; both complete successfully
and remain byte-identical.
