# HeadFold R44 plus N256 productionization

Started: 2026-07-24 23:41 UTC

## Isolation

- Immutable reference: `ae2b41d682e0c18f5c7277860dd0566244413fcb`
- Detached source:
  `/home/arian/llama.cpp-q36-moe/.worktrees/affinitywave-frontier-20260724`
- Separate build:
  `/home/arian/llama.cpp-q36-moe/.worktrees/build-p100-affinitywave-frontier-20260724`
- Production worktree is read-only.
- No commit, push, or PR is authorized.

## Acceptance gates

- Every transport, allocation, reduction, and HalfPipe transformation after
  the R44 row-pooling boundary must be byte-identical. The complete runtime
  must satisfy the project accuracy gate: PPL delta within +/-0.003 using the
  paired KLD method.
- The first candidate is HeadFold R44 plus N256 with BroadWave.
- HalfPipe is promoted only after a paired, exact comparison against that
  composed candidate.
- No persistent expert replicas are allowed.
- Combined HeadFold ring, N256 arena, and double-buffered R44 storage must
  remain at or below 417,213,440 bytes/GPU.

## Chronological log

### 2026-07-24 23:41 UTC - resume audit

- Confirmed detached HEAD at the immutable reference.
- Confirmed the production worktree retains its pre-existing modifications
  and untracked captures.
- Confirmed all four P100s are idle and the prior GPU reservation was
  released.
- Re-read the required shared memory index, plans of record, current-round
  results, rig gotchas, and the paused HeadFold, N256, and HalfPipe audits.
- Current targeted builds (`llama-bench`, `llama-affinity-wave-bench`, and
  `test-backend-ops`) pass. The unrelated all-target build still exposes the
  pre-existing parallel `test-chat`/`mtmd.h` dependency race.

### 2026-07-24 23:51 UTC - N256 live-service component

- Added the default-off selector
  `GGML_CUDA_AW_SERVICE=legacy|n256`.
- N256 traverses gate and up as two 256-column exact-T64 panels, evaluates
  the unchanged SwiGLU expression into the full middle tensor, then traverses
  down as eight 256-column panels with immediate rank-ordered owner
  reduction.
- Gate, up, and route-output scratch now hold one panel rather than a full
  projection. BF16 partial and receive storage is allocated at its actual
  two-byte element size.
- Legacy remains the default. N256 currently rejects shared-service,
  fused-gate/up, and direct-owner-check modes because those diagnostic paths
  reuse full-width scratch.
- Targeted builds pass.
- Paired BroadWave pp2048 runs:

| service | elapsed | rate |
| --- | ---: | ---: |
| legacy | 2566.544 ms | 797.960 tok/s |
| N256 | 2535.357 ms | 807.776 tok/s |

- The layer-0 oracle compared 16 BF16 owner partials and four final FP32
  reduced outputs. Every file is byte-identical between legacy and N256.
  The paired captures are under `n256-dumps/legacy-pp2048` and
  `n256-dumps/n256-pp2048`.
- This establishes arithmetic integration and exactness. Context-sized
  middle, partial, and receive liveness is not yet the final 123,087,872-byte
  phase-reused arena, so the capacity gate remains open.

### 2026-07-25 01:07 UTC - HeadFold runtime correctness and first trace

- Added a default-off layer-synchronous HeadFold runtime with independent
  GDN and attention selectors. The exact control runs the four 2,032-token
  sequence shards through same-layer service.
- The first custom GDN attempt failed because graph partitions changed
  allocator lifetimes and because the V tensor has an 8,192-float token
  stride rather than a compact 4,096-float stride. Preserving the original
  pre-graph order and using the measured tensor strides made the custom
  recurrence indistinguishable from the scheduler control at c512.
- The concurrent pre-graph initially corrupted the three-row convolution
  halo. Device captures proved two separate faults:
  - gate and beta were materialized before allocations that alias them in the
    original graph;
  - `conv_state_update` identifies node 17's destination view, while node 18
    performs the actual copy. Stopping at `marker + 1` therefore published
    98,304 bytes of zeros on every lane boundary.
- The retained concurrent split executes node 18 before the peer handoff,
  then launches convolution and scalar work on all four GPUs. Its GDN-only,
  attention-only, and composed c512 runs all reproduce the control exactly:
  PPL 4.078325, mean KLD 0.000689, and 99.216% same-top-token agreement
  against the tensor-split reference.
- Full pp8128 exposed a padded KV-cache capacity. The attention gather now
  validates capacity greater than or equal to the 8,128-token active span
  instead of requiring equality.
- The first valid full-length BroadWave HeadFold timing is
  4,023.041 ms, or 2,020.362 tok/s. This is not a production candidate.
- A production-graph trace attributes the critical GPU as follows:

| family | measured time |
| --- | ---: |
| exact-Q8 projections | 832.877 ms |
| target SGEMM | 716.141 ms |
| memcpy | 635.363 ms |
| FlashAttention | 440.985 ms |
| other AffinityWave | 249.991 ms |
| other graph kernels | 242.121 ms |
| owner reduction and sum | 59.069 ms |
| conversion and dequantization | 58.298 ms |

- The trace window is 4,084.020 ms, critical-GPU busy union is
  3,194.065 ms, and the perfect device-work balance floor is 2,966.824 ms.
  The skeleton has not yet implemented balanced attention or selective R44
  request transport, and its full K/V plus service transfers create
  484.7 MiB of corridor traffic. The R44 composition must reduce both copied
  work and exposed layer-barrier gaps; kernel substitution alone cannot make
  this skeleton production-worthy.
- Evidence:
  - `headfold-gdn-only-kld-c512-v8-split-pre-copy-ppl-aw-p512-ub512.out`
  - `headfold-full-broadwave-v3-split-pre-bench-p8128-ub8128.out`
  - `headfold-full-broadwave-split-pre-v1.nsys-rep`
  - `headfold-full-broadwave-split-pre-v1-analysis.analysis.md`
  - `headfold-conv-link-split-v3-layer00-link0-source.bin`

### 2026-07-25 01:31 UTC - live R44 streaming gate

- Generated a 28,184-byte identity-placement manifest from
  `cache-relay-model-frequency-wide-v1.json`. It retains the existing four
  primary 64-expert groups and records 44 unique, non-local cache experts per
  layer and GPU.
- Added default-off `GGML_CUDA_AW_R44=prefetch|service` and
  `GGML_CUDA_AW_R44_MAP`. The first live stage allocates two exact-T64 cache
  slots of 147,062,784 bytes/GPU and queues all 132 expert-projection slices
  for each layer on a dedicated non-blocking stream.
- The pp512 allocation and execution smoke passed. A pp8128 timing run with
  diagnostic prefetch enabled completed in 4,003.221 ms, versus 4,023.041 ms
  for the prior HeadFold skeleton. The difference is not claimed as a speedup;
  it establishes that the added traffic does not expose a wall-time penalty.
- The production trace contains exactly 21,120 1,114,112-byte peer copies:
  40 layers x 4 destination GPUs x 44 experts x 3 projections. This is
  5,882,511,360 bytes, or 5.4785 GiB, delivered to every GPU over the pass.
- Per-layer cache-copy spans were:

| destination | minimum | median | maximum |
| ---: | ---: | ---: | ---: |
| GPU 0 | 14.194 ms | 18.963 ms | 20.237 ms |
| GPU 1 | 25.198 ms | 43.379 ms | 45.502 ms |
| GPU 2 | 14.647 ms | 16.674 ms | 22.517 ms |
| GPU 3 | 16.020 ms | 19.086 ms | 23.857 ms |

- Host enqueue timestamps make the copies appear late, but the service
  kernels are themselves delayed behind already-required dense/device work.
  Correlating each service launch to the first executed exact-Q8 kernel shows
  every cache ready in time. The minimum/median ready margin is 82.755/111.877
  ms on GPU 0, 70.940/92.140 ms on GPU 1, 58.606/79.659 ms on GPU 2, and
  46.140/64.379 ms on GPU 3. There are zero late layer/device pairs.
- The double buffer is exactly 294,125,568 bytes/GPU, matching the replay
  bound. The measured gate therefore passes: R44 weight streaming is
  completely hidden at pp8128 and can now be consumed by the exact service
  placement.
- Evidence:
  - `r44-identity-placement-v1.bin`
  - `r44-identity-placement-v1.out`
  - `headfold-r44-prefetch-smoke-v1-bench-p512-ub512.out`
  - `headfold-r44-prefetch-broadwave-v1-bench-p8128-ub8128.out`
  - `headfold-r44-prefetch-broadwave-v1.nsys-rep`
  - `headfold-r44-prefetch-broadwave-v1.analysis.md`

### 2026-07-25 01:49 UTC - live R44 pooled N256 service

- Replaced primary-only same-layer service with a 108-descriptor physical
  plan: 64 primary experts plus 44 exact-T64 streamed experts per GPU.
- Placement is computed after the real top-8 routes are available. A logical
  owner remains the original `expert_id/64`; physical execution can move to
  the token home only when the complete selected logical group is resident
  there. This preserves the original BF16 owner boundary and logical-owner
  accumulation order.
- Gate and up use two N256 panels. SwiGLU writes the unchanged row-major
  middle tensor. Down uses eight N256 panels and immediately reduces each
  panel.
- The first complete R44 service run was
  3,792.425 ms, or 2,143.219 tok/s. This established the exact composed
  runtime but still copied every 8,128 x 2,048 FP32 input to every GPU.
- R44 row pooling changes a small number of final floats relative to the
  unpooled HeadFold control because routes cross the M64/M32/M16 tile-class
  boundary. The measured c512 gate was:

| quantity | R44 pooled | HeadFold control | tensor reference |
| --- | ---: | ---: | ---: |
| PPL | 4.080746 | 4.078325 | 4.082694 |
| R44 minus control | +0.002421 | - | - |
| mean KLD vs tensor reference | 0.000970 | 0.000689 | - |
| same top token | 99.608% | 99.216% | - |

- The pooled path is inside the formal +/-0.003 PPL gate and is closer to the
  tensor reference than the HeadFold control. No lower precision or
  approximate arithmetic is used.
- Evidence:
  - `headfold-r44-service-broadwave-v1-bench-p8128-ub8128.out`
  - `headfold-r44-service-kld-c512-v1-ppl-aw-p512-ub512.out`
  - `r44-service-dumps/c512-v1`

### 2026-07-25 02:00 UTC - exact selective request transport

- Added a device-side selected-row gather. A physical GPU now copies an FP32
  token row only when at least one selected logical group is actually
  assigned there. Tiny route IDs and weights remain fully replicated.
- Layer-0 final FP32 outputs are byte-identical to the first R44 service.
- Full pp8128 improved from 3,792.425 to 3,578.856 ms:
  213.569 ms removed without changing arithmetic.
- The mapped production trace changed the critical-GPU memcpy slice from
  about 1,031 ms to 731.925 ms. Its remaining principal slices were:

| family | critical-GPU time |
| --- | ---: |
| SGEMM | 715.607 ms |
| exact-Q8 | 711.407 ms |
| other AffinityWave | 523.425 ms |
| FlashAttention | 439.695 ms |
| memcpy | 731.925 ms |

- The trace window was 3,648.335 ms and the perfect-balance floor was
  2,799.343 ms.
- Evidence:
  - `headfold-r44-selective-broadwave-v1-bench-p8128-ub8128.out`
  - `headfold-r44-selective-broadwave-v1.nsys-rep`
  - `headfold-r44-selective-broadwave-v1.analysis.md`
  - `r44-service-dumps/c512-selective-v1`

### 2026-07-25 02:06 UTC - exact logical-owner reduction fold

- Replaced four separate logical-owner thread spaces with one
  token-column thread. The thread traverses route ranks once, maintains four
  independent FP32 sums in the original route order, and writes only the
  logical partials assigned to the executing physical GPU.
- Layer-0 outputs remained byte-identical.
- pp8128 improved from 3,578.856 to 3,542.679 ms, removing another
  36.178 ms.
- Added default-off route high-water telemetry. The captured code route
  distribution reached 19,811 physical route rows and 4,963 selected token
  rows per GPU, versus conservative capacities of 65,024 routes and 8,128
  tokens.
- Evidence:
  - `headfold-r44-reducefold-broadwave-v1-bench-p8128-ub8128.out`
  - `headfold-r44-stats-broadwave-v2-bench-p8128-ub8128.err`
  - `r44-service-dumps/c512-reducefold-v1`

### 2026-07-25 02:17 UTC - two-slot R44 requirement

- Added `GGML_CUDA_AW_R44_SLOTS=1|2` and tested one slot as the lower-memory
  control.
- One slot remained exact at c512 but regressed pp8128 from about 3,543 ms to
  4,383.695 ms, or 1,854.144 tok/s.
- The second slot is not merely copy double buffering. It lets the next
  layer's copies begin while the prior layer's service remains queued. One
  slot waits for the prior service and exposes about 21 ms per layer.
- Two slots remain the default for R44. The one-slot form is closed unless a
  future scheduler removes that dependency.
- Evidence:
  - `headfold-r44-oneslot-broadwave-v1-bench-p8128-ub8128.out`
  - `r44-service-dumps/c512-oneslot-v1`

### 2026-07-25 02:28 UTC - route-capacity allocator experiments

- A host-synchronized exact high-water allocator first copied per-device
  route counts to the CPU before allocating gate, up, middle, and down
  storage. It was exact but regressed to 3,685.982 ms.
- Counting all four GPUs before synchronization did not help; the result was
  3,710.118 ms. The device-to-host scheduling fence, not count-kernel
  placement, is the bottleneck.
- Replaced the synchronized allocator with demand-paged managed storage for
  the four route-sized local-only arrays. The worst-case virtual capacity is
  retained, while only live route prefixes become resident.
- The managed path is byte-identical and runs at 3,543.957 ms, statistically
  unchanged from the 3,543.646 ms prior mean.
- A synchronization-only diagnostic measured an eager reservation of
  82-84 MiB/GPU and a process-relative high-water of 438-440 MiB/GPU. The
  latter also includes graph-pool growth after the first service baseline,
  so it is not an isolated arena size. It does prove that the implementation
  does not materialize the multi-gigabyte conservative route capacity.
- Evidence:
  - `headfold-r44-highwater-broadwave-v1-bench-p8128-ub8128.out`
  - `headfold-r44-highwater-parallel-broadwave-v1-bench-p8128-ub8128.out`
  - `headfold-r44-managed-broadwave-v1-bench-p8128-ub8128.out`
  - `headfold-r44-managed-memory-v1-bench-p8128-ub8128.err`
  - `r44-service-dumps/c512-managed-v1`

### 2026-07-25 03:05 UTC - retained and rejected phase reuse

- The down route-output panel now aliases the gate panel after all gate/up
  work has completed. This is local to one GPU and preserves every computed
  value.
- Owner partials use a fixed seven-token-shard index bound instead of four
  complete logical-owner planes. For a physical GPU, at most four logical
  groups for its 2,032 home tokens plus one logical group for each of the
  other 6,096 tokens can be assigned there. The exact BF16 panel allocation
  is therefore 7,282,688 bytes/GPU instead of 16,646,144 bytes/GPU.
- The compact owner panel must remain ordinary device memory. Aliasing it
  into the managed up panel is byte-identical but makes peer consumers
  migrate unified-memory pages among P100s; pp8128 regressed to
  4,935.417 ms.
- Deterministic activation compaction was also byte-identical, but even with
  the owner panel restored to device memory it regressed to 3,949.918 ms.
  The exact-Q8 activation access loses enough locality/TLB efficiency to
  cost 410.605 ms. Compact activation storage is closed until a new tiled
  mainloop changes the access pattern.
- The retained route-output alias plus compact device owner panel runs at
  3,539.313 ms in its first full-length check. Its eager reservation is
  74-76 MiB/GPU and its synchronization-only process-relative high-water is
  430-432 MiB/GPU.
- Evidence:
  - `headfold-r44-phasealias-nocompact-broadwave-v1-bench-p8128-ub8128.out`
  - `headfold-r44-compact-devicepartial-broadwave-v1-bench-p8128-ub8128.out`
  - `headfold-r44-routealias-broadwave-v1-bench-p8128-ub8128.out`
  - `headfold-r44-routealias-memory-v1-bench-p8128-ub8128.err`
  - `r44-service-dumps/c512-production-broadwave-v1`

### 2026-07-25 03:16 UTC - production qualification

- Final BroadWave and HalfPipe layer-0 outputs are byte-identical across all
  four 512-token FP32 dumps.
- Both final binaries independently pass the paired c512 accuracy gate:

| quantity | BroadWave | HalfPipe |
| --- | ---: | ---: |
| PPL | 4.080746 | 4.080746 |
| reference PPL | 4.082694 | 4.082694 |
| PPL delta | -0.001948 | -0.001948 |
| mean KLD | 0.000970 | 0.000970 |
| same top token | 99.608% | 99.608% |

- Interleaved pp8128 production samples:

| engine | samples (ms) | mean | rate |
| --- | --- | ---: | ---: |
| BroadWave | 3,539.313; 3,547.284; 3,544.068 | 3,543.555 ms | 2,293.742 tok/s |
| HalfPipe | 3,536.868; 3,538.551; 3,530.956 | 3,535.459 ms | 2,298.995 tok/s |

- HalfPipe saves 8.096 ms and adds 5.253 tok/s in the composed runtime. It is
  retained as an exact default-off selector, not promoted over BroadWave as a
  new architectural baseline.
- Targeted builds for `llama-bench`, `llama-perplexity`, and
  `test-backend-ops` pass. The single-GPU backend correctness sweep was
  intentionally stopped after a broad set of FlashAttention cases passed
  because the unfiltered matrix covers unrelated types and shapes; the
  full-model byte oracle and paired KLD runs are the acceptance evidence.
- Evidence:
  - `headfold-r44-production-broadwave-kld-v1-ppl-aw-p512-ub512.out`
  - `headfold-r44-production-halfpipe-kld-v1-ppl-aw-p512-ub512.out`
  - `headfold-r44-production-broadwave-v2-bench-p8128-ub8128.out`
  - `headfold-r44-production-broadwave-v3-bench-p8128-ub8128.out`
  - `headfold-r44-production-halfpipe-v1-bench-p8128-ub8128.out`
  - `headfold-r44-production-halfpipe-v2-bench-p8128-ub8128.out`
  - `headfold-r44-production-halfpipe-v3-bench-p8128-ub8128.out`
  - `r44-service-dumps/c512-production-halfpipe-v1`

## Banked status

The implementation is production-ready as an isolated, default-off
experimental path:

```text
GGML_CUDA_AW_HEADFOLD=1
GGML_CUDA_AW_HEADFOLD_SPLIT_PRE=1
GGML_CUDA_AW_R44=service
GGML_CUDA_AW_R44_MAP=/absolute/path/to/r44-identity-placement-v1.bin
GGML_CUDA_AW_R44_SLOTS=2
GGML_CUDA_AW_SERVICE=n256
GGML_CUDA_AW_Q8_ENGINE=broadwave
```

Use `GGML_CUDA_AW_Q8_ENGINE=halfpipe_sync` for the separately qualified
HalfPipe variant.

This path must not replace the current diagonal production default yet. Its
best measured rate is 2,298.995 tok/s, below the retained diagonal
BroadWave production rate of about 2,573.9 tok/s. The scheduler still uses
unbalanced attention and layer-wide HeadFold barriers. The strict
417,213,440-byte combined HeadFold/R44/service bound also remains open:
two-slot R44 alone is 294,125,568 bytes/GPU, while the current live service
has not yet been phase-reused with all scheduler scratch. These are explicit
continuation items, not hidden acceptance claims.

## Final isolation audit

- Production worktree:
  `/home/arian/llama.cpp-q36-moe/.worktrees/affinitywave-openai-sass`
- Production branch: `affinitywave-production-20260723`
- Production HEAD: `ae2b41d682e0c18f5c7277860dd0566244413fcb`
- Its pre-existing modified files and untracked captures remain present and
  were not edited, built, cleaned, or deleted by this campaign.
- Experimental source remains detached at the same immutable commit.
- No commit, branch update, push, or PR was made.
