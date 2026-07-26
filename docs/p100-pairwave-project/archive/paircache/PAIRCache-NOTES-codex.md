# PairCache notebook

## Scope

Implement the approved default-off exact predictive gate/up prefetch plan for
PairWave. Prediction is a cache hint only. True routes select work, a missing
or late cache entry remains on the canonical owner, immutable lane tile
classes are preserved, and the canonical down/reduction path remains exact.

## Initial coordination and frozen-base status

The staging worktree is detached at
`ae2b41d682e0c18f5c7277860dd0566244413fcb`. At notebook start, no
arithmetic diff had been imported. The repeat-qualified production
comparator was the cleaned CohortRail diagonal `panel2048` result:
2867.465188 ms and 2834.559 tok/s over three pp8128 runs, with all 160
service boundaries exact and saved-logits SHA-256
`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`.

At that point, the active P100-exact session had newer positive screens but
had not published a terminal repeated pp8128 result or released its source
for import. Source import and GPU work were deferred until the release
recorded below.

## Pre-implementation audit

The existing 41,628,160-byte route capture contains four passes of one
synthetic pp8128 prompt. Three are identical warmups and the fourth is the
measured pass. The previously quoted static R16 hit rate of 76.4789% and
oracle rate of 76.8816% therefore are not held-out code-domain evidence.

PairFold is replay-only. There is no `GGML_CUDA_AW_PAIRFOLD` runtime flag or
C/CUDA scheduler. PairWave has an exact two-cell service primitive and
one-copy loader, but production PairFold scheduling would itself be new work.
The approved 32-window, 16/8/8 replay gate must run before either scheduler or
PairCache CUDA integration.

## Replay topology

PairCache uses the truly inactive physical pair, not the earlier RelayWave
two-owner rebalance. For a canonical source GPU, the helper is `source ^ 2`
(`0 <-> 2`, `1 <-> 3`). RelayWave's published 75.887 ms R16 ceiling therefore
does not bound PairCache: a zero-new-copy one-window screen reduces the
PairFold N512 25%-copy replay from 2782.101 ms to about 2419.592 ms when the
two extra SM resources are credited. That is only an arithmetic feasibility
screen. The decision replay must charge unique F32 input rows, every F32
512-wide middle row, double-bank gate/up weight copies, endpoint
serialization, CTA tail waves, and native PairFold work displaced from the
helper GPUs.

The runtime pre-code gate remains both at least 72 ms mean improvement over
R0 and at most 2637.333 ms conservative full-replay wall, with a positive
untouched test split. The 2709.333 ms / 3000 tok/s threshold is a later
promotion boundary, not permission to start CUDA integration.

The aggregate displacement-aware SM-work floor is 2428.340 ms. Provisional
P100-exact work removal would lower it to 2361.698 ms if it transfers, but
that credit is not used before qualification. On the existing one-window
routes, an oracle inactive-pair R16 assignment can move 1308.371 GPU-ms,
34,944 immutable tiles, and 1,996,074 routed rows. Its new traffic is the
limiting uncertainty: 2.852 GB of gate/up weights, 4.088 GB of F32 middles,
and at least 5.242 GB of F32 inputs even with perfect within-panel
unique-token packing. This is roughly 627-629 ms of raw wire per helper
before existing PairFold traffic. It can pass only if the endpoint replay
demonstrates enough overlap; fully exposing that traffic erases the
arithmetic gain.

Bulk P100 copy-engine measurements are 11.96 GB/s under GEMM and
12.51-12.52 GB/s for full-duplex/permutation traffic. The older 9.7 GB/s
PairWave constant is conservative for large coalesced DMA, but it is
optimistic for the fragmented cache-weight stream. Measured R16 whole-expert
prefetch achieves only 5.687-7.744 GB/s depending on source skew and shape.
The replay must therefore keep 9.7 GB/s for coalesced input/middle/native
slabs, use the measured R16 envelope for weight banks, and mark DMA command
startup uncalibrated rather than substituting the unrelated 1.15-1.23 us SM
peer-load latency.

As a validator oracle only, ranking the final legacy route pass by routed
rows selects 1,996,718 of 2,600,960 rows (76.7685%), 640,311 unique input
vectors, and 34,897 immutable tiles. These imply 4,089,278,464 middle bytes
and 5,245,427,712 perfectly deduplicated input bytes. The close agreement
with the earlier independent 1,996,074-row/34,944-tile calculation is a
useful parser check, but this same-window ranking is not held-out evidence
and is not used for the implementation gate.

## Runtime seam audit

The reusable CUDA pieces are the PairWave placement/catalog, lane-exact work
descriptors, route pack/count/fill kernels, CohortRail P2/singleton launches,
SwiGLU, and the existing canonical owner down/BF16 sum. The R44 slot code is
a structural template for two R16 gate/up banks, but its events are not
generation-safe enough for overlapping pair work.

PairFold itself is still absent. Meta submits all four lanes together, calls
the two PairWave panels sequentially, and publishes all post graphs only
after both calls. The CUDA wrapper joins and rebroadcasts every stream;
recurrent/GDN and attention remain four-lane; corridor and PairWave events
and scratch are singleton. A qualifying implementation must first stage the
old PairWave primitive behind generation tickets, then add the exact
two-pair DAG and unified endpoint scheduler before adding cache help.

PairCache also needs new compact true-route input gather/dispatch, helper
gate/up plus SwiGLU, and compact middle return. Returns must land in shadow
storage: accept them only when ready before the owner deadline, otherwise
run the unchanged canonical descriptor locally. Copying directly into the
canonical middle would race a late fallback. Native traffic has priority;
new copy priority is middle return, helper input, then weight prefetch.

## Released arithmetic base and isolated build

The source owner released the P100-exact result at 02:08 UTC. Its terminal
tracked source-diff SHA-256 is
`82d5d9826e31e7f238b9d33fca0483ac08094121d66df38f86e15cfb864c624f`;
its qualified binary SHA-256 is
`44d0db6d838953a57fe5dc18d9045f1d3274f933c94d7e5494816a0d254403b7`.
All 640 c512 boundaries and saved logits were byte-identical. Saved-logits
SHA-256 remained
`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`.
The five paired pp2048 measurements improved by 3.851% at the median. The
three pp8128 walls were 2790.956, 2790.443, and 2791.422 ms, giving a
2912.264 tok/s median.

Only the released tracked `source.diff` was imported into this detached
worktree. Its SHA-256 was rechecked after import. No owner build artifact or
untracked file was copied. An isolated production build completed with CUDA
12.8.61, GCC C 15.2, G++ 14.3, `sm_60`, Release mode, CUDA, FA, and NCCL
enabled. The rebuilt `libggml-cuda.so` SHA-256 is
`c0258c26dcfae47461d764cd12cc98895276c0373bc988814da0117539ae795a`;
the differing binary hash is expected for an independent build and path.
The isolated `llama-bench` SHA-256 starts `29a87b1d`, and
`llama-affinity-wave-bench` starts `c41eab6e`.

## Held-out route capture

The production `--chunks 32` path emits 34 windows: raw windows 0, 1, and 2
are byte-identical internal prelude replays, followed by 31 distinct corpus
windows. The capture normalizer therefore accepts only that exact shape,
proves the three preludes identical, drops raw windows 1 and 2, and selects
raw indices `[0, 3, ..., 33]`. It then validates a canonical 32-window
artifact with fixed train/validation/test splits of 16/8/8. It rejects wrong
sizes, a nonidentical prelude, invalid IDs or ordering, duplicate canonical
passes, and output overwrite.

The clean v2 capture completed under the four-GPU coordination lock and
watchdog. These binary captures are ignored by the repository's `*.bin`
rule, so their absolute paths and hashes are part of the handoff record:

- raw bytes: 353,839,360
- raw path:
  `results/qwen36-35b-moe-pp-20260721/paircache-20260726/paircache-route-capture-code32-p100exact-v2/paircache-routes-code-32xpp8128.raw.bin`
- raw SHA-256:
  `c9003a8d2b165ff70139fcc1f970c51bba271cd9e886bc9d176558458266223c`
- canonical bytes: 333,025,280
- canonical path:
  `results/qwen36-35b-moe-pp-20260721/paircache-20260726/paircache-route-capture-code32-p100exact-v2/paircache-routes-code-32xpp8128.bin`
- canonical SHA-256:
  `9547982ef910dd884705c24571b601776d27206f1d7fe0122f69db0e8fea6135`
- metadata SHA-256:
  `f2b1a4989bbf7fa22bda8bffd3ba85ccfe209865c09df954b302b841e95fcc64`

Every canonical window has 160 diagonal records and 2032x8 valid route IDs;
all 32 complete-pass hashes are distinct.

A second valid process-level capture was retained as robustness evidence.
After normalization, 30 of 32 windows match the clean v2 capture. Windows
24 and 28 differ substantially despite both artifacts passing all structural
checks. This is consistent with route-order sensitivity in rare recurrent
states and is a reason to keep the untouched test gate strict rather than
using a single same-window hit-rate estimate.

## Corrected exact replay

The replay models immutable `(layer, lane, expert, tile index, tile class,
projection)` descriptors, one compute resource per GPU, serialized source
and destination copy endpoints, native/middle/input/weight copy priority,
two exact R16 weight banks, one compact-input slot, one helper-middle slot,
one full-layout shadow slot, and generation-specific leases.

Its arithmetic DAG is:

`gate/up -> accepted shadow or owner fallback -> canonical down -> BF16
combine -> canonical sum/post`.

The first full run exposed a real generation-inversion deadlock. The replay
was corrected so every first acquisition depends on the prior release for
the same GPU and slot, input atomically acquires compact and helper-middle
storage only after the final weight transfer, and generation order is
validated structurally. A negative leapfrog regression now covers the
failure. Real window 16 passed at static prefixes 1 and 16 with exact
descriptors, ordered leases, no overlapping SM work, and no scheduler
deadlock.

The first complete replay exposed a separate selector-coverage error:
validation compared only density-prefix candidates even though the
train-only static-frequency R16 policy was manifest-emittable. Its
historical report and manifest SHA-256 values are
`aaf6c6135ecd87a47203ee6366af6ba8828f03bb98a4c46fe48c232ac4158a4c`
and
`80097ebf779a9255c8b118e6445b409a90403ab4847af7a6cc6d56bf479d1110`.
That report is retained as rejected evidence, not a decision artifact.

Both serial and deterministic four-worker replay paths now validate the
same candidate set: density prefixes 0 through 16 plus the train-only
static-frequency R16 policy. Selection is a stable tuple of validation
wall, actual entries per slot, family, and policy hash. The manifest is
sealed before the untouched test split is evaluated. Serial and parallel
self-tests, exact manifest parity, normalized report parity, selector
recomputation, Python compilation, and whitespace checks pass. The current
serial replay source SHA-256 is
`b552619e6daf6467ba8092c8096d0833c0cee128e8ca3eefa5f041c793ead0cd`;
the parallel driver SHA-256 is
`29cb6642b85b0d5ce2015231b6912f5eab33dd8a928a1bca77c4207b397ab46c`.

The decision replay retains four explicit qualification blockers even when
reporting numerical gates:

1. pair-local PairFold pre cost is unmeasured;
2. its recurrent/attention transport is not decomposed into endpoint-copy
   tasks;
3. the released PairWave arena is about 183.1 MB/GPU before metadata, not
   the provisional 131.0 MB phase-reused allocation;
4. compact BF16 return/combine and generation-safe shadow publication do not
   exist in the released runtime.

Blindly double-buffering the three data slots would add 166,461,440
bytes/GPU and violate the 0.8 GiB reclaimed-memory floor. Therefore a
numerically favorable replay would still require measured boundary probes
before production qualification.

## Authoritative replay decision

The corrected four-worker replay completed at 07:32 UTC. Its report
SHA-256 is
`32257395cdd13ea40b0ab9632ad1ac28c6158c24b747ae384a788257a420a488`;
the manifest file SHA-256 is
`7a1e0e90e1fb706f545170a2996dd6023ca4fb1a754d4ece543604f86e506387`.
The manifest's canonical SHA-256 is
`2a4d9600bc46e26ab20b6322c4d67f39cd1327999373b57127b82e3e507df7e3`,
which matches both the validation seal and the test-open marker.

Validation selected the train-only `static_frequency_r16` policy with 16
entries per active source slot and policy SHA-256
`47467a7e5563acce88f796c63ffa71e119af3d12cbe97805c2df2fd2f7059427`.
Across validation windows 16 through 23 it predicts:

- 2596.672676 ms mean wall;
- 3130.159637 tok/s;
- 185.428261 ms mean gain over R0;
- 171.904491 ms minimum window gain;
- all eight windows positive.

The manifest was sealed before opening test windows 24 through 31.
Untouched test predicts:

- 2614.724940 ms mean wall;
- 3108.548771 tok/s;
- 167.375997 ms mean gain over R0;
- 86.222731 ms minimum window gain;
- all eight windows positive.

The numerical 72 ms gain and 2637.333 ms wall gates pass, as do descriptor,
scheduler, generation-order, endpoint-overlap, and slot-capacity
invariants. The wall margin is only 22.608060 ms. All 80 manifest rows cover
the qualified layer/source owners exactly; every row contains 16 unique
owner-valid experts and uses inactive helper `source_gpu ^ 2`. Historical
density-candidate, R0, static-test, and oracle results are deep-identical
after the selector correction. Serial and parallel self-tests pass again.

The authoritative status is nevertheless
`replay-blocked-unresolved-model-boundaries`. Modeled reclaimed memory is
955,040,768 bytes/GPU (0.88945 GiB), but the runtime allocation model is
unresolved, so the memory gate is false. Required terms are also unresolved:
pair-local pre is unmeasured, recurrent/attention transport contention is
not endpoint-modeled, the phase-reused arena and PairFold live metadata are
not implemented, compact BF16 combine/shadow publication is absent, and
there is no released PairFold scheduler. Consequently:

- `required_terms_gate_passes: false`;
- `all_replay_gates_pass: false`;
- `production_qualified: false`.

Per the approved pre-code gate, no PairCache CUDA/runtime implementation or
GPU promotion campaign was started. The tracked source remains exactly the
released production diff, and the repeat-qualified production result
remains 2790.443 ms median / 2912.264 tok/s at pp8128. PairCache R16 is a
promising hypothesis, not a production rate.
