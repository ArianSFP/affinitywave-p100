# Exact Qwen3.6 prefill on four P100 GPUs

This directory is the handoff record for the AffinityWave, PairWave,
CohortRail, and exact diagonal-prefill experiments performed on four Tesla
P100 PCIe 16 GB GPUs between 2026-07-22 and 2026-07-26.

The production arithmetic checkpoint is commit
`36bad6bb3ad4c9b31edc4dae9fb6c3716b95704b`. It is based on
`ae2b41d682e0c18f5c7277860dd0566244413fcb` and its 15-file source diff has
SHA-256
`82d5d9826e31e7f238b9d33fca0483ac08094121d66df38f86e15cfb864c624f`.
The later PairFold runtime checkpoint is committed with
[PAIRFOLD-CHECKPOINT.md](PAIRFOLD-CHECKPOINT.md) on documentation base
`619e7cc7c5425b57ad1f5350d2eaa85feecc4187`.

## Current status

The production-qualified path is the default-off exact diagonal N2048 path:

```text
GGML_CUDA_AW_P100_EXACT=1
GGML_CUDA_AW_DIAGONAL_SERVICE=panel2048
GGML_CUDA_AW_Q8_ENGINE=cohortrail
GGML_CUDA_AW_DENSE_SELECTORS=exact
```

It reaches a median 2912.264 prompt tokens/s at pp8128. All 640 c512 service
boundaries and the saved logits are byte-identical between the umbrella OFF
and ON arms. Both saved-logits files have SHA-256
`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`.

The default-off PairFold runtime is also end-to-end and byte-identical at
c512. Its accurate warm pp8128 median is 3027.038 ms, or 2685.133 tok/s,
versus 2788.594 ms / 2914.731 tok/s for its same-build diagonal control.
PairFold is therefore a measured experimental architecture, but it is not
production-qualified.

An important naming distinction:

- PairWave is an exact two-GPU-pair placement, loader, and service primitive.
- CohortRail is the retained exact-Q8 tile engine.
- The measured 2912.264 tok/s production result belongs to the exact
  diagonal scheduler using CohortRail. It is not a measured PairWave-only
  scheduler rate.
- PairFold is the implemented 80-task pair-local scheduler built on
  PairWave. Its measured warm rate is 2685.133 tok/s.
- StitchRail scheduling, InterferenceWave, and PairCache remain replay-only,
  paused, rejected, or otherwise not production-qualified.

Do not quote replay rates as production results.

## Suggested reading order

1. [PAIRFOLD-CHECKPOINT.md](PAIRFOLD-CHECKPOINT.md) - the implemented
   pair-local runtime, accurate warm result, prior attempts, and next work.
2. [RESULTS.md](RESULTS.md) - what is qualified and what is only modeled.
3. [ARCHITECTURE.md](ARCHITECTURE.md) - data flow, ownership, exactness, and
   component boundaries.
4. [EXPERIMENT-LEDGER.md](EXPERIMENT-LEDGER.md) - retained, rejected, and
   paused experiments.
5. [REPRODUCING.md](REPRODUCING.md) - build and qualification procedure.
6. [ARTIFACTS.md](ARTIFACTS.md) - evidence layout, hashes, and omitted bulky
   files.
7. [../backend/AFFINITY_WAVE.md](../backend/AFFINITY_WAVE.md) - runtime
   controls already shipped with the checkpoint.

The full historical notebooks are preserved as an ASCII-normalized archive:

- [frontier archive](archive/frontier/)
- [final exact campaign archive](archive/stitchrail/)
- [prehistory archive](archive/prehistory/)
- [PairCache investigation](archive/paircache/)

Machine-readable replay and qualification evidence is under
[evidence/](evidence/). The exact campaign scripts are under
[harness/](harness/); they retain the original absolute rig paths and must be
adapted on another machine.

## Non-negotiable constraints

- Q8_0 model weights only. Sub-Q8 weight quantization was never in scope.
- Accuracy before speed: byte identity where possible, otherwise the paired
  KLD/PPL gate is `+/-0.003`.
- Router IDs, router weights, lane identity, tile classes, FP32 accumulator
  order, SwiGLU, BF16 owner boundaries, and canonical owner sum order must
  remain unchanged.
- Experimental behavior is default-off and must reject incompatible
  manifests or hardware instead of silently changing arithmetic.
- Only one four-GPU job may run at a time. Every run needs the coordination
  lock and `.xsession-errors` watchdog described in
  [REPRODUCING.md](REPRODUCING.md).

## Scope of this archive

This archive includes 63 frontier Markdown records, 12 final-campaign
Markdown records, the relevant pre-AffinityWave plans and results, the
PairCache no-go investigation, principal replay manifests, campaign TSVs,
static hashes, and the qualification harness.

Multi-gigabyte route captures, Nsight databases, SASS dumps, cubins, object
files, model files, and generated logits are intentionally excluded. Their
paths, sizes, and hashes are recorded where available so the evidence chain
does not silently become a source-code claim.
