# Exact Qwen3.6 prefill on four P100 GPUs

This directory is the handoff record for the AffinityWave, PairWave,
CohortRail, and exact diagonal-prefill experiments performed on four Tesla
P100 PCIe 16 GB GPUs between 2026-07-22 and 2026-07-27.

The production arithmetic checkpoint is commit
`36bad6bb3ad4c9b31edc4dae9fb6c3716b95704b`. It is based on
`ae2b41d682e0c18f5c7277860dd0566244413fcb` and its 15-file source diff has
SHA-256
`82d5d9826e31e7f238b9d33fca0483ac08094121d66df38f86e15cfb864c624f`.
This branch points at that exact production source commit. The documentation
tree also preserves later PairFold and host-streaming work for historical
continuity; that later experimental source is not part of this branch.

## Current status

The rig-qualified production path is the opt-in exact diagonal N2048 path.
"Production-qualified" here means exact and repeat-qualified on this private
four-P100 rig. It does not mean upstream, enabled by default, or deployed by
the llama.cpp project.

The four selectors that distinguish the retained arm are:

```text
GGML_CUDA_AW_P100_EXACT=1
GGML_CUDA_AW_DIAGONAL_SERVICE=panel2048
GGML_CUDA_AW_Q8_ENGINE=cohortrail
GGML_CUDA_AW_DENSE_SELECTORS=exact
```

They are a delta, not the complete runtime environment. The full MoE,
AffinityWave, graph, F32 wire, BF16 partial, deterministic scatter, exact
sum-width, map, and NCCL settings are in
[the qualification harness](harness/run-qualification.sh). The production
diagonal does not require the PairWave placement manifest or `PAIR_ENV`.

It reaches a median 2912.264 prompt tokens/s at pp8128. All 640 c512 service
boundaries and the saved logits are byte-identical between the umbrella OFF
and ON arms. Both saved-logits files have SHA-256
`47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`.

The default-off PairFold runtime is also end-to-end and byte-identical at
c512. Its accurate warm pp8128 median is 3027.038 ms, or 2685.133 tok/s,
versus 2788.594 ms / 2914.731 tok/s for its same-build diagonal control.
PairFold is therefore a measured experimental architecture, but it is not
production-qualified.

The follow-up host-streaming checkpoint keeps exact Q8_0 expert shards in
pinned host memory and uses two bounded device slots per GPU. The current
all-host diagnostic recovers 7.109375 GiB/GPU and measures
4131.291 ms / 1967.42 tok/s. The best long-shape exactness-qualified mixed
policy streams only down projections: 3353.908 ms / 2423.44 tok/s while
recovering 1.840 GiB/GPU. It keeps the local-owner bypass disabled because
that shortcut was nondeterministic under pipelined pp8128.

An important naming distinction:

- PairWave is an exact two-GPU-pair placement, loader, and service primitive.
- CohortRail is the retained exact-Q8 tile engine.
- The measured 2912.264 tok/s production result belongs to the exact
  diagonal scheduler using CohortRail. It is not a measured PairWave-only
  scheduler rate.
- PairFold is the implemented 80-task pair-local scheduler built on
  PairWave. Its measured warm rate is 2685.133 tok/s.
- PairFold host streaming is the implemented loader-placement extension. Its
  safe down-only warm rate is 2423.44 tok/s and is not a production rate.
- StitchRail scheduling, InterferenceWave, and PairCache remain replay-only,
  paused, rejected, or otherwise not production-qualified.

Do not quote replay rates as production results.

## Suggested reading order

1. [RESULTS.md](RESULTS.md) - what is qualified and what is only modeled.
2. [REPRODUCING.md](REPRODUCING.md) - build and qualification procedure.
3. [ARCHITECTURE.md](ARCHITECTURE.md) - data flow, ownership, exactness, and
   component boundaries.
4. [EXPERIMENT-LEDGER.md](EXPERIMENT-LEDGER.md) - retained, rejected, and
   paused experiments.
5. [ARTIFACTS.md](ARTIFACTS.md) - evidence layout, hashes, and omitted bulky
   files.
6. [../backend/AFFINITY_WAVE.md](../backend/AFFINITY_WAVE.md) - runtime
   controls already shipped with the checkpoint.
7. [PAIRFOLD-CHECKPOINT.md](PAIRFOLD-CHECKPOINT.md) - the later pair-local
   runtime and prior attempts.
8. [PAIRFOLD-HOST-STREAMING.md](PAIRFOLD-HOST-STREAMING.md) - later
   loader-time host placement and streaming results.

The full historical notebooks are preserved as an ASCII-normalized archive:

- [frontier archive](archive/frontier/)
- [final exact campaign archive](archive/stitchrail/)
- [prehistory archive](archive/prehistory/)
- [PairCache investigation](archive/paircache/)

Machine-readable replay and qualification evidence is under
[evidence/](evidence/). The exact campaign scripts are under
[harness/](harness/). The production maps and helper scripts are included;
model weights, corpus data, build output, and machine paths must still be
provided for another rig.

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

This is a private rig-provenance archive. Historical records intentionally
retain `/home/arian` worktree and result paths so runs can be traced back to
the original machine. No credentials, private keys, tokens, model weights,
raw prompts, or generated logits are included.
