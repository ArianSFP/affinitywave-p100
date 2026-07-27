# AffinityWave frontier pause handoff

Paused: 2026-07-24 22:59 UTC

## Isolation state

- Immutable reference:
  `ae2b41d682e0c18f5c7277860dd0566244413fcb`
- Detached source:
  `/home/arian/llama.cpp-q36-moe/.worktrees/affinitywave-frontier-20260724`
- Separate build:
  `/home/arian/llama.cpp-q36-moe/.worktrees/build-p100-affinitywave-frontier-20260724`
- Results:
  `/home/arian/llama.cpp-q36-moe/.worktrees/affinitywave-frontier-20260724/results/qwen36-35b-moe-pp-20260721/affinitywave-frontier-20260724`
- Production remains on `affinitywave-production-20260723` at the immutable
  reference. Its pre-existing changes and untracked captures were not
  modified.
- No commit, push, or PR was made.
- No experimental GPU job is active and the four-GPU flock is released.

## Performance state

| configuration | pp8128 wall (ms) | rate (tok/s) | decision |
| --- | ---: | ---: | --- |
| production reference | 3,229.024 | 2,517.169 | immutable baseline |
| detached reproduction mean | 3,228.696 | 2,517.425 | reproduction passed |
| BroadWave matched mean | 3,157.830 | 2,573.919 | exact retained base |
| HalfPipe safe mean | 3,147.205 | 2,582.610 | exact, default-off credit |
| direct owner, clean run | 3,264.180 | 2,490.059 | exact, speed closed |
| 4,000 tok/s target | 2,032.000 | 4,000.000 | not reached |

The strongest measured exact configuration is still about 1,115 ms from the
4,000 tok/s wall-time target. No tested composition passes the required
approximately 2.00-second conservative integration gate.

## Most recent decisions

- RecurFill:
  closed after exact four-GPU selector testing. Its best favorable bound is
  28.96 ms over all 30 recurrent layers before real dependencies.
- HalfPipe:
  `halfpipe_sync` is bitwise exact and saves 10.626 ms pp8128.
  `halfpipe_bar` is exact but slower. No further staging sweep is justified.
- Direct peer owner-gather:
  bitwise exact and skips 254 MiB/GPU, but regresses pp8128 by 120.804 ms.
  Matched traces show a 70.053 ms increase in the perfect-balance work floor.
  Do not pursue panel-ready variants of the same peer-load primitive.
- Materialization-free service:
  N256 remains the preferred capacity boundary at 123,087,872 bytes/GPU,
  about 1.126 GiB/GPU below the current four-state service, with a measured
  1.8160 ms/service arithmetic tax.

## Canonical records

- Full chronological notebook: `NOTES-codex.md`
- RecurFill: `RECURFILL-AUDIT.md`
- HalfPipe: `HALFPIPE-AUDIT.md`
- Direct owner-gather: `DIRECT-OWNER-AUDIT.md`
- Placement and bounded queue: `TEMPORAL-QUEUE-AUDIT.md`
- Head-sharded placement: `HEADFOLD-AUDIT.md`
- N-panel service: `NPANEL-SERVICE-AUDIT.md`
- Complete bounded service: `MATERIALIZATION-FREE-SERVICE-AUDIT.md`

Each audit names its raw outputs, traces, resource reports, and source probes.

## Resume boundary

On resumption:

1. Recheck `bench/COORDINATION-20260724.md`, acquire
   `/tmp/affinitywave-4gpu.lock`, and start the `.xsession-errors` watchdog.
2. Keep `GGML_CUDA_AW_Q8_ENGINE=halfpipe_sync` only as a measured 10.626 ms
   credit; it is not the next architecture.
3. Do not reopen RecurFill, PTX split barriers, direct peer owner
   panelization, route-scatter fusion, or same-layer synchronized owner
   copies without materially new evidence.
4. Require the next architecture replay to predict at most 2.00 seconds
   before invasive integration. The remaining gap requires a joint
   placement/arithmetic change, not another local launch or tile sweep.
