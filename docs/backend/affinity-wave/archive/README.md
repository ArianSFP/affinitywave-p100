# Raw experiment archive

This archive preserves the small, text-based evidence generated while reaching
and then probing the immutable AffinityWave checkpoint. The curated
[experiment log](../EXPERIMENTS.md) is the authoritative interpretation. Files
here are chronological source material and may describe candidates as "next"
before a later experiment closed them.

## `banked/`

This directory is an unmodified copy of every non-profiler file in the banked
result directory. It includes:

- the original plan, results, README, and Codex notes;
- Phase-0 simulator, tests, report, trace patch, and trace analyzer;
- all Phase-1 and Phase-1b service logs;
- ordinary graph scope, smoke, MMVQ, and perplexity outputs;
- benchmark and profiling wrappers;
- the exact retained checkpoint log and two clean pp8128 confirmations;
- the placement files and model checksum, but not the model itself.

The original profiler reports and SQLite exports are omitted.

## `post-checkpoint/`

This directory is an unmodified copy of every non-profiler file from the
separate experimental worktree after the checkpoint. It includes the complete
chronological notes, benchmark wrappers, and raw output for:

- grouped streams and alternate admission;
- tail and pair gap scheduling;
- fitted token balancing and attention lookahead;
- asynchronous and early corridor variants;
- recurrent, attention, and state-split release variants;
- host-side dispatch caching;
- all-M64 projection dispatch;
- owner submission pooling;
- `CUDA_SCALE_LAUNCH_QUEUES=4x`;
- the final incomplete profiling attempt's text output.

The incomplete `.nsys-rep` file is omitted because it is not valid evidence and
is not needed to reconstruct the recorded conclusion.

## Integrity and scope

The archive was copied byte-for-byte from the result directories. It contains
no model weights, compiled binaries, credentials, private prompts, or upstream
submission metadata. Some commands retain the original workstation paths so
that provenance is explicit; use [REPRODUCTION.md](../REPRODUCTION.md) and the
top-level scripts for portable commands.
