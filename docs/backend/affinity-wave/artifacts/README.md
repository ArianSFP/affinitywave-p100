# Curated AffinityWave artifacts

This directory intentionally contains only small, reviewable inputs and
evidence. It excludes:

- model weights;
- compiled binaries;
- build directories;
- multi-megabyte nsys reports and SQLite exports;
- private prompts or route traces.

## Included calibration

| File | Purpose | SHA-256 |
|---|---|---|
| `placement-hot16.json` | Aggregate-proxy primary owners and hot-16 placement | `96de3b381bb197b5d843bc9e536496114b74db2cfb77d6f7bf254c75dd851030` |
| `placement-primary.eplb` | Parent production expert placement | `e7ea71d1a6939a49152f8aae3b6792cc1d98a6be31e7abdd411c6e8466a5de85` |

## Included evidence

| File | Purpose | SHA-256 |
|---|---|---|
| `checkpoint-exact-t64-bf16.log` | Four-GPU exact service check | `64efe823b251307e9704dff95df09ad84e8f86229ae7344edf462724028d2550` |
| `checkpoint-pp8128-confirm1.err` | First clean projected pp8128 confirmation | `574b1465f068c16c3ffab87b2daba1e7ecd2136dca6c7f72c7d639fdcf3936ed` |
| `checkpoint-pp8128-confirm2.err` | Second clean projected pp8128 confirmation | `65e49b1951e680527a9977b8f155c29119ca6dbaaefa6103423ff970fbd0c7bc` |

## Raw evidence archive

The adjacent [archive](../archive/README.md) includes the small raw logs,
scripts, reports, and chronological notes from the banked round and the
post-checkpoint experiments. This makes the numerical history auditable
without requiring access to the original workstation.

The original private workstation additionally retains:

- four complete nsys reports and their SQLite exports;
- one incomplete final nsys capture;
- the build directories and the separately obtained model.

The profiler set is about 45 MB and is intentionally not committed. Share
specific traces as private release assets if a collaborator needs timeline
analysis.
