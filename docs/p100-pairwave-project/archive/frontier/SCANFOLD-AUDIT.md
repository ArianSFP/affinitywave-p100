# ScanFold audit snapshot

Date: 2026-07-24

Canonical live record:

`NOTES-codex.md`

This file is a focused snapshot of the ScanFold proposal audit. The notes
remain live because every subsequent experiment must continue to be
recorded.

## Decision

Full ScanFold is rejected on this hardware. Its CP4 Gated DeltaNet premise
fails the isolated timing gate: the current exact 2,032-token recurrent
kernel alone is 5.46481 ms versus a proposed 3.5 ms budget for local work,
summary construction, 4 MiB communication, scan, and merge combined.
Affine chunk composition also changes FP32 operation order and is not
bitwise exact.

The useful idea is retained as a module-specific FoldWave composition:

- current exact chronological GDN placement;
- exact balanced attention epochs at the ten attention layers;
- R16 ephemeral exact-T64 StreamCacheRelay for complete owner groups;
- existing exact dense panel and selector mechanisms;
- a measured dependency replay before integration.

## Exact attention result

The equal 1,016-token proposal is not bitwise. The exact replacement uses
three fixed K partitions and within-lane splits:

`1024/1008, 1040/992, 1056/976, 1072/960`

All 16,646,144 checked FP32 values matched bitwise. The two-KV-head critical
pair is 29.640 ms. Variable-size hidden shuffle, chronological K/V
all-gather, and inverse shuffle total 5.024 ms with zero byte errors and
33,685,504 bytes/GPU scratch. The isolated epoch is 34.664 ms.

## Low-memory MoE result

R16 streams 53,477,376 exact-T64 bytes/GPU/layer into reusable scratch.
Under the long production-shaped dense corridor, even the worst captured
source skew exposes only 0.027 ms. The same transfer exposes 4.190 ms under
the short corridor, so static next-layer prefetch placement is mandatory.
A double buffer occupies 106,954,752 bytes/GPU.

The complete 40-layer route pass checked 2,663,383,040 owner values and
665,845,760 final values at full width with zero mismatches.

## Closed parts

- CP4 affine GDN scan as currently formulated.
- Equal 1,016-token attention chunks.
- Global seven-by-1,024 attention tiling.
- Full FP32 hidden all-gather MoE.
- Persistent multi-GiB expert replication.
- Three-CTA compact exact-Q8 kernel.
- BF16/protected-column GuardWire transport.

## Next gate

The conservative replay must reproduce BroadWave within 2%, then compose:

- measured R16 source-skew prefetch windows;
- exact 34.664 ms attention epochs;
- complete-owner-group service dependencies;
- existing exact dense panel/selector savings;
- PCIe and SM contention.

No full scheduler implementation is justified unless that replay predicts
an architectural-scale wall-time reduction.

## Replay result

The replay reproduced BroadWave within 0.0228%, but the architecture gate
failed. The central exact composition is 2,971.971 ms, or 2,734.89 tok/s.
Even a noncomposable upper bound that grants free same-layer rendezvous and
the incompatible RelayWave panel ceiling is only 3,002.35 tok/s.

The earlier 278.4 ms R16 result depended on same-layer row pooling. At the
actual different-layer diagonal service unit, R16 reduces exact-Q8 work by
only 14.679 ms and total DAG wall by 52.698 ms. See `FOLDWAVE-REPLAY.md` and
`foldwave-replay-v1.json`.

Full ScanFold and its adapted FoldWave composition are therefore closed.
Exact balanced attention remains retained as a supporting primitive. A
fundamentally faster exact expert kernel is now a prerequisite for any new
placement integration.
