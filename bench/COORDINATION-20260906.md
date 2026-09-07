# Coding-serving implementation - 2026-09-06

Isolated detached source from 45fccdd5665d6c6c2f0fb669c1f630af84d0630e.
User authorized autonomous implementation and validation, no commits/push/PR.
Other worktrees and qualified builds are read-only. Results and build stay
in this worktree. Shared four-P100 jobs use /tmp/affinitywave-4gpu.lock,
check existing compute clients, and run with a scoped .xsession-errors
watchdog. Host GPU check at 22:42 UTC: all four idle, no compute clients.

ACTIVE: baseline measurements and serving lifecycle port. One GPU job at
a time; no concurrent compile while collecting performance measurements.

2026-09-07: Investigated Web UI server exit. The wrapper's prewarm probe was
hard-coded to 127.0.0.1 while the server was bound to 192.168.1.37, so it
waited 300 s and then terminated the healthy server. Probe address now follows
--host, with loopback fallback for wildcard binds. Corrected server restart
will be the only active four-GPU job.

2026-09-07: The corrected UI server was stopped cleanly before a controlled
serving timing run. No other compute clients are active.

2026-09-07: Implemented an opt-in canonical KV suffix-copy path for append-only
serving continuations. Recurrent state and reset/unsupported KV layouts retain
the full-copy path. The production wrapper now enables the suffix flag; the
first validation run will use the rebuilt ggml libraries and one four-GPU job.
The suffix-enabled validator completed; because its artifact differs from an
older control campaign, an exact same-build suffix-disabled validator is
required before accepting byte identity.
The first paired comparison exposed missing invalidation across llama_memory_clear;
the backend now uses an epoch bumped by memory/sequence/state mutation APIs.
Both sides of the validator pair will be rerun after this fix.
The rerun completed with identical logits through seed2048 and the cached65/
cached130 append transitions; later long-reset boundaries vary across runs while
the suffix path is inactive there. A post-fix HTTP serving smoke is next.
Position-aware invalidation is now built into the rebuilt ggml and llama
libraries; the next HTTP run will verify cache-boundary suffix copies remain
active after the server's normal sequence trimming.
Added only env-gated suffix lifecycle diagnostics to locate why the current
server cache path still disables the marker; this run is diagnostic and will
not be used as a performance result.
Diagnostics showed logical trim 8188 versus padded canonical 8192; the trim
path now lowers the suffix baseline without bumping the state epoch.
Added a conservative 4096-token/4-token-padding eligibility guard for server
slot trims; smoke validation will confirm fresh cache_prompt=false requests do
not inherit stale short prefixes.
Guarded smoke passed transport/chat and kept fresh requests on full-copy paths;
the 2048-token cached continuation used suffix transport (646.9 -> 527.1 MiB).
Diagnostics were removed before the final ggml-base rebuild.
Final source hardening added invalidation for scheduler reservation, internal
memory-module updates, and state-file loads. `cmake --build build-coding-serve
--target llama -j8`, `git diff --check`, shell/Python syntax checks, and the
12-test CPU harness all pass. No additional GPU job was started after these
invalidation-only changes; the guarded four-GPU smoke remains the latest
performance/correctness run.
An attempted final smoke after these invalidation-only edits was blocked before
launch because the rig's `nvidia-smi --query-compute-apps` returned driver
communication status 9; no GPU process was started by the attempt.
2026-09-07: Continuing with a bounded serving graph-plan metadata cache. It
will not duplicate scheduler arenas or alter arithmetic; scheduler allocation
still runs on plan switches. Cache entries are invalidated on reserve/memory
layout changes and remain opt-in through the serving wrapper.
The cache and owner-side publication overlap are now implemented and built in
`libggml-base.so`/`libllama.so`. Static checks and the CPU harness pass. They
remain GPU-unqualified until the driver recovers; no four-GPU job was started
while `nvidia-smi` was returning status 9.
The `llama-server` target was also attempted; it reaches the pre-existing
dirty `common/speculative.cpp` errors (`common_speculative_init_result`), so
the rebuilt shared `libllama`/ggml libraries are the validated build artifacts.
Host-level execution confirmed all four P100s and `/dev/nvidia0`-`3` are
healthy. The guarded `plan-cache-final-smoke-20260907` then passed fresh,
cached, and chat probes with plan-cache/publication-overlap enabled; no GPU
process remained afterward. A paired performance/quality campaign is still
needed before reporting a rate improvement.

2026-09-07: Host-level P100 qualification completed with tag
`p100-qualification-8128-20260907` (4 repeats, fixed shape, Qwen3.6-35B-A3B
Q8_0, CUDA, tensor split, FA enabled, 4x Tesla P100). `llama-bench` reported
`avg_ts=2900.109901`, `stddev_ts=3.095575` tok/s with samples
`[2899.7, 2897.03, 2904.4, 2899.3]`. All GPU processes exited cleanly.
This is 12.154 tok/s (0.417%) below the 2912.264 documented checkpoint; it is
a fresh qualification measurement, not an accuracy/PPL gate. The service-phase
diagnostic timing in the same run was higher (~2967 tok/s) and is not used as
the qualified benchmark rate.

2026-09-07: Incorporated upstream P100 patch 06 (`06-mmq-mul-mat-id-sm60`)
from shinbunbun/llama-cpp-p100-patches. It enables MMQ only for Pascal
`MUL_MAT_ID`/MoE calls and preserves the cuBLAS path for ordinary `MUL_MAT`.
The CUDA/shared libraries built successfully. A guarded 4-P100 qualification
with the rebuilt libraries completed cleanly at `2896.223626 +/- 5.552398`
tok/s (8128 tokens, 4 repeats), within run variance of the pre-patch
`2900.109901 +/- 3.095575` result. The current EP/device-plan serving path
does not exercise patch 06 by design; it remains active for the non-EP,
plan-disabled MoE fallback.
