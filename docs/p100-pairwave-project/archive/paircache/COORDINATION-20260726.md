# PairCache coordination

2026-07-26 01:01 UTC: created detached staging worktree at
`ae2b41d682e0c18f5c7277860dd0566244413fcb`. The active
`stitchrail-20260725` worktree remains read-only and owns selection of the
final repeat-qualified arithmetic source. No source import or four-GPU job
will begin until that owner records its terminal source identity and release.
This worktree is limited to host-side capture/replay tooling in the meantime.

2026-07-26 01:15 UTC: Codex replay-tooling session is implementing only the
standalone CPU route-capture validator/replay and guarded capture wrapper.
No arithmetic source import, build, or GPU job is in scope for this work.

2026-07-26 01:18 UTC: the strict 32-window route-capture validator passes
its full 333,025,280-byte synthetic self-test. It requires corpus
provenance, fixed train/validation/test splits of 16/8/8, all 160
diagonal-ordered records per window, valid expert IDs, and distinct pass
hashes. The active source owner has not released a terminal source, so
PairCache remains host-only.

2026-07-26 01:24 UTC: Codex endpoint-scheduler subtask is adding only a
standalone deterministic CPU discrete-event scheduler and self-test under
the PairCache results directory. It will not edit the replay, arithmetic
sources, build state, or use GPUs.

2026-07-26 02:00 UTC: the source-pinned PairCache cost profile was generated
and validated from PairWave-v3 replay, trace, placement, route, and four
measured R16 bandwidth sources. Its file SHA-256 is
`0bfbe9f791acdc27dfd3202cee541c5a69c6d933df6a0534dd5bfe24a87043b3`;
the conservative high-water projection leaves 955,040,768 bytes/GPU
reclaimed and remains explicitly `runtime_ready: false`.

2026-07-26 02:05 UTC: `stitchrail-final-prefill-v2` completed its artifacts
with exactness PASS, all five pp2048 pairs PASS, and all three pp8128 pairs
PASS. No campaign process remains and the global lock probes free. PairCache
has requested, but has not yet received, the owner-recorded terminal source
scope/hash and explicit source/GPU release, so import and GPU work remain
deferred.

2026-07-26 02:08 UTC: the owner explicitly released its worktree, terminal
tracked source, and all four GPUs. The released tracked source-diff SHA-256
is `82d5d9826e31e7f238b9d33fca0483ac08094121d66df38f86e15cfb864c624f`;
the qualified binary SHA-256 is
`44d0db6d838953a57fe5dc18d9045f1d3274f933c94d7e5494816a0d254403b7`.
PairCache is importing exactly the released static `source.diff` into this
detached worktree, with no untracked owner artifacts.

2026-07-26 02:14 UTC: exact source import and the isolated CUDA 12.8.61,
GCC C 15.2 / G++ 14.3, sm_60 production build completed. The imported
tracked diff still matches the released SHA-256. PairCache is starting its
first four-GPU job:
one locked, watched, no-warmup capture of 32 distinct pp8128 code-corpus
windows with P100-exact selectors enabled. No other llama/Nsight process is
active, the lock probes free, `.xsession-errors` is 22,762 bytes, and the
root filesystem has 35 GiB free.

2026-07-26 02:22 UTC: the first route run completed safely but the harness
rejected its size. `--chunks 32` emits 34 complete runtime windows: raw
indices 0, 1, and 2 are byte-identical internal prelude replays, while
indices 3 through 33 are unique. A strict normalizer now requires that exact
34-window shape and byte-identical prelude, drops raw indices 1 and 2, and
then applies the existing full validator to the resulting 32 distinct
windows. A real-data normalization test passed with 32/32 unique canonical
pass hashes. The first rejected raw capture remains archived. Starting one
clean end-to-end v2 capture with the corrected harness; the lock is free,
the watchdog remains 22,762 bytes, and no competing process is active.

2026-07-26 02:31 UTC: the v2 capture completed end-to-end with status 0.
The 353,839,360-byte raw SHA-256 is
`c9003a8d2b165ff70139fcc1f970c51bba271cd9e886bc9d176558458266223c`;
the normalized 333,025,280-byte canonical SHA-256 is
`9547982ef910dd884705c24571b601776d27206f1d7fe0122f69db0e8fea6135`.
All 32 windows have 160 diagonal-ordered records and distinct canonical pass
hashes, with 16/8/8 train/validation/test splits and 32 validated split
files. The global lock is free, no project GPU process remains, and
`.xsession-errors` remains 22,762 bytes. Starting the CPU-only decision
replay; GPUs remain released.

2026-07-26 07:34 UTC: the first complete CPU replay was rejected after its
selector was found to validate only density prefixes while still permitting
the train-only static R16 policy to be emitted. Both serial and deterministic
four-worker paths were corrected to validate density prefixes 0 through 16
plus static-frequency R16, use the same stable selection tuple, seal the
manifest before test, and keep test results out of the manifest. Compilation,
self-tests, manifest parity, normalized report parity, and selector
recomputation passed before the authoritative restart.

2026-07-26 07:34 UTC: the corrected authoritative CPU replay completed.
Validation selected `static_frequency_r16` at 2596.672676 ms /
3130.159637 tok/s; untouched test predicted 2614.724940 ms /
3108.548771 tok/s and +167.375997 ms versus R0, with every validation and
test window positive. Report SHA-256 is
`32257395cdd13ea40b0ab9632ad1ac28c6158c24b747ae384a788257a420a488`;
manifest file SHA-256 is
`7a1e0e90e1fb706f545170a2996dd6023ca4fb1a754d4ece543604f86e506387`;
canonical manifest/seal SHA-256 is
`2a4d9600bc46e26ab20b6322c4d67f39cd1327999373b57127b82e3e507df7e3`.
An independent 490-assertion audit passed.

The terminal replay status is
`replay-blocked-unresolved-model-boundaries`. Although numerical,
descriptor, scheduler, generation, and slot-capacity gates pass, the memory
and required-term gates fail: PairFold runtime and pair-local measurement,
endpoint-modeled pre transport, resolved arena/live metadata, and compact
BF16 combine/shadow publication are absent. Per the approved gate, no
PairCache CUDA/runtime implementation, GPU benchmark, or promotion was
started. Tracked source remains exactly the released production diff
SHA-256
`82d5d9826e31e7f238b9d33fca0483ac08094121d66df38f86e15cfb864c624f`.
The global lock probes free, no project GPU process is present, and
`.xsession-errors` remains 22,762 bytes. Device nodes have only the existing
Nautilus and GNOME text-editor handles. `nvidia-smi` cannot provide an
independent utilization reading because NVML 580.173 does not match kernel
driver 580.159.03. PairCache releases all GPU scope.
