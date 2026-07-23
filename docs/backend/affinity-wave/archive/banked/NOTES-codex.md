# Codex handoff notes

- Read the shared memory index, round-3 and post-MTP plans, current round-3
  results, relevant execution/review/gotcha memories, and the 35B coordination
  ledger before working.
- The requested result directory is in `/home/arian/llama.cpp-qwen36`, but that
  checkout is the older 27B branch at `beac5309f`. The measured 35B implementation
  is a separate, dirty shared worktree at `/home/arian/llama.cpp-q36-moe`, HEAD
  `05dcabf9`. No files in that worktree were modified.
- Added an offline Phase-0 implementation and a capture patch that checks cleanly
  against the 35B worktree. Did not transplant the 35B stack into the 27B tree.
- No GPU command was run. The model hash and GGUF metadata reads were CPU/disk
  only. `nvidia-smi` was unavailable from the sandbox, so no current driver-state
  claim is made.
- CUDA compiler observed: 12.8.61. Python: 3.14.4.
- The failed 2.05 s gate is the terminal condition specified by PLAN.md. No
  `build-p100-affinitywave` directory was created, and no runtime env controls
  were integrated.

## 2026-07-22 user override / Phase 1

- User explicitly overrode the Phase-0 stop condition and asked for a prototype.
- Created an isolated 35B source clone at `affinitywave-src` and distinct
  `build-p100-affinitywave`; the shared q36-moe worktree stayed untouched.
- Implemented strict env/manifest validation and a native-Q8 cross-layer service
  benchmark through the CUDA proc-address table. Full `llama-bench` links.
- Four-GPU F32 owner-service accuracy is exact, but the retained minimum is
  4.814 TFLOP/s versus the 5.5 gate. 32x128 and direct-grid variants were
  measured and reverted. Do not claim Phase 2 or 3200/4000 tok/s.

## 2026-07-23 retained checkpoint

- The user accepted the exact T64/K32 M64 split-K2 path and explicitly approved
  one local checkpoint commit in the isolated worktree.
- Removed the unfinished ring-mirror branch and its environment control before
  banking. Also removed the temporary direct CUDA error print, graph-disable
  benchmark switch, and live VRAM probes. Default-off exploratory paths remain
  isolated, and the banked command pins shared service, lane balancing, fused
  gate/up, and down cache off.
- Rebuilt `llama-bench` and `llama-affinity-wave-bench` successfully in Release
  mode for sm_60 with CUDA 12.8.61.
- Two clean watchdoged pp8128 runs with T64, BF16, GDN 2/4, stagger 1, and
  singleton `1111` measured 2874.222 ms / 2827.9 tok/s and 2874.839 ms /
  2827.3 tok/s. Mean: 2874.531 ms / 2827.6 tok/s. The status 1 after each timed
  line is the expected output-withheld benchmark abort.
- Clean exactness rerun: 16,777,216 checked values per GPU, zero mismatches on
  all four GPUs, expected and observed BF16 bits 48020. Wall-derived service
  range was 5.128-5.285 TFLOP/s/GPU; the earlier exact retained result was
  5.237 TFLOP/s/GPU.
- Full pp command:
  `bash /home/arian/llama.cpp-qwen36/results/qwen36-35b-moe-pp-20260721/affinitywave-20260722/benchmark-phase2a-smoke.sh t64 t64 0 0 1 1 0 8128 1 service 1 1 0 0 0 0 bf16 1 2 1111 4 0`.
- Exactness command:
  `bash /home/arian/llama.cpp-qwen36/results/qwen36-35b-moe-pp-20260721/affinitywave-20260722/benchmark-phase1b.sh t64-bf16`.
- Parent: `05dcabf9a97b8a91bec5621d2b4c5ce1ce9b2ca8`.
- Local checkpoint: `9d3983b89952c7fc1c6aa38fc1a7bd3182992382`.
- Tree: `acbecf68f512212b95015540e3774864bf935aed`.
- Complete staged binary diff SHA-256:
  `62ec789b8e55fcc9a3128dc7f27c8d6f3157b9b7b65b428f26cbf1e269f656aa`.
- Source diff: 14 files, 4,979 insertions, 71 deletions. Recover with
  `git show --binary 9d3983b89952c7fc1c6aa38fc1a7bd3182992382`.
- Binary hashes: `llama-bench`
  `eb949e2cf086b0e6dab6fd018418f2f2ec356a1585c035ff06e93c439e8a26c8`;
  `llama-affinity-wave-bench`
  `0f49ab934b6a80e5109b43dbe4276ccdf4cf4a8f96893a1aa344ee2109a093a8`;
  `libggml-cuda.so.0.15.3`
  `d38b3a4f9fea749a0edfdab6ded9d88135bc3e8ec87291a2b5b1802aaf82e223`.
- Raw logs: `checkpoint-pp8128-confirm1.err`,
  `checkpoint-pp8128-confirm2.err`, and `checkpoint-exact-t64-bf16.log`.
  Complete effective environment, hashes, and limitations are recorded in the
  retained-checkpoint section of `RESULTS.md`.
- Remaining limitations: output is withheld; full-wave logits/KLD validation,
  append-prefill, decode normalization, and production request lifecycle remain
  unimplemented. This is approximately 2828 tok/s, not 3500 tok/s.
- Production sources and protected builds were untouched. No push or PR. Treat
  checkpoint commit `9d3983b89952c7fc1c6aa38fc1a7bd3182992382` as immutable; continue only in a
  separate experimental worktree/state.
- Separate clean experimental state created at
  `/home/arian/llama.cpp-qwen36/affinitywave-experimental`, branch
  `affinitywave-experimental-20260723`, rooted at the checkpoint commit. Keep
  `/home/arian/llama.cpp-qwen36/affinitywave-src` unchanged.
