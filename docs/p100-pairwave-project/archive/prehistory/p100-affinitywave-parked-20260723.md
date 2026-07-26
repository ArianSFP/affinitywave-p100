---
name: p100-affinitywave-parked-20260723
description: "AffinityWave (Codex-built wavefront/owner-service MoE redesign) PARKED 2026-07-23: Phase-0 NO-GO stands; banked 2828 tok/s pp8128 is SERVICE-ONLY (no logits, numerics unvalidated) - never quote vs prod; 3 harvest candidates listed"
metadata:
  node_type: memory
  type: project
  originSessionId: f8ed8941-b4e7-4f80-82f0-abe4d5c74239
---

**AffinityWave PARKED by user decision 2026-07-23.** A Codex-run, user-directed
exploration of a wavefront/owner-service MoE redesign (40 layers x 4 token
lanes, cross-layer expert service on primary owners, hot-16 replication) for
the 4xP100 Qwen3.6-35B-A3B rig. Raw record:
`results/qwen36-35b-moe-pp-20260721/affinitywave-20260722/{RESULTS,NOTES-codex,PLAN}.md`
(under /home/arian/llama.cpp-qwen36 - do not edit; authoritative).

**Phase-0 verdict: NO-GO, and it stands.** Discrete-event sim calibrated on
the A6 plan-pp8192 nsys trace: pessimistic pp8192 2.181 s vs the 2.05 s gate;
more decisively the OPTIMISTIC bound (no NCCL, no gathers, perfect balance,
free transport) is 2.1122 s = already over the gate. Needs ~7.0 effective
TFLOP/s/GPU expert service; measured isolated service is 5.13-5.29 (M64
split-K2 on T64/K32 layout, exact BF16 checks 4/4 GPUs).

**Banked checkpoint (user-approved): ~2828 tok/s pp8128 - SERVICE-ONLY.
Never quote this against prod (~1697 e2e) numbers.** The run withholds model
output (returns GGML_STATUS_ABORTED, no logits); full-wavefront numerics are
UNVALIDATED (no KLD/ppl; the chunked-GDN kernel inside is ungated math);
append-prefill/decode-normalization/request lifecycle unimplemented. At
measured service rates their own optimistic ceiling is ~3780 tok/s, so max
remaining headroom ~+34% before paying for the unbuilt correctness machinery.
Checkpoint commit `9d3983b89952c7fc1c6aa38fc1a7bd3182992382` (IMMUTABLE) in
`/home/arian/llama.cpp-qwen36/affinitywave-src` (parent 05dcabf9a; diff 14
files +4979); experimental worktree `affinitywave-experimental` (branch
affinitywave-experimental-20260723); build `build-p100-affinitywave`.

**HARVEST CANDIDATES for future prod-line rounds (not started):**
1. T64/K32 exact-Q8 weight layout + M64 split-K2 kernel: 5.13-5.29 TF/s/GPU
   INCLUDING packing + owner reduction at cross-layer service shapes = direct
   evidence for the parked A4c planes-repack question. A/B it against the A6
   moe_gemm_q8_plan native-34B-read kernel on the prod path.
2. Token-level top-8 trace patch (`affinitywave-trace-05dcabf9.patch`, git
   apply --check clean vs 05dcabf9): the calibration instrument the EPLB line
   needs - both AW manifests and the prod EPLB map are aggregate-proxy-based.
3. Chunked GDN kernel (chunk columns 2, 4 warps): candidate for the ~8%
   prefill GDN slice; math-changing - requires exactness/ppl gate first.

**Reopening condition (per the report, agreed):** a token-level route trace
PLUS an isolated cross-layer expert-service demonstration of ~7.0 TF/s/GPU
including packing at >=10 GB/s peer transport. Not another backend attempt.

**Disk:** AW artifacts ~1.5 GB (src clone + build dir + experimental
worktree) are reclaimable later; keep the results dir and the immutable src
checkout at the checkpoint commit.

Related: [[qwen36-35b-moe-pp-execution]], [[p100-pp-research-round3-20260722]].
