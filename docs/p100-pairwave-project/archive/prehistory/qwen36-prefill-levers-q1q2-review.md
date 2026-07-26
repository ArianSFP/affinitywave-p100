---
name: qwen36-prefill-levers-q1q2-review
description: "Build A grouped GEMM is now BUILT+shipping; external review of two prefill levers (Q1 FABsum half2 mainloop, Q2 copy-engine 4-GPU allreduce) reconciled vs code - documented, decide later"
metadata:
  node_type: memory
  type: project
  originSessionId: e4728da3-7eb9-4ecb-8126-1ba5565fcb61
---

Build A (the round-3 "grouped GEMM GO" from [[qwen36-35b-moe-pp-roadmap]]) is **BUILT and
shipping**, not a future item: `llama.cpp-q36-moe/ggml/src/ggml-cuda/moe-gemm.cu`
(`moe_gemm_q8_grouped`), committed HEAD `a0553010f` on `qwen36-35b-moe-pp`, enabled via
`GGML_CUDA_MOE_GROUPED=1` in `results-staging/deploy-35b-moe-ep.sh` (ub=4096). It is a
**non-persistent** host-tile-table + one-grid-launch kernel with a **pure FP32 8x8 FMA**
mainloop (`moe-gemm.cu:114-134`), output in expert-sorted order. Continues [[qwen36-35b-moe-pp-execution]].

2026-07-22: got 2 external-reviewer answers on the two remaining prefill levers; reconciled
vs actual code. **Decision: documented both, decide later - no code authorized.** Full plan +
gates + file map: `~/.claude/plans/our-current-most-optimised-rippling-lightning.md`.

**Q1 - FABsum chunked-FP16 (B=32) mainloop swap of moe-gemm.cu.** Reviewer: GO opt-in B=32
only, 1.35-1.55x. Corrected end-to-end = **~8-11% wall-time** (Amdahl on 30% expert-GEMM),
NOT the 15-25% originally framed. KEY: this exact idea was already prototyped on-rig
(`llama.cpp-qwen36/results/decode-opt-20260712/splitk-hgemm-20260714/hgemm_splitk.cu`) - accuracy
proven (rel-L2 4.4e-4 ~ COMPUTE_32F) but dense form hit the register-doubling wall (~10-12
TFLOPS) and was shelved "not worth it vs COMPUTE_32F." That was a DENSE HGEMM vs cuBLAS
COMPUTE_16F (which wins); the untested angle is the **MoE small-m grouped regime** (no cuBLAS
alternative - only baseline is Build A's own FP32 loop). NOT blocked by [[p100-fp16-half2-refuted]]
(that's decode matvec, memory-bound, n<=8; Q1 is compute-bound prefill GEMM at large m). **Gate 0
before any kernel work: measure Build A roofline - if memory-bound, Q1 is dead.**

**Q2 - copy-engine 4-GPU allreduce hidden under compute.** Reviewer: 3-5.5% prefill, stretch
6.5%. Wire-accounting ceiling only 7.3-7.9% (vs 13-18% profiler share - the gap is non-wire
stalls, matches Stage-D zero-sum finding). Reviewer's "persistent producer flag" design BREAKS
on the non-persistent Build A. Simpler LLMQ-style path fits: generalize `allreduce.cu`'s 2-GPU
copy-engine RS/AG (`:593-707`, hard-capped at 2 GPU today) to 4 GPU on AsyncEP's proven
`copy_stream` (`ggml-cuda.cu:2827-3061`) at the delayed-allreduce boundary
(`ggml-backend-meta.cpp:2005-2192`, already removes 2/3 FFN allreduces). Reuse events+threadfence,
NOT `cuStreamWaitValue32` (codebase avoids it). Lower ROI + higher complexity than Q1 -> secondary.

Both are opt-in speed modes needing the +/-0.003 perplexity + KL gate; sits against the
accuracy-first priority ([[qwen36-accuracy-over-prefill-speed]]).
