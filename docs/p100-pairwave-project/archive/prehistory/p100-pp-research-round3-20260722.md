---
name: p100-pp-research-round3-20260722
description: Round-3 research (planning only) - grouped-GEMM feasibility assessed vs rig trace; EPLB answer in; Q1 (SASS toolchain) + Q2 (decode small-msg allreduce) pending
metadata:
  node_type: memory
  type: project
  originSessionId: 774c08bf-e788-4fc6-b3f5-41a570c672a0
---

Round-3 research, 2026-07-22, PLANNING PHASE - no execution authorized yet.
Three questions handed to user's deep-research process (framings in session):
Q1 sm_60 SASS toolchain/precedents (de-risk grouped GEMM), Q2 custom small-message
P2P allreduce for DECODE on 4xPCIe Pascal, Q3 EPLB-lite minimal static design.

**Grouped-GEMM feasibility doc ASSESSED vs ctrl-d4.sqlite (key trace facts, GPU0, 2 evals):**
- Deployed expert GEMM = maxwell_sgemm_fp16_{128x32,64x64}_tn + separate splitKreduce
  (43.9k launches) + gemv2T leakage; achieved ~5.5-6.4 TFLOP/s = 60-70% of PCIe peak.
  Mainloop vs custom kernel is ~a WASH - all the win is fusion/launch elimination.
- mmid gather (k_get_rows_float big launches) ~1.45 s/eval = AS LARGE AS the expert
  GEMM slice; dequant tiny (1.7%) but 26k launches/eval; per-expert GEMM fixed floor
  19us (mean 37us @ m~128) -> fixed cost ~half of mean.
- Break-even for fused grouped kernel ~0.5-0.6x deployed GEMM rate (overhead ~1.1x
  GEMM); doc's 0.90-0.92x Phase-1 gate = stretch goal, NOT kill bar. Realistic prize
  +20-28% pp (1475 -> ~1800-1900). Amdahl cap: NCCL 26%/FA 10%/GDN 8% untouched.
- Model-agnostic for MoE+Q8_0 (B iterator is the only quant-specific part; Q4_0 would
  work, slightly lesser %). Dense models (Gemma) get nothing. Hardware-locked sm_60.
- Trace observation: main activation allreduce already runs BF16 on the wire
  (AllReduce_Sum_bf16_RING_LL) -> deferred P5 Q8-wire upside is ~half original est.

**Q3 EPLB answer ASSESSED (recommendation: R=0 repack -> R=8 -> maybe R=16):**
- Its %s are of the MoE-layer critical path; e2e prefill ceiling from perfect balance
  ~5-7% (MoE slice ~30% wall; arrival-wait pool = NCCL minus wire ~11% wall, expert
  skew ~half of it). R=0 pack ~4-6% e2e IF skew is stable across batches.
- Skew 1.24 cannot be iid noise (iid -> ~1.01); structure real, but stable-popularity
  vs per-batch-bursty is THE open question -> Phase-1 replay (few hundred ubatch
  histograms) decides. Same capture window doubles as grouped-GEMM Phase-0 m-histogram.
- OUR assignment is literally contiguous load-blind ranges (EP4) -> R=0 upside maximal.
  R=0 = load-time per-layer permutation of expert order + ffn_gate_inp rows (logical=
  physical), zero kernel work, split stays a partition. NOT byte-identical (partial-sum
  regrouping + tie flips) -> ppl/KLD gate class.
- Replication R>0: fork split is a partition by construction -> replicas need side
  buffers + logical->physical remap = real surgery. AND today's fixed cost/expert
  (~60-100us/layer) means a replica's destination absorbs more fixed work than the hot
  GPU sheds (~42us/layer variable) -> replication BARELY PAYS pre-grouped-kernel.
  SEQUENCING: R=0 now-viable; defer all R>0 until after grouped kernel (which zeroes
  the fixed cost). Post-kernel R=8 est +1-3% e2e, 256 MiB/GPU. Exact quota splitting
  is native to csort (occurrence indices exist). Decode: ~nil (8-expert draw too sparse).

**Q1 SASS-toolchain answer ASSESSED (2026-07-22):**
- Toolchain EXISTS, not turnkey-proven: RCAsm-bundled CuAssembler (adapted for CUDA
  12.8 cubins, retains SM60 backend incl. Pascal 64-bit encoding + control words),
  used directly (RCAsm frontend itself is SM89/SM120-only). Plan C = original OpenAI
  maxas with old-toolkit ptxas scaffold in a container (native-cubin forward compat
  on R580). denvdis (Pascal-aware, updated 2026-07-22) for encoding verification and
  surgical patches. NO published P100/R580 end-to-end validation -> build Phase 1 =
  round-trip gate: nvcc -cubin sm_60 -> cuasm disasm/reasm unchanged -> cuModuleLoad
  -> identical output (single GPU, minutes). ABI scaffold must carry the FINAL param
  list (fragile parts = const-bank offsets, reg-count/smem metadata, control packing).
- (b) NO public compiler-scheduled result exists in our exact class (A16/B16 storage,
  F32 acc, P100, skinny). Hand-SASS openai = 74.8% of SXM2 peak @ m=128 (PCIe-scaled
  pass bar ~6.5-7.0 TF/s); ISAAC autotuned-PTX skinny = 38.5% (fp32 storage, N=32).
  => SASS is the PRIMARY perf route; CUDA-C port = correctness scaffold + floor, not
  the bet. Floor math: CUDA-C @ 40-50% peak fused still nets ~+10-14% e2e (overhead
  elimination alone); SASS @ ~7 TF/s nets the +20-28%. Project has a positive floor
  even if SASS route fails.
- (c) Partial cuBLAS RE MATCHES our trace: 128x32_tn = 256 thr / 16 KiB smem /
  workspace split-K + separate splitKreduce (our 43.9k reducer launches); inferred
  4x4 microtile (16 acc/thread vs openai's 64) explains deployed 65-70%-of-peak.
- (d) NOTHING to harvest, build-vs-wait = BUILD: mainline fused mul_mat_id is
  MMA-only + rejects quantized src (Sep 2025, never selects on sm_60); no sm_60 work
  in flight in mainline/ik_llama; AMD RDNA3.5 block-map PR (#63) independently
  validates the static-launch device-scheduler design (our csort already produces
  its inputs).

**Q2 decode small-msg allreduce answer ASSESSED (2026-07-22) - ROUND COMPLETE:**
- sm_60-feasible via membar.sys + volatile load/store; NO peer atomics over PCIe on
  P100 (NVLink-only) -> one-writer epoch flags, RMW-free. No published 4xP100-PCIe
  4-64KB table exists anywhere. vLLM removed >2-PCIe-GPU support (small gain + sync
  races) -> copy-compile is NOT an option; repaired protocol (all-thread fences,
  epoch banks, final lifetime barrier) required. One-shot first; two-shot only after
  measuring a correct middle-barrier cost. Keep NCCL default, env-gated opt-in.
- RIG FIT STRONGER than generic case: decode messages sit exactly in the one-shot
  window (batch-1 = 4 KB = 2048 hidden x bf16; MTP verify n~5-8 = 20-32 KB);
  single-process llama.cpp -> direct peer pointers, NO IPC layer; canonical-order
  FP32-accum one-shot is MORE accurate than current bf16 ring reduction. Weaker/
  unknown: PCIe topology never characterized (need topo -m + 12-directed-pair probe;
  nonuniform root complexes -> NCCL likely dominates); CUDA-graphs decode -> epochs
  must be device-side-derived under graph replay (custom kernel captures natively,
  unlike NCCL). Prefill untouched (16 MB messages, ring correct there).
- Prize model: ~80-96 allreduces/decode token (trace: 2 per MoE layer per pass); if
  NCCL PCIe LL is ~25-40us/call that is 2-4ms of the 13.4ms step; one-shot saving
  10-20us/call -> est +8-15% decode IF the share confirms. Determinism: bitwise
  across ranks, differs from NCCL order -> ppl/KLD gate class.
- GATES before build (cheap, shareable): decode-NCCL share from EXISTING P1-era
  decode nsys traces (offline, do first); nccl-tests -b 4K -e 64K forced Ring/LL
  latency (same rig session as the TCCL busbw gate); topology/12-pair probe.

**PHASE 0 EXECUTED (2026-07-22, same day; results/qwen36-35b-moe-pp-20260721/round3-phase0/RESULTS.md):**
- A0 SASS toolchain: PASS, PROVEN ON-RIG - github.com/gpuocelot/cuasm (ships
  DefaultInsAsmRepos.sm_60; "RCAsm" does not exist under that name) round-trips
  CUDA 12.8 sm_60 cubins BYTE-IDENTICAL, cuModuleLoad+run OK on driver 580, and a
  hand-modified SASS body assembled bit-identical to nvcc's reference. Needs venv
  with pyelftools+sympy; nvcc needs -ccbin gcc-14 AND -lstdc++ (operator-new quirk).
- P0.2 topology: ALL SIX PAIRS PIX single-switch, 12 pairs uniform 12.9 GB/s,
  1.2 us peer latency, nativeAtomics=0; one-shot all-to-all contention sustains
  9.7 GB/s/GPU. Best case for Build C.
- P0.3 NCCL: ~30 us FLAT 4-64KB (auto=Ring/LL class; bf16 32-36); large busbw
  9.9 GB/s = 76% of wire => TCCL CLOSED for good.
- P0.1 decode share (fresh prod trace; env -i KILLS nsys injection - use plain env):
  80.4 allreduce/token, avg 47 us (30 clean + wait), = 3.8 ms = 28% of the 13.4 ms
  step; decode wire dtype is F32 8KB (bf16 is prefill-only). Build C GO, +10-13% tg.
- P0.4 routing (GGML_CUDA_MOE_HIST instrument added to csort fallback, kept):
  EXPERT ROUTING IS DOMAIN-SPECIALIZED TO THE EXTREME - wiki-vs-code hot-16 overlap
  0.016 (disjoint). Within-code stability 0.892; code-calibrated R=0 repack held-out
  S 1.271->1.086 (68% of excess removed) vs wikitext only 28%. => Build B GO with
  CODE-DOMAIN calibration (~3-5% e2e on coding workloads); never calibrate on wiki.
  m-distribution: p50 54, p95 476, p99 1266, max 4028, zeros 4.7%; greedy
  M64/M32/M16 eta = 94.4% (80% rows via M64) - doc's tile policy CONFIRMED.
- P0.5 SASS facts: deployed maxwell_sgemm_fp16_128x32_tn (40.7k of 45k launches) is
  a 4x4-microtile 15%-FFMA-density kernel (99 FFMA/651 ins); 64x64 = 8x8/64-acc
  34% density. Custom 64-acc mainloop headroom SASS-verified.
Execution plan: ~/.claude/plans/qwen36-round3-execution.md. Next: Build A phases
A1+ (dense mainloop port), Build C kernel, B2 code-calibrated permutation impl.

Remaining cheap gates: combined
trace capture (EPLB replay + m-histograms), #25929 vectorized get_rows cherry-pick
(gather is 14% kernel time TODAY), D1a K=1 MTP A/B, cuBLAS ALGO pin sweep, nccl-tests
busbw (TCCL gate), decode-NCCL-share attribution from existing traces (gates Q2).
Related: [[p100-pp-research-round2-20260721]], [[qwen36-35b-moe-pp-execution]].
