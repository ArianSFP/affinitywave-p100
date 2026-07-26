# Round-3 execution plan - 4xP100 Qwen3.6-35B-A3B (2026-07-22)

Plan of record following research round 3 (all 3 questions answered + assessed vs rig
traces; see memory p100-pp-research-round3-20260722.md). Baseline: pp8192 ~1475-1520,
tg128 74.6 (ngram+MTP stack ~2x on workloads). Bars: byte-identical OR ppl within
+/-0.003 (KLD-pair methodology); NEVER sub-Q8 weights; watchdog on every 4-GPU run;
env -i + taskset 0-11 bench pattern; no git commit/push without user approval.

## Measured basis (from ctrl-d4.sqlite, GPU0, 2 evals pp16384)
- Expert GEMM deployed ~5.5-6.4 TF/s (60-70% PCIe peak) via maxwell_sgemm_fp16_
  {128x32,64x64}_tn + splitKreduce (43.9k launches) + gemv2T leakage.
- mmid gather (k_get_rows_float) ~1.45 s/eval == expert-GEMM-sized; dequant 1.7%
  but 26k launches; per-expert fixed cost ~60-100us/layer (19us kernel floor).
- NCCL 26% kernel time (wire ~13%, rest arrival wait; skew 1.24); FA ~10%; GDN ~8%.
- Steady busy 80-85%; TBO proved idle yields ONLY to fewer launches/kernel-seconds.
- Wire is already bf16 (AllReduce_Sum_bf16_RING_LL).
- Topology (2026-07-22): ALL SIX PAIRS PIX, single switch, one NUMA node =
  best case for P2P all-to-all custom collectives.

## Phase 0 - combined measurement session (gates everything; ~half a day)
- P0.1 decode-NCCL share: mine EXISTING decode nsys traces (offline, no GPU).
  GO gate for Build C: decode allreduce+wait >= ~10% of step time.
- P0.2 topology probe: topo -m DONE (all PIX). Remaining: 12-directed-pair
  cudaDeviceCanAccessPeer + P2P attrs (expect native atomics = NO) + in-kernel
  peer-load latency/BW microbench (small single-purpose CUDA program).
- P0.3 nccl-tests: build if absent; all_reduce_perf -b 4K -e 64K auto AND forced
  NCCL_ALGO=Ring NCCL_PROTO=LL (Build C comparator; per-size us table);
  plus large sizes (1M-256M) busbw vs PCIe gen3 wire (~13 GB/s class) = TCCL gate
  (if NCCL >= ~70% wire, close TCCL axis for good).
- P0.4 routing capture: add env-gated GGML_CUDA_MOE_HIST=<path> dump in the csort
  mmid fallback (per-call 256-expert counts, dev0 only, host side already has ids
  via P8a pinned staging). Run few hundred ubatches over representative corpus
  (wikitext + code). Feeds BOTH: (a) grouped-GEMM m-histograms -> tile mix policy
  + real eta_T table; (b) EPLB replay -> skew stability across windows + R=0
  4-bin pack simulation.
- P0.5 SASS facts: cuobjdump/nvdisasm the two maxwell kernels from libcublas
  (offline) -> microtile/K-stage/dual-buffer facts to set Build A gates.
- P0.6 (optional) CUTLASS 2.11 sm60 SIMT grouped compile smoke (CPU-only).

## Build A - grouped fused MoE GEMM (prize +20-28% pp; floor +10-14%; effort L)
The only remaining large prefill lever (TBO/AsyncEP/fusion classes all closed).
- A0 toolchain gate (hours): CuAssembler bundled in RCAsm on CUDA 12.8:
  nvcc -cubin -arch=sm_60 trivial kernel -> cuasm disassemble -> reassemble
  UNCHANGED -> cuModuleLoad -> run -> identical output; then modified-body variant;
  verify reg/smem attrs via cuFuncGetAttribute. PASS -> SASS route open.
  FAIL -> plan B: old-toolkit ptxas scaffold in container; else CUDA-C floor only.
- A1 dense mainloop: CUDA-C port of openai-gemm 32x64x32 NN dataflow (128 thr,
  64 acc/thread, 4-way intra-CTA split-K, K-stage 32) with contiguous fp16 A/B.
  Measure vs deployed 6.3 TF/s. SASS pass bar (PCIe-scaled): ~6.5-7.0 TF/s @
  m=128. CUDA-C outcome ~40-50% peak still viable (floor path).
- A2 device scheduler: tile_prefix from csort counts (one scan kernel), fixed
  persistent grid, M32+M16 greedy (M64 later). Gate: <=2-3% overhead K=2048,
  <=5% K=512 (chunk N-tiles if not).
- A3 indexed-A fusion (kills the gather pass ~14% kernel time). Gate: <=5% core
  regression AND end-to-end mmid slice improves.
- A4 fused Q8_0 dequant. Default mode = exact-current-semantics (int8->half round
  ->f32). Gate: <=8-12% vs fp16-B core, no spills, coalesced.
- A5 M64 bulk + dual gate+up (shared A) if profiling justifies.
- All gates ppl/KLD class (reduction order changes; byte-identity impossible).
- Engage only at mmid ne12 >= threshold (decode untouched), env-gated, default off
  until soak.

## Build B - R=0 load-aware expert repack (~4-6% pp; effort S-M; independent)
- B1 (offline, after P0.4): replay per-layer histograms; DeepSeek greedy 4-bin
  pack with measured-time weights (a+b*m model, a~60-100us); simulate S_new per
  window. GO if stable hot-set across windows AND S_new <= ~1.08. NO-GO if skew
  is per-batch bursty (static placement can't help).
- B2: load-time per-layer permutation of expert order + ffn_gate_inp rows
  (logical=physical; split stays a partition; zero kernel changes). Gates:
  ppl/KLD (partial-sum regrouping => not byte-identical) + pp bench + tg neutral.

## Build C - one-shot P2P decode allreduce (+8-15% tg; effort M; gated on P0.1-P0.3)
- Topology gate PASSED (all PIX). Remaining gates: P0.1 share >= ~10%; P0.3 shows
  NCCL small-msg latency high enough to beat (>= ~20us at 4-32KB).
- Design (from Q2 research, repaired protocol MANDATORY): one-shot only initially;
  canonical rank order 0,1,2,3 + FP32 accumulation of bf16/f16 (MORE accurate than
  current bf16 ring, bitwise across ranks, deterministic); one-writer uint32 epoch
  flags, separate start/end banks, incrementing epochs; all-producer-thread
  membar.sys before publication, consumer membar.sys after wait; final lifetime
  barrier; 1-8 CTAs at 4-64KB; device-side epoch derivation (CUDA graph replay
  compatible); direct peer pointers (single process, no IPC); sm_60: NO peer
  atomics over PCIe ever. Env-gated opt-in; NCCL stays default + fallback.
- Integration: meta backend comm layer (alongside NCCL proc addresses); engage
  only for small messages (<= ~64KB) on the primary stream.
- Gates: bitwise all-ranks identical, delay-injection stress (their 7.6 list),
  ppl/KLD (order differs from NCCL), tg A/B interleaved r>=3, ep-soak before prod.

## Build D - R=8 EPLB replication (+1-3%; ONLY after Build A; needs partition->
  replica surgery + side buffers; today's fixed cost makes replicas net-negative).

## Harvests (interleave; each S)
- H1 #25929 vectorized get_rows cherry-pick + A/B (gather 14% kernel time NOW;
  superseded later by A3 but cheap immediate win; also helps GDN gathers which
  A3 does NOT cover).
- H2 D1a K=1 MTP re-test (--spec-draft-n-max 1, config only; -9.5% reject was n4).
- H3 cuBLAS ALGO0..23 pin sweep at expert shapes (env-gated pin; superseded by A).

## Session sequencing (this session)
1. Coordination check + plan doc (this file) + memory update.        [done first]
2. P0.1 offline trace mining + P0.5 SASS dump (no GPU).
3. P0.2 P2P probe + P0.3 nccl-tests (GPU, short, watchdogged).
4. P0.4 instrument (small code change) + capture run (GPU, watchdogged).
5. B1 pack simulation offline from capture.
6. A0 CuAssembler round-trip gate.
7. Time permitting: H1/H2.
Decisions recorded in results/qwen36-35b-moe-pp-20260721/round3-phase0/.

## STATUS 2026-07-22 (Phase 0 executed same day)
ALL GATES GREEN - see results/qwen36-35b-moe-pp-20260721/round3-phase0/RESULTS.md.
A0 PASS (gpuocelot/cuasm proven on-rig); Build A GO; Build B GO with CODE-domain
calibration (wiki/code hot sets disjoint!); Build C GO (28% decode share, ideal
topology); TCCL closed. Next session: A1 dense mainloop port (target 6.5-7.0 TF/s
@ m=128), Build C one-shot kernel, B2 code-calibrated permutation.
