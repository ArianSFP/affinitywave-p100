# Round-3 Phase 0 - measurement gates (2026-07-22)

Plan: ~/.claude/plans/qwen36-round3-execution.md. All runs watchdogged, prod env
(EP4+P1+P8a stack). Everything below is measured on-rig today unless noted.

## P0.2 PCIe topology + P2P probe (bench/p2p-probe.cu) - BUILD C TOPOLOGY GATE: PASS
- nvidia-smi topo -m: ALL SIX PAIRS PIX (single PCIe switch), one NUMA node.
- All 12 directed pairs: access=1, perfRank=0 (uniform), nativeAtomicSupported=0
  (confirms research: NO peer atomics over PCIe on P100 -- flag-store protocol only).
- Pairwise in-kernel peer-read BW: 12.9-13.0 GB/s every pair (= PCIe gen3 wire).
  Local HBM ref 535-551 GB/s.
- Peer-read dependent-load latency: 1.15-1.23 us/hop, uniform (local 0.19-0.23).
- One-shot contention pattern (all 4 GPUs concurrently reading all 3 peers):
  9.69-9.71 GB/s aggregate per GPU (~75% of pairwise) -- switch degrades all-to-all
  only mildly. Best-case topology for the custom one-shot allreduce.

## P0.3 nccl-tests comparator (files nccl-*.txt) - BUILD C LATENCY GATE: PASS; TCCL: CLOSED
- Small allreduce latency is FLAT ~30 us from 4KB to 64KB:
  auto f32 31.2-32.3 us; forced Ring/LL f32 29.9-31.0 us; Ring/LL bf16 (prod dtype)
  32.1-36.0 us. Auto already = Ring/LL class.
- vs measured P2P constants (1.2 us flag hop, ~10 GB/s contended reads): a correct
  one-shot kernel has ~10-15 us structural budget at 4-32KB => ~15-20 us/call saving.
- Large-message busbw (auto): 8.7 -> 9.9 GB/s (1MB -> 256MB) = 76% of the measured
  12.9 GB/s wire on a uniform single-switch topology. TCCL's wins come from
  pathological path selection on asymmetric topologies -> TCCL AXIS CLOSED.

## P0.5 cuBLAS SASS facts (maxwell_sgemm_fp16_*.sass, from libcublasLt 12.8.3.14)
- maxwell_sgemm_fp16_128x32_tn (sm_60): 651 instructions, 99 FFMA (15% density),
  67 XMAD + 56 IADD + 46 ISETP address/int overhead, longest FFMA run 13
  => 256 threads x 16 outputs = 4x4 register microtile. This kernel takes 40.7k of
  our 45k expert-GEMM launches (the small-m majority: p50 m = 46).
- maxwell_sgemm_fp16_64x64_tn (sm_60): 1304 instructions, 438 FFMA (34%), runs of
  56-57 => 8x8 microtile / 64 accumulators (Scott Gray lineage), 64 threads/CTA class.
- Conclusion for Build A: the deployed small-m path runs a LOW-density 4x4 kernel;
  an openai-class 64-accumulator mainloop has ~2.5x its arithmetic density. The
  custom kernel's mainloop headroom is SASS-verified, not just inferred.

## A0 CuAssembler toolchain gate (scratchpad/a0-gate) - PASS, PROVEN ON-RIG
Toolchain: github.com/gpuocelot/cuasm (CuAssembler fork; ships
DefaultInsAsmRepos.sm_60.txt) + venv(pyelftools, sympy). NOT RCAsm (does not exist
under the researched name); gpuocelot/cuasm handles CUDA 12.8 cubins natively.
1. nvcc -cubin -arch=sm_60 (CUDA 12.8) -> cuasm disassemble -> reassemble:
   BYTE-IDENTICAL cubin.
2. cuModuleLoad on driver 580 + launch: correct results, attrs intact (regs=8).
3. Hand-modified SASS body (FADD operand negation = semantic change to a-b):
   assembles, runs with correct new semantics, and the assembled cubin is
   BYTE-IDENTICAL to nvcc's own cubin for the equivalent C source.
=> The hand-SASS route for Build A is OPEN and validated end-to-end on this rig.
   Caveat found: default sm_60 instruction repo has good coverage but silent
   fallbacks are possible -- always verify edited encodings via cuobjdump diff
   (the modifier DID encode here; first-attempt confusion was operand mapping).

## P0.4 routing histograms (hist-wiki.txt, hist-code.txt; GGML_CUDA_MOE_HIST
   instrument added to ggml-cuda.cu csort fallback)
Wikitext (64 chunks x 4096 tok, 40 MoE layers, 2560 ubatch-layer observations):
- Current contiguous EP4 skew: mean 1.260, p95 1.505 (matches recorded 1.24).
- Load-aware R=0 repack (greedy 4-bin, calibrate first half, test second half):
  S -> 1.187 held-out (1.171 in-sample) = only 28% of excess skew removed.
- Hot-16 stability calib->test: 0.727 mean. Skew is mostly PER-BATCH BURSTS,
  not stable hot experts.
- m-distribution per (layer,ubatch,expert): mean 128, p50 46 (!), p90 315,
  p95 487, p99 1238, max 4028, zeros 7.5% -- far more concentrated than the
  research's gamma model.
- Tile padding efficiency (measured): M16 94.4%, M32 88.6%, M64 78.3%, M128 62.2%;
  greedy M64/M32/M16 = 94.4% with 82% of rows through M64.
Code domain (64 chunks of concatenated project sources) + cross-domain:
- EXPERT ROUTING IS EXTREMELY DOMAIN-SPECIALIZED: hot-16 overlap wiki vs code =
  0.016 (essentially DISJOINT hot sets). Wiki-calibrated repack on code removes
  only 9.5% of excess skew.
- WITHIN the code domain routing is highly STABLE: hot-16 overlap 0.892 across
  halves; code-calibrated repack held-out: S 1.271 -> 1.086 = 68.3% of excess
  skew removed (in-sample bound 1.077). Wikitext's 28% was the outlier (topically
  diverse corpus = bursty); the user's actual workload (code) is the good case.
- Combined m-distribution (2.62M observations): mean 128, p50 54, p95 476,
  p99 1266, max 4028, zeros 4.7%; greedy M64/M32/M16 eta = 94.4% (80% of rows
  via M64). Identical across domains.

## P0.1 decode NCCL share (decode-prod2.sqlite, prod stack, 192 greedy tokens)
- env -i gotcha: nsys injection vars are stripped by env -i -> empty trace; wrap
  with plain `env` instead (first trace decode-prod.sqlite is empty, kept as a
  reminder).
- Decode-only window (post-prefill): 80.4 allreduce calls/token (2 per MoE layer,
  ncclDevKernel_AllReduce_Sum_f32_RING_LL -- decode wire dtype is F32, 8 KB
  messages at hidden 2048; bf16 wire is prefill-only), avg 47.1 us/call
  (vs 30 us clean nccl-tests floor -> ~17 us is arrival wait/jitter).
- NCCL kernel time = 3.79 ms/token = 28% of the real 13.4 ms decode step.
- Note: compute kernels largely hidden inside CUDA-graph replays in this trace
  (busy % is an undercount); NCCL runs outside graphs so its numbers are exact.
- One-shot structural budget ~10-15 us/call (from P0.2/P0.3 constants) =>
  savable ~1.2-1.6 ms/token => +10-13% decode. GATE: PASS.

## H2 (D1a) K=1 MTP re-test - ATTEMPTED, DEFERRED (harness bug, no data)
llama-completion rejects --spec-type (spec flags are server/cli-only in this
fork); llama-cli with -f/-n 256/-no-cnv RAN AWAY (unbounded generation, ~2.6 GB
stdout per leg, no timings; all junk deleted, rig verified clean after).
Baseline leg that did complete via llama-completion: 73.85 tok/s decode
(matches prod 74.6). Re-run H2 via the deploy-ngram.sh llama-server + API
bench, or a smoke-tested llama-cli invocation. Gotcha recorded in the
rig-gotchas memory.

## Gate verdicts (FINAL)
- Build A grouped GEMM: GO. SASS toolchain PROVEN on-rig (A0: byte-identical
  round trip + semantically-modified body matching nvcc reference bit-for-bit),
  mainloop headroom SASS-verified (deployed small-m kernel is a 4x4-microtile
  15%-FFMA-density design), tile policy M64/M32/M16 at measured 94.4%.
- Build B R=0 repack: GO WITH CODE-DOMAIN CALIBRATION. Calibrate the per-layer
  pack on code traces; expect S 1.27 -> ~1.09 on coding workloads (~3-5% e2e
  prefill). Do NOT calibrate on wikitext (disjoint hot sets). Per-domain packs
  (or a workload-weighted mix) are the deployment model.
- Build C one-shot decode allreduce: GO on all three gates (topology ideal:
  uniform single-switch PIX, 12 pairs OK, no peer atomics as expected; NCCL
  ~30 us flat 4-64KB; decode share 28% of step). Projection +10-13% decode.
- TCCL: axis CLOSED (76% of wire already on uniform topology).

## A1 kickoff: isolated cuBLAS baseline at EXACT expert shapes (a1-cublas-baseline.txt)
Single-stream GemmEx fp16/fp16->f32 COMPUTE_32F TN, GPU0, bench/a1-gemm/baseline.cu.
Key rows (TFLOP/s, gate-up | down): m=32 2.4|2.5, m=46(p50) 2.3|2.5, m=128(mean)
4.1|3.9, m=190 4.1|4.4, m=1024 6.2|6.4. The trace-blended "~6 TF/s deployed" was
carried by mega-experts; AT THE MEAN/MEDIAN SHAPES cuBLAS is only 2.3-4.1 TF/s
(25-44% of peak). openai-gemm PCIe-scaled does ~7.0 @ m=128 -> the custom mainloop
target is ~1.7x cuBLAS at the operating point, NOT the research doc's 0.85x-of-
cuBLAS framing (that assumed dense-peak cuBLAS). Build A expected value UP.

## A1 mainloop progress (bench/a1-gemm/, 2026-07-22)
v1 naive single-buffer: 2.23 TF/s @ m=128. v2 (+float4 global staging, double
buffer): 3.00 @ m=128 (grid-limited: 32 CTAs), 4.79 @ m=1024 = 51% peak = 68% of
the openai-SASS PCIe target (7.0) - EXACTLY the documented CUDA-C-vs-SASS gap.
v3 float4 smem reads REGRESSED to 3.87 (stride mod-4 alignment vs mod-32 store
conflict spreading are mutually exclusive in this staging layout; kept as
m32v3-vecread-regression.cu).
v4 (smem union: sRed aliases dead staging buffers, 33KB -> 25KB -> 2 CTAs/SM):
5.84 @ m=1024 gate-up, 5.63 down-shape (+22%).
v5 (interleaved XOR-swizzled layout, LDS.128 reads; XOR term (k>>3)&3 is
per-thread constant inside a K-stage so alignment AND bank spread coexist -
the fix v3 could not reach with padding): 5.97 gate-up / 6.27 down @ m=1024.
SASS-verified: mainloop body 639 instr, 512 FFMA = 80% static density, 32
LDS.128, zero LDS.32 in loop. Remaining gap to peak = latency stalls, not mix.
v6 (KSTAGE 16 -> 12KB smem -> 4 CTAs/SM): REGRESSED 4.15 (2x sync frequency
costs more than 16 warps hide). CUDA-C PLATEAU = v5: 5.97/6.27 = 64-67% peak,
~1.5x cuBLAS @ m=1024, ~2.4x @ p50 operating shapes.
fp64 sampled reference (4096 dots, in-harness): custom maxrel 4.2e-4 vs cuBLAS
1.1e-3 @ m=1024 - the custom kernel is MORE accurate than cuBLAS COMPUTE_32F
(split-K-4 shortens accumulation chains). "bad vs cublas" counts are cases
where cuBLAS is farther from fp64 truth. Accuracy gate CLEARED.
DECISION: park SASS polish (caps at +17% GEMM slice = +3-4% e2e); proceed to
A2 grouped-launch bench vs per-expert cuBLAS loop on REAL m-distributions
(hist-code.txt) - grouping carries the e2e prize and dissolves small-m
underfill (m=128: 3.18 solo -> ~6 effective when 8k tiles fill the device).

## A2+A3 grouped-launch bench on REAL routing (a2-group.cu, 2026-07-22)
blk.20 code-domain hist, EP4 slice 0: Mtot=7735 rows, 61 live experts,
275 M32 tiles. Grouped = ONE launch, device tile table (expert,row0,rows),
v5 mainloop, FUSED index-gather (A3: ids resolved per thread at kernel
start, ~1 extra LDG per CTA-thread). Baseline = gather kernel + 61
per-expert GemmEx COMPUTE_32F (isolated-cuBLAS-honest: 3.98-4.07 TF/s eff,
matches a1-cublas-baseline at these shapes).
- gate/up (K=2048 N=512, gathered): 4076.6 -> 2547.0 us = 1.60x (6.37 TF/s
  effective INCLUDING M32 padding waste and gather)
- down (K=512 N=2048, contiguous input): 3987.4 -> 2896.0 us = 1.38x
- layer-weighted (2x gate/up + down): ~1.52x on the expert-GEMM+gather slice
- verify: maxrel 1.7e-2 order-rounding class vs cuBLAS loop, 0 bad @ 2e-2
NEXT: A4 fused Q8_0 (exact-semantics load-time repack: 16B-aligned qs plane
+ d plane, same bytes/values; dequant d*q in fp32 during staging = EXACT,
more accurate than prod's fp16-rounded dequant; halves B global traffic).

## A4 fused-Q8_0 grouped bench (a4-q8.cu planes, a4b-q8native.cu native)
Baselines now PROD-SEMANTICS: dq (per-call weight dequant Q8_0->fp16 + gather
+ per-expert GemmEx) and nodq (lower bound, no dequant pass). blk.20 slice 0:
- a4 PLANES (qs plane + d plane, LDG.64 qs loads):
  gate 6.10 TF/s eff = 1.76x vs dq / 1.48x vs nodq
  down 5.40 TF/s eff = 1.55x / 1.30x
  robustness (blk.1, blk.39, slices 2-3): 1.63-1.89x vs dq everywhere.
- a4b NATIVE ggml layout (packed 34B blocks, LDG.16 qs loads, NO repack,
  NO extra VRAM, MMVQ untouched): gate 5.04 = 1.43x/1.22x, down 4.76 =
  1.40x/1.14x. Native-read penalty vs planes: 17-21%.
- ACCURACY (fp64 exact-Q8 sampled, 4096 dots): custom maxabs err 3.0e-5 /
  7.4e-6 vs cublas-fp16-dequant 9.9e-3 / 5.1e-3 -> fused kernel is ~300x MORE
  accurate than the deployed path (d*q in fp32 is exact; prod rounds to fp16).
  All grouped-vs-cublas "mismatches" are baseline fp16-dequant rounding at
  near-zero outputs (uniform scatter, delta ~0.004 = K*eps_fp16 class).
- DECISION: integrate NATIVE variant first (zero VRAM/decode risk, 1.4x);
  planes = A4c follow-up via IN-PLACE repack (34B block = 32B qs + 2B d,
  byte-count identical) + MMVQ addressing patch, gated on decode byte-identity.

## A-INTEGRATION (2026-07-22): GGML_CUDA_MOE_GROUPED in the fork
New files ggml/src/ggml-cuda/moe-gemm.cu/.cuh (kernel = m32v5 mainloop +
a4b native-Q8 staging). Seam: ggml_cuda_mul_mat_id else-branch - when
moe_f16_direct && Q8_0 && ne00%32==0 && ne0%64==0 && n_local>0, the
per-expert dequant+GemmEx loop is replaced by ONE launch on a host-built
tile table (from tokens_per_expert, already host-resident). A = src1_sorted
(existing gather kept for v1; A3 in-kernel gather is follow-up), C =
dst_sorted same row order; scatter/ZERO_ROW/EP semantics downstream
untouched. DECODE COMPLETELY UNTOUCHED (mul_mat_id exits into MMVQ at
ne2<=8 before this seam). Not byte-identical to loop (split-K order + exact
fp32 dequant, more accurate) -> ppl gate.
Smoke: pp512 grouped ON = 1189.85, clean run. A/B 3-round pp2048/8192 + ppl
8-chunk both arms: bench/grouped-ab.sh -> grouped-20260722/.
A/B RESULTS (rounds 2-3, r4 each):
- ppl GATE PASS: off 5.8332 +/- 0.10814, on 5.8312 +/- 0.10808 -> delta
  -0.0020 (inside +/-0.003, and DOWN = consistent with exact dequant).
- pp2048: 1523.9 -> 1602.7 = +5.2%. pp8192: 1478.0 -> 1493.9 = +1.1% (!).
- pp8192 gap vs microbench (~8% expected): ub4096 wall is NOT GPU-bound -
  the recorded ~28% host-drain idle absorbs kernel-time wins. pp2048-on
  (1602.7) BEATS pp8192-on -> the ub4096 optimum (chosen for the old
  per-expert cuBLAS path) may have shifted down: small-m tiles are now
  cheap, and smaller ubatches shrink the host-side drain per step.
- FOLLOW-UP: bench/grouped-ub.sh ub sweep 2048/3072/4096/8192 grouped-on.
UB SWEEP (pp8192): on-ub2048 1432, on-ub3072 1474, on-ub4096 1488, off-ub2048
1366 -> ub4096 STAYS optimal (grouped helps more at small ub but not enough
to shift the knee). ub8192 ASSERTS (meta.cpp:1558 bufs.back()!=nullptr) WITH
AND WITHOUT grouped = pre-existing compute-buffer OOM at ub8192+pp8192, not
a grouped bug (verified both arms standalone).

## NSYS attribution ON vs OFF (grouped-pp8192/off-pp8192.sqlite, dev0/pass)
OFF: busy 73.3%. Expert GEMM+dequant = 22.4% of wall in ~48k launches/pass
  (maxwell 128x32 751ms + 64x64 533ms + expert-share of 32x128 ~981ms +
  dequant_q8_0 168ms + splitKreduce 137ms). NCCL bf16 18.9%, get_rows 12.6%.
ON:  busy 73.8%. moe_gemm_q8_grouped = 16.2% (1832ms, 240 launches/pass).
  NCCL 19.9% (now #1 kernel), get_rows 12.8%, dense 32x128 8.8%.
=> kernel-time saved 739ms/GPU/pass = 6.4% of wall, but wall shrank only
1.8% and busy% is FLAT: the saving became IDLE. UB4096 PREFILL WALL IS
HOST-SUBMIT-BOUND. Any further kernel-seconds cut (A4c planes -20% GEMM,
SASS +17%, A3 gather-fusion 12.7% slice) pays ~25% of face value until the
host drain is fixed. Grouped kernel vs cuBLAS-class kernels it replaced:
1832 vs ~2402ms = 1.31x kernel-only, matching a4b microbench (consistency).
NEXT LEVERS (prefill, in value order): (1) host-side op-count reduction -
A3 fused gather kills 120 get_rows launches/pass AND ~half the 12.8% slice;
MoE-block op fusion; graph-capturable prefill (needs device-side tile table
+ sort). (2) NCCL overlap/compression (P5 Q8 allreduce still POLICY
DEFERRED). (3) A4c planes / SASS polish only AFTER host-bound is broken.

## A6 DEVICE PLAN (2026-07-22, reviewer-endorsed A6/A7 direction)
Sync-free mul_mat_id path, env GGML_CUDA_MOE_PLAN (moe-gemm.cu [TAG_MOE_PLAN]):
- moe_plan_build (1 CTA): smem histogram over local experts + scans + M32
  tile-table emit + cursor init + n_local/n_tiles scalars (device-resident).
- moe_plan_scatter (fixed grid): global-atomic cursors, to_sorted = token
  index (UNSTABLE within expert = safe: each output row depends only on
  (token,expert); from_sorted is the exact inverse; non-local -> ZERO_ROW
  at the CAPACITY slot n_assign, shape-static).
- moe_gemm_q8_plan: persistent fixed grid (2*nsm CTAs) grid-striding over
  m_tile x n_tile read from the device plan; in-kernel A3 gather from the
  ORIGINAL f32 src1 (no f16 gather pass, activations unrounded); same
  swizzled Q8_0 mainloop. 144 regs, 0 spills, 2 CTAs/SM kept.
- Inverse scatter: existing get_rows_cuda, already capacity-launched.
- Workspace: persistent per-(device,slot) cudaMalloc (NOT pool - addresses
  must survive graph replay; Q8_1_DEDUP lesson), ~300KB more or less.
- NO ids D2H, NO stream sync, NO host-data-dependent launches -> A7
  graph-capture eligible; n_local==0 is a normal execution (no zero-grid).
- Fallbacks: HIST/DEBUG_STATS envs, asyncep, non-Q8_0, ne11!=1 -> host path.
VALIDATION: GGML_CUDA_MOE_PLAN_CHECK harness vs host reference on live
routing: 64/64 PASS (counts, scalars, tiles, from/to_sorted inverse +
uniqueness), coherent greedy output, clean exit.
FIRST A/B EXPOSED PARTIAL ENGAGEMENT: down projection fell back (its src1
is the GLU output with ne11=n_expert_used; the ne11==1 gate excluded it),
so ONE host sort + drain per layer REMAINED (plan-mode gate/up no longer
prime the mcache, so down missed the cache and paid the FULL D2H+sync).
Even so: pp2048 +6.8% (1715), pp8192 +6.3% (1586); nsys: window 11.28 ->
10.52s, D2H 321 -> 161, cudaStreamSynchronize 1496 -> 856.
FIX: general ne11 (to_sorted = i12*ne11 + iex%ne11, eligibility requires
contiguous dim-1/2). Re-validated 64/64 PASS with gate/up/down ALL on plan.
A6 COMPLETE RESULTS (3-round interleaved, r4):
- pp2048: 1601.8 -> 1753.5 = +9.5%; pp8192: 1490.8 -> 1654.5 = +11.0%
- ppl: base 5.8332, plan 5.8318 (delta -0.0014, in band, noise family)
- cumulative vs pre-Build-A prod (1478): +11.9% pp8192, +15% pp2048
The host-drain diagnosis is CONFIRMED BY INTERVENTION: removing the per-
layer ids D2H sync bought 7x what the kernel-time cut alone bought.

## A7 GRAPH CAPTURE (2026-07-22): BUILT + VERIFIED, FLAT - env-gated OFF
Change: [TAG_MOE_PLAN_GRAPHS] - the MUL_MAT_ID large-batch graph-compat
blocker is relaxed when ggml_cuda_moe_plan_node_supported(node) (mirror of
runtime plan eligibility + PLAN_CHECK/ASYNCEP excluded + opt-in env
GGML_CUDA_MOE_PLAN_GRAPHS). Capture-safety: workspace growth guarded by
cudaStreamIsCapturing assert (growth only ever on uncaptured warmup evals).
MEASURED (nsys a7-pp8192.sqlite): capture CONVERGES exactly as designed -
648 chunk-graphs/eval (162/device; meta backend already chunks at NCCL
boundaries so collectives stay outside), all captured on pass 2, pass 3+ =
PURE REPLAY, zero re-capture churn, streamSync 856 -> 360. One-time cost:
cudaGraphInstantiate 8.86s total (~13.7ms/graph, the pre-Ampere node-count
scaling the reviewer flagged) - amortizes to zero in a long-lived server.
A/B (r12 x 2 interleaved rounds, instantiation dilution ~1.2% AGAINST the
graphs arm): plan 1654.6/1654.4 vs plan+graphs 1653.5/1654.9 = FLAT.
WHY: A6 removed the dependency STALL; the P1 submit threads had spare
throughput, so batching submissions into replays saves host cycles the
host no longer needed. Same zero-sum shape as Stage-D TBO, on the submit
axis. Post-A6 idle (~25%) = NCCL arrival skew + true dependencies, which
replay cannot touch. VERDICT: keep default-OFF (adds a working mechanism
for any future host-cost growth; zero risk while off). Reviewer's +2-7%
was still optimistic for this rig because A6 consumed the host headroom.
NEXT LEVERS (post-A6 reranked): NCCL slice 20% (P5 Q8-wire = POLICY
DEFERRED to user; arrival skew -> EPLB/B2 code-calibrated repack ~3-5%);
A4c planes repack + SASS polish (kernel-seconds now pay closer to face
value at 75% busy: ~2-3% each); H2 K=1 MTP re-test (decode side).

## ADOPTION (2026-07-22): GGML_CUDA_MOE_GROUPED=1 added to deploy-35b-moe-ep.sh
All gates green: ppl PASS (-0.002, accuracy UP), never slower (+1.1% pp8192
/ +5.2% pp2048), decode untouched by construction, ep-soak PASS 20/20 clean
exit zero CUDA errors, VRAM drift 256MiB normal. Ladder: 539 -> 659 -> 952
-> 1217 -> 1460-1488 -> 1470-1521 (P8a) -> ~1494-1603 (grouped, shape-dep).

## ITEM 1a: p-fold (down-GEMM + routing-weight MUL fusion, 2026-07-22 late)
[TAG_MOE_PLAN] try_fuse case {MUL_MAT_ID, MUL} -> moe_plan_exec with rw:
epilogue multiplies each output row by weights[to_sorted[row]] (for the down
matrix to_sorted IS the assignment index = flat weights index) BEFORE the
inverse scatter. Guards: large-batch fallback only (decode MMVQ untouched),
EP only, weights layout checked, GGML_CUDA_MOE_PLAN_NOFOLD kill-switch.
GATES: greedy BYTE-IDENTICAL fold vs nofold (sha256 equal); bench 2-round
interleaved r8: pp8192 1641 -> 1686 (+2.8%), pp2048 1745 -> 1796 (+2.9%)
(also removes the full [2048,8,N] tensor read+write, not just the multiply);
soak PASS 20/20. ADOPTED (active whenever PLAN is on).

## ITEM 1b: EPLB code-calibrated expert permutation (2026-07-22 late)
[TAG_MOE_EPLB] load-time permutation in meta set_tensor (env
GGML_CUDA_MOE_EPLB_MAP=<file>): permutes ffn_*_exps along the expert axis +
ffn_gate_inp output rows into a host scratch BEFORE split dispatch, so the
EP4 contiguous slices become calibrated balanced bins; routing ids live in
the permuted space, runtime code untouched. Map: bench/eplb-map-code.txt
(greedy 4-bin cap-64 from hist-code.txt aggregates; in-sample slice skew
1.254 -> 1.000 mean, 1.646 -> 1.003 max). No exp_probs_b in this GGUF.
GATES: engagement WARN line; greedy BYTE-IDENTICAL map on vs off; bench:
SYNTHETIC llama-bench prompts FLAT/-0.3% (INVALID test - hot sets are
domain-disjoint, llama-bench tokens are not code); REAL CODE prompts
(llama-completion 14.6k-token source prefill, 3 interleaved rounds):
1563/1562/1566 -> 1586/1586/1585 = +1.4% consistent. ADOPTED in deploy
(this rig serves code). Held-out gain sub-linear vs skew fix (S 1.25->1.0
in-sample) - post-A6 the wall's skew sensitivity is partially hidden in
dependency idle. LADDER (code-prompt basis differs from synthetic):
synthetic pp8192 ~1686 (p-fold), code-prefill ~1585 (p-fold+EPLB).

## ITEM 2: AR_P2P copy-engine allreduce (2026-07-22/23)
[TAG_AR_P2P] allreduce.cu: peer-DMA permutation-round allreduce for large f32
boundary tensors (env GGML_CUDA_AR_P2P=1, threshold 4MB - decode 8KB stays
NCCL). bf16 wire; reduce-scatter into per-owner inboxes; owners sum 4 bf16
shards in f32 (ONE rounding, fewer than NCCL's bf16 ring); allgather DMAs
reduced shards directly into peers' wire buffers (dead slots, race-free via
the host-serial event chain - no flags, no peer atomics). Duplex probe
(scratchpad duplex-probe.cu): 13.2 uni / 12.51 duplex / 12.52 ring-round
GB/s per link = near-perfect full duplex, permutation rounds at full rate.
BUG FOUND BY THE PPL GATE: v1 summed all 4 tensors blindly; the NCCL
wrapper ZEROES devices without GGML_TENSOR_FLAG_COMPUTE (inactive shards).
Summing their uninitialized memory produced degenerate logits ("negative
standard deviation") on full-ubatch eval - INVISIBLE to greedy smoke (only
last position + decode validated). Fix: memset wire[i] for non-COMPUTE.
Post-fix gates: ppl 5.8335 PASS (in band); pp2048 1789 -> 1805 (+0.9%),
pp8192 1684 -> 1697 (+0.8%), both rounds consistent; smoke coherent.
v2 upside NOT yet taken: chunked pipelining (skew tolerance like NCCL's
ring + copy/compute overlap into panels). v1 barrier-like stage waits
amplify arrival skew; the wire-speed win (12.5 vs 9.9) shows anyway.

## ITEM 3: fp16 HFMA2 kill-gate microbench - KILLED (2026-07-23)
bench/a1-gemm/m32h2.cu: m32v5 structure, half2 smem, HFMA2 mainloop with
lane-parallel k-striping (2 chains of 16 per 32-block = S2-tree class) and
per-32-block fp32 flush - the reviewer's EXACT proposed geometry ("Q8 block
size should remain 32"). 180 regs, 0 spills, 2 CTAs/SM. MEASURED at m=1024:
5.89 / 6.27 TF/s vs fp32 mainloop 6.64 / 6.28 = 0.89x / 1.00x. KILL
THRESHOLD 1.55x: FAILED BY A WIDE MARGIN. Mechanism as predicted: the
halved HFMA2 issue count is consumed by the block-32 flush (convert + adds
every 16 FMAs per output) + non-vectorized half2 smem reads. The Q8 block
structure FORCES the 32-flush for exact scales -> the fp16 program's
central premise fails on sm_60 at the mandated granularity. (Variant has an
index bug - outputs wrong - but the instruction stream is the intended mix;
speed verdict stands.) Per the reviewer's own criterion ("below 1.55x the
full program cannot plausibly reach 4000"): fp16-GEMM axis CLOSED for this
project, now with a direct measurement of the proposed structure. Consistent
with the 2026-07-14 splitk-hgemm probe and the p100-fp16-half2-refuted class.

## ITEM 2 CLOSE-OUT (2026-07-23): AR_P2P ADOPTED; EPLB OOM bug found+fixed
Second bug found by the soak gate: the EPLB full-tensor host scratch (285MB
transient per exps tensor) OOM-KILLED the --no-mmap server load (model
already fills the 45GB host; journalctl: llama-gguf killed at 29.3GB RSS).
The completion-based EPLB tests used default mmap and never saw it. FIX:
scratch-free per-expert gather inside the AXIS_2 set_tensor branch (source
expert perm[slot] copied straight from the loader buffer to the device
slice; scratch kept only for the 2MB router tensor). Verified BYTE-IDENTICAL
output vs the scratch path. Full-stack soak (GROUPED+PLAN+p-fold+EPLB+
AR_P2P, --no-mmap): PASS 20/20, clean exit, 0 CUDA errors. AR_P2P and the
fixed EPLB both ADOPTED in deploy-35b-moe-ep.sh.
Adopted-stack ladder: synthetic pp8192 ~1697, pp2048 ~1805; real-code
prefill ~1585-1600 class; decode 74.6 untouched.
