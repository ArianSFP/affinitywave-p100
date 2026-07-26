# Qwen3.6-27B on 4xP100 - post-MTP performance plan ("round 3")

**Date:** 2026-07-14 * **Prepared from:** 3-track research audit (results ledger, code map, upstream delta) + rig checks.
**Objective:** increase realized performance across all 4 GPUs for the accuracy-first coding workflow, on top of the ~53 tok/s lossless MTP compose result.

---

## EXECUTION STATUS (updated 2026-07-14 late)

- **P0 - DONE PASS.** Compose build validated for production, all 6 checks pass. `results/mtp-production-20260714/` (RESULTS.md + `deploy-production.sh`). Findings: byte-identical/lossless; COMPUTE_32F free under MTP; `--cache-reuse` unsupported by the hybrid context (omit it); prompt cache works for the append workflow; tool-calls OK. Decode 48 tok/s on a 67.4%-acceptance prompt (53 was 76.2%); soak 40-50 tok/s.
- **P1 - DONE PASS.** nsys attribution: `results/mtp-attribution-20260714/`. **The MTP step is COMPUTE-bound**, GPU >=80.5% busy, real host residual only ~6.5 ms/step (~8%), NOT 26-40 ms. Node-level budget: **verify batch-5 matvec = 58.8% of decode and runs GENERIC untuned MMVQ params**; NCCL 10.1% (HW optimum); draft matvec 7.1%; small-kernel pool (norm/quantize/residual/get_rows) ~8%; FA ~2%; context-independent (depth 0 ~ 7900). **Re-ranking:** P2 = confirmed primary; P3 = bounded ~8% ceiling; **P4 = deprioritized** (per-collective host gaps only ~1.7 ms/step).
- **P2 - DONE PASS, ADOPTED.** `results/mmvq-batch-20260714/`. Extended `MMVQ_PARAMETERS_PASCAL_SM60` to batch>1; isolated sweep found **rows_per_block=6 (nwarps=2)** for Q8_0 ncols_dst 5..8 (batch-5 verify matvec was ~2x its batch-1 per-byte cost). **+7.1% end-to-end MTP decode (48.05->51.46), lossless (byte-identical), plain-neutral, numerically correct** - 3 interleaved pairs. New production build `llama.cpp-q36-mmvq-batch/build-p100-mmvq-batch` (compose config + this tuning); `deploy-p2.sh` supersedes P0's deploy.
- **P3 - MEASURED, LOW-VALUE (closed).** `~/profiles-p100/scripts/mtp-residual/FINDINGS.md` (user-built sudo CPU-sampling tooling). The ~6.5 ms/step host residual is **~53% `cudaStreamSynchronize` spin-wait** (host waits ON the GPU -> confirms GPU-bound; **P3.4 overlap NOT justified**, nothing to overlap) + **~15% greedy sampling** (full ~150k-vocab sort+sample when only argmax needed -> P3.3, but only ~1.5% wall-clock). No major lever here.
- **P9 - MODELED, KILLED.** `results/tree-mtp-model-20260715/`. Measured per-depth acceptance (q~0.73 flat, no cliff) + verify-batch cost curve (compute-bound, +15%/column) -> **every tree shape is NEGATIVE vs linear n4, even at optimistic r=0.8** (-9 to -22%); even a hypothetical free-verify tree only ~break-even. Tree-MTP loses on P100 because verify is compute-bound (not the batch-invariant "free verify" it needs). Modeling pre-step prevented a losing multi-day build.
- **P7 (Q6_K) - PERMANENTLY CLOSED.** User (2026-07-15): never sub-Q8, not even to measure ([[qwen36-never-sub-q8]]). Do not revisit.
- **P5a - DONE PASS, ADOPTED (2026-07-15, autonomous).** Harvested upstream **683f0c72e** branchless-MMVQ writeback (compile-time accumulator index -> no local-memory spill at batch>1) onto the P2 build. **+8.4% MTP decode over P2 (51.46->55.79, +15.6% over compose), byte-identical/lossless**, plain-neutral, soak-clean, correct; holds at depth (53.3 tok/s @7900). `results/branchless-mmvq-20260715/`. build-p100-mmvq-batch rebuilt in-place = now (2,6)+branchless; deploy-p2.sh launches it. P5b re-sweep: (2,6) still optimal on the branchless kernel. Re-profile: verify matvec 58.8%->51.6%, no new safe big lever (NCCL closed; cuBLAS gemmSN_TN 7.7% is architectural/accuracy-locked; small-kernel pool ~12% only via invasive P6d fusion).
- **LADDER (all lossless):** 26.45 plain -> 33 winner -> 48.3 compose (P0) -> 51.5 (P2) -> **55.8 (P5a)** = **2.11x plain, +69% over the non-MTP winner.**
- **REMAINING (smaller/riskier):** P6d small-kernel epilogue fusion (~5%, invasive - isolate+validate before touching prod); P3.3 greedy-argmax (~1.5%, host-side); P5-proper rebase (robustness). No large safe decode lever remains after P5a.

---

## 1. Ground truth (measured, sourced)

**Throughput ladder** (decode tok/s, Q8_0, 4xP100 `-sm tensor 1/1/1/1 -fa on`):

| config | decode | pp8192 | source |
|---|---:|---:|---|
| plain beac5309f build | 26.45 | - | mtp-20260714 |
| winner + 3 gates (`build-p100-pascal-graphs-dedup`) | 33.2-33.6 | 723-730 | phase5a/8a |
| winner + `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1` (accuracy mode, adopted) | 33.4 | **451** | accuracy-mode-20260714 |
| **MTP compose n4 (`build-p100-nccl-mtp-compose`)** | **53.0 lossless** (n4 byte-identical) | - | mtp-20260714 |

**Non-MTP decode budget** (phase4r, 30.3 ms/token wall): MMVQ matvec 19.1 ms (63%; measured **82% of the Q8_0-layout read ceiling** - kernel-level improvement refuted across 10+ experiments, CLOSED), NCCL 4.5 ms (128 collectives; ~3.8 floor + ~1.3 host-side skew), rms_norm 2.0 ms (209 launches), GDN+elementwise ~4.7 ms (~370 launches, incl. the already-fused `gated_delta_net` op), get_rows 1.1 ms (97), residual quantize ~0.9, FA 0.8-1.0 (16). GPU ~93% busy under graphs - host is hidden at batch-1 decode.

**MTP production math - the open gap:** 53.0 tok/s x 4.0 tokens/step (measured acceptance n4: 192 accepted + 64 target per 256) = **75.5 ms/step**. Estimable GPU work: verify forward (batch-5, same weight bytes as batch-1) ~30-37 ms + 4 draft forwards (1 MTP block each, 2 allreduces each) ~6-12 ms => **~26-40 ms/step unattributed** (host spec-loop, sampling, graph rebuild churn, sync). All 4 GPUs are idle during that residual. **No MTP-mode trace exists** - every budget above is non-MTP. This is the largest un-mined signal in the project.

**Code facts that shape the plan (verified 2026-07-14):**
- MTP verify (batch 5) **stays on MMVQ** (`MMVQ_MAX_BATCH_SIZE=8`; MMQ disabled on cc 6.0 < DP4A 6.1; crossover to cuBLAS is at batch 9) - but the winner's SM60 table (`GGML_CUDA_SM60_MMVQ_*`, +11.2% at batch 1) **only applies at `ncols_dst==1`; batches 2-8 run GENERIC params.**
- Decode allreduce = 2/layer x 64 = 128/token (verify payload ne=5120*5 still takes the f32 smallpath, threshold ne<262144). Draft steps add 2 collectives each (MTP block wo + ffn_down; eh_proj/enorm/hnorm/head are MIRRORED = replicated, no comm).
- GDN autoregressive decode is **already fused** (`fused_gdn_ar` default true -> one `GGML_OP_GATED_DELTA_NET` per layer + `ssm_conv`; the long elementwise chains are prefill-only fallbacks). Per GDN layer at decode: 8 mul_mat + fused GDN + ssm_conv + ~10-15 small elementwise/norm ops.
- CUDA-graph cache is keyed on `nodes[0]` + uid; **alternating draft(b1)/verify(b5) shapes give fresh uids per rebuild** - only one uid tracked per subgraph slot, so a whole-token capture needs a per-shape graph cache. NCCL is issued by the host between per-subgraph graph launches (`ggml-backend-meta.cpp:2186-2215`); collectives are not inside any graph today.
- **Repo topology correction** (memory had it inverted): worktree base beac5309f = **2026-06-26**, winner base 5a460dea9 = **2026-07-03** (newer), origin/master e3546c794 = 2026-07-11 (mirror ~3 days stale; `git fetch` before Phase P5). Actionable upstream delta = 5a460dea9..master, 114 commits: one real Pascal decode micro-opt (**683f0c72e** branchless/compile-time-indexed MMVQ), a TP+spec+FA **stability cluster** (4b2a0cdee, 3e5036fbf, **2da668617** stale tensor-split for draft models, defa95c30, cb295bf59, a4107133a, 3cec3bcd1), **b5315e16e** SSE keepalive (stops client drops during slow P100 prefill). No new GDN/ssm kernels, no MTP work, graphs untouched. Conflict risk: **74976e1ae** (1296-line ggml-cuda.cu cuBLAS refactor) overlaps the fork's patches.
- **The production server still runs the non-MTP winner (~33 tok/s)** with LCP prompt-cache on (runtime-server log, 2026-07-13, no RESULTS.md/launch command captured). The 53 tok/s build has never served the real workflow.

**Rig state:** all 4 GPUs on one PCIe switch (all-PIX), single NUMA node; VRAM peak 11.66 GB (GPU0) at depth-7900 (~4.6 GB free); RAM 48 GB (mmap now clean - `--mmap 0` no longer required per baseline-postram-20260714); **disk 38 GB free - below the 40 GB `.xsession-errors` runaway seen before => watchdog mandatory on every 4-GPU run** (pattern in `results/fa-ncols2-experiment/run-fa-p0.sh`).

## 2. House rules (unchanged, binding)

One variable at a time * ABA interleaved fresh-process runs, first (cold) sample discarded, >=3 pairs * byte-identity check for anything claimed lossless; KLD harness (p100-fp16-fix methodology, fp32 ref) for any numeric change * tool-call API + long-context checks before adopting into production * never touch protected builds/models (AGENTS.md list, incl. `build-p100-nccl`, `build-p100-nccl-mtp-pr24566`, `build-p100-pascal-graphs-dedup`, `build-p100-nccl-mtp-compose`) * all experiments env-gated opt-in, default-off, in NEW build dirs * save commands/versions/hashes/full logs under `results/<phase>-<date>/` * no commits/pushes unless asked * no sudo without explicit approval * watchdog on all 4-GPU runs.

Baseline env for all benches: `CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_IB_DISABLE=1 NCCL_CUMEM_ENABLE=0 NCCL_ALGO=Ring GGML_CUDA_P2P=1`, `taskset --cpu-list 0-11`, gates `GGML_CUDA_GRAPHS_PRE_AMPERE=1 GGML_CUDA_GRAPHS_SPLIT_BUFFER=1 GGML_CUDA_Q8_1_DEDUP=1`, `NCCL_MAX_NCHANNELS` unset. MTP: `--spec-type draft-mtp --spec-draft-n-max 4`. llama-bench cannot drive spec decode - MTP benches go through llama-server + the scripted greedy client (mtp-20260714 harness).

---

## 3. Phases

### P0 - Ship the 53 tok/s build to the actual workflow  *(hours, no new code - do first)*
The single biggest realizable "across all 4 GPUs" win is deployment, not kernels: production serves 33 tok/s while a validated 53 tok/s lossless build sits idle.

1. Launch `llama.cpp-q36-mtp-compose/build-p100-nccl-mtp-compose/bin/llama-server` with: baseline env + 3 gates + `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1` + `-m ...Q8_0.gguf -ngl -1 -sm tensor -ts 1,1,1,1 -fa on --ctx-size 8192 --parallel 1 --spec-type draft-mtp --spec-draft-n-max 4` and **prompt cache ON** (no `--no-cache-prompt` - first-ever test of MTP x LCP cache reuse).
2. Validate, in order: (a) 3x scripted greedy decode ~53 tok/s; (b) byte-identity spot-check n4 vs `--spec-type none`; (c) OpenAI tool-call formatting (AGENTS.md item 6); (d) **cache-hit turn followed by correct drafting** - watch for pr24549-class symptoms (garbage/crash from shared-KV draft ctx under slot reuse); (e) >=20-request stability soak; (f) one COMPUTE_32F on/off decode pair (expect neutral, confirming accuracy mode is free under MTP).
3. Sweep `--cache-reuse` (0 vs 256) on a simulated multi-turn agent transcript - incremental-prefill latency is workflow-relevant even though tok/s isn't.
4. Artifacts: `results/mtp-production-20260714/RESULTS.md` + captured launch command (fills the runtime-server documentation gap). Update C4130 runbook notes. Rollback = winner build (33 tok/s), one command.

**Gate:** all six checks pass -> this becomes the deployed default. Any failure is itself a high-value finding (files under the same results dir).

### P1 - Attribute the MTP step  *(half day - decides everything after it)*
No trace of the production mode exists; the ~26-40 ms/step host residual is a hypothesis to confirm or kill.

1. nsys trace of the compose config under the scripted greedy bench (3 runs; depth 0 and depth-7900), phase4r/7a-style kernel-class analysis but **per step**: draft x4 vs verify vs gaps; collective count/size/arrival skew per step (expect 128 + ~8); MMVQ batch-5 kernel ms vs batch-1; `gated_delta_net`/`ssm_conv`/norm/get_rows class table under MTP; graph reuse/rebuild counts across the b1<->b5 shape ping-pong; logits D2H and host sampling time.
2. If nsys can't resolve the host side, add coarse env-gated timers in `common/speculative.cpp` (`common_speculative_impl_draft_mtp`) - measurement build only, never adopted.
3. Output: `results/mtp-attribution-<date>/RESULTS.md` with a ms/step budget table and an explicit re-ranking of P2-P6 (each gets keep/kill/resize).

### P2 - Tune MMVQ for batch 2-8 on sm_60  *(1 day, high confidence)*  <- **P1-CONFIRMED PRIMARY LEVER**
The verify forward is the dominant GPU work of every production step and runs generic MMVQ launch params; the batch-1 analogue of this exact tuning was +11.2%. **P1 hard data: the `mul_mat_vec_q<Q8_0,ncols_dst=5>` verify kernel is 58.8% of all decode GPU time and runs generic params (SM60 table applies only at ncols_dst==1).** This is the single biggest, highest-confidence lever.

1. Extend the fork's `MMVQ_PARAMETERS_PASCAL_SM60` + env knobs to `ncols_dst in 2..8` (Q8_0 first); add batch-5 variants of the 8 real TP decode shapes to test-backend-ops perf cases (batch-1 cases already exist in the winner diff).
2. Isolated sweep (nwarps x rows_per_block per ncols bucket) -> pick per-batch table -> end-to-end MTP A/B (3 interleaved fresh pairs), plus batch-1 and plain-decode neutrality pairs.
3. Build dir `build-p100-mmvq-batch/` in the decodeopt tree; winner files backed up first (established pattern).

**Gate:** >=1.5% MTP tok/s, byte-identical output, batch-1/plain neutral. **Est: +2-6%** (P1 tells the exact batch-5 MMVQ share). Artifacts `results/mmvq-batch-<date>/`.

### P3 - Cut spec-loop host overhead  *(scope from P1; potentially the biggest lever)*
If P1 confirms tens of ms/step where all 4 GPUs idle, attack it cheapest-first, each sub-item its own gated experiment:
1. **Graph rebuild churn:** count rebuilds/step; if the b1<->b5 alternation forces per-step recapture/update, add a per-shape (per-n_tokens) graph/uid cache slot so both shapes stay resident (fork-local, env-gated; the pr24566 fullstack port is prior art and proved correctness-equivalent at 53.1 tok/s).
2. **Fixed per-`llama_decode` host costs** x5 calls/step (graph build, sched, KV bookkeeping): measure, then trim what P1 fingers (e.g., skip redundant re-plans between identical-shape draft calls).
3. **Sampling/verify path:** greedy verify needs only argmax-equality per position; if host sampling shows up, move argmax to GPU (`GGML_OP_ARGMAX` exists) or batch the readback.
4. **Overlap scheduler - SGLang's "zero-overhead" pattern, the invasive endgame of this phase** *(added 2026-07-14 per user)*: restructure the spec loop so host work for step N+1 (draft-batch assembly, acceptance processing, KV bookkeeping, logits handling) runs concurrently with the GPU's in-flight forward of step N, so the 4 GPUs never wait on the host between steps. Justified only if P1 shows >=5 ms/step of host residual surviving items 1-3. Must preserve byte-identical outputs - only timing overlaps, never ordering/semantics.
5. Drafting is inherently sequential (single MTP head) - do not try to parallelize the draft chain itself.

**Gate per sub-item:** >=1% MTP tok/s, byte-identical, revert on miss. **Ceiling: unknown until P1 - plausibly +10-25% if the residual is real and half-recoverable.** Artifacts `results/spec-host-<date>/`.

### P4 - Phase-9 Stage-2: whole-token CUDA-graph capture  *(2-4 days, pre-approved direction, re-gated)*  <- **DEPRIORITIZED per P1**
Removes the host from the per-collective loop (the ~1.3 ms/step arrival skew that survived clock-locking, plus per-launch issue overhead). **P1 KILLED the MTP rationale:** per-collective host-issue gaps are only ~1.7 ms/step (compute already lives inside per-subgraph graphs), so whole-token capture buys <2% under MTP - its ~4% ceiling does not materialize. Keep only as a non-MTP curiosity; production is MTP. Do NOT start unless P2+P3 leave a measured gap that only this can close.
1. **Re-verify the pre-check first** - the earlier GREEN (9231 ptrs byte-stable) has no surviving probe code; re-add `GGML_CUDA_META_PTRCHECK`, confirm stability for both the b1 draft-ctx and b5 verify shapes.
2. Implement per the phase9 design: outer per-device `cudaStreamBeginCapture` around the meta loop (`ggml-backend-meta.cpp:2186-2215`); each device's `ncclAllReduce` issued on its captured stream (restructure the 4-comm `ncclGroupStart/End` span); smallpath `cudaMemsetAsync` onto the captured stream; inner `use_cuda_graph` forced false under outer capture; **per-shape whole-token graph cache** (b1 + b5 slots, host-loop fallback on unknown uid); thread-per-device at capture only, single-thread replay (Stage-1 proved a persistent MT driver adds jitter, -0.70%).
3. Never route through the custom `allreduce.cu` host-staged path (not capturable).

**Gates:** ABA >=1% on BOTH plain tg256 and the MTP bench; byte-identical; pp-neutral; clean fallback. **Kill criteria:** ptr probe unstable -> stop; capture hangs (Stage-1's seqgraph/seqgroup both hung) -> stop and record. Honest ceiling: ~4% plain decode; MTP value = P1-measured. Build `build-p100-p9s2/`, artifacts `results/decode-opt-20260712/phase9/stage2/`.

### P5 - Rebase-forward harvest  *(1-2 days)*
1. `git fetch origin` (mirror stops at 2026-07-11). New worktree at true master tip; port the 4 winner files + pr24549 reuse-disable (+ anything adopted from P2/P3); resolve the **74976e1ae** cuBLAS-refactor conflicts.
2. Harvest: **683f0c72e** (upstream branchless MMVQ - overlaps the fork's table; merge, then re-run the P2 sweep on the new base, cheap with the harness), the stability cluster (incl. **2da668617**, directly on the MTP-under-TP surface), **b5315e16e** SSE keepalive + server fixes.
3. Full re-gate: decode + MTP (3 interleaved pairs, no regression tolerated - the old "master regresses -3-4%" pin rationale must be re-tested, not assumed), pp within noise, quick KLD, tool-call check.

**Gate:** >=0% (no regression) -> becomes the new production base. Miss -> fall back to cherry-picking only the stability cluster onto 5a460dea9 (half day). Artifacts `results/rebase-<date>/`.

### P6 - GDN kernel + small-kernel spike  *(1 day, gated on P1, kill fast)*
~8.5 ms/token of non-matvec, non-NCCL kernel mass at batch-1 (and its batch-5 analogue from P1). GDN AR is already fused, so the plays are: (a) tune `gated_delta_net.cu`/`ssm-conv.cu` launch geometry for sm_60 (occupancy audit - same playbook that found the FA tile kernel at 47% of peak); (b) rms_norm+mul neighbor fusion if the winner base lacks the upstream fused-ops matcher coverage; (c) fold the 97x/token get_rows conv-state gather into `ssm_conv`; (d) *(added 2026-07-14, TRT-LLM pattern)* **epilogue fusion of the residual chain** - fold residual-add + rms_norm + q8_1 activation-quantize into one kernel, killing full read+write round-trips of the 5120-wide hidden state per site (pool: 209 norm + ~257 quantize launches/token).
**Caution:** Phase 5B fusion was rejected at -0.05% with the GPU 93% busy - wins must remove memory round-trips or kernel-time floors, not launches. **Spike gate:** hand-optimize only the single hottest item from P1's histogram; <1% decode -> kill the whole phase and record.

### P7 - OPT-IN (explicit user approval required): quality-gated Q6_K  *(1 day)*
The only remaining matvec lever (matvec = 63% of decode, kernel-closed). Q8_0->Q6_K ~ -24% weight bytes => est **+10-15% decode**, composing to ~58-61 tok/s with MTP.
Hard pre-registered accuracy gate (accuracy-first standing directive): KLD harness vs fp32 ref with COMPUTE_32F on - adopt only if top-token agreement >=99.5% and mean KLD <= a user-set bar (suggest <=0.003; Q4_0 failed at 90.3%/0.0508). Re-validate the 3 env gates on Q6_K (gates + Q4_0 crashed in q4-track). **Not to be run until the user approves the bar.**

### P8 - OPT-IN long-context: FA ncols2=6 productization  *(2-3 days, only if >=64K contexts become routine)*
(a) **Split-KV occupancy tuning for the TP-starved nh=1 shape** *(added 2026-07-14 per user)*. Why: under TP each GPU holds 1 KV head, so ncols2=6 launches a single 192-thread head-block and leans entirely on the split-KV heuristic (`fattn-common.cuh` occupancy + 95% wave-efficiency search, never designed for this shape) to fill 56 SMs. That dilution is why the per-GPU kernel gets 1.39x @128K where the nh=4 geometry gets 1.83x, and it is the mechanism behind the sub-64K end-to-end regression (fewer blocks on the 16x/token AllReduce critical path). Do: sweep forced KV-partition count / minimums / combine-kernel geometry for the D256, GQA6, nh=1, ncols2=6 case - scoring **main kernel + partial-combine together**, isolated via test-backend-ops (kv 8K->128K, existing harness in `results/fa-ncols2-experiment/`), then FA-P0 spot pairs @32K and @128K with the watchdog. Prize (bounded, pre-registered): up to ~+2-3% more end-to-end @128K (total ~+7-8%) and possibly a crossover below 64K; kill if isolated gain <10% or end-to-end @32K stays negative.
(b) Context-adaptive dispatch `K->ne[1] >= threshold`, **threshold set from (a)'s retuned crossover** (stock value >=65536 per FA-P0: +5% @128K, zero regression below).
(c) Extend the kernel past its `Q->ne[1]==1` guard to `ncols1 <= 5 x ncols2=6` so it fires during MTP verify (today it is inert in production).
Branch `fa-ncols2-exp` @ acbe0c9d8 is based on beac5309f (older than the winner base) - rebase onto the P5 base when productizing. Re-measure with the FA-P0 watchdog methodology. Not recoverable regardless of tuning: the Amdahl ceiling (attention ~15% of the step @128K) - do not chase past ~+8%.

### P9 - Tree-MTP: multi-candidate speculative verification  *(1/2-day modeling pre-step, then 2-4 days, gated)*  *(added 2026-07-14 per user)*
Besides P7, the only remaining *bandwidth multiplier*: raise tokens-per-weight-stream by verifying a small candidate tree instead of the linear n4 chain (Medusa/EAGLE pattern). Today 3.0 of 4 drafts accepted + 1 bonus = 4.0 tok/step (ceiling 5.0); second-choice branches at the depths where first-rejections concentrate could recover ~+0.3-0.5 tok/step (**est +8-12%**), and verify batches of 7-8 stay on the MMVQ <=8 fast path (covered by P2's table).
1. **Modeling pre-step (mandatory gate, no code):** extract per-depth/per-rank acceptance probabilities from existing + fresh acceptance logs; compute expected tokens/step for candidate tree shapes (e.g. width-2 at depths 1-2) against verify-batch costs measured in P1/P2. Build only if the model predicts >=+8% wall-clock.
2. Build: express the tree via `llama_batch` multi-sequence masks (shared-prefix branching is supported); extend `common_speculative_impl_draft_mtp` to draft alternates (MTP-head top-k) and accept the longest matching root-path; keep greedy argmax-equality acceptance => lossless by construction (byte-identity check still mandatory). Flag-gated, default off.
3. Risks: the reproducible n3 dip proves verify-shape sensitivity - sweep tree sizes; the acceptance cliff past depth 4 means width helps early depths only, so keep trees shallow.
**Gate:** >=4% measured MTP tok/s (half the modeled floor), byte-identical, else revert. Composes with P2 (batch-MMVQ) and P3 (host overlap).

### P10 - PARKED, LOW PRIORITY (user, 2026-07-14: single-agent workflow today): continuous batching for concurrent sessions
When/if parallel agents become routine: `--parallel 2..4` on the existing TP4 server shares every weight stream across sessions - ~1.8x aggregate at batch 2 for ~10-15% per-session latency cost; KV ~128 MB/GPU per extra 8K slot; zero kernel work. Prerequisite: validate the untested MTP x multi-slot interaction. Revisit on workflow change only. (Distinct from the also-parked TP2 dual-replica idea below.)

---

## 4. Closed - do not revisit (and why)

- **Matvec arithmetic on P100** - 10+ refutations (4X/4Y/4Z, DMMV, FP64-SWAR, half2/HFMA2 x6, reg caps, geometries); 82% of layout read ceiling. See `dp4a-alternatives-survey-20260713.md`.
- **Comm micro-optimization** - custom P2P/fp16 allreduce (Phase 7), NCCL proto Simple (decode -13.5%), channels/threads/buffsize, chunked convert overlap (8B), comm/compute overlap (8C - SM-resident NCCL contention), clock locking (7S), MT host driver (9 Stage-1, -0.70%). bf16 large-path allreduce is at a hardware local optimum.
- **GDN layer weight replication to skip collectives** *(evaluated and rejected this round by arithmetic)* - trades ~60 us of collectives for 4x weight streaming on a bandwidth-bound decode (~+0.9 ms/layer vs -0.06 ms); full duplication also OOM'd at depth-7900 (phase6).
- **Context-parallel / sequence-sharded attention** *(evaluated this round)* - the K/V-traffic win is already captured by ncols2=6 on the existing head-split layout; it would add 16-32 latency-bound PCIe collectives/token against an attention slice that is ~15% of the step.
- **f16-residual stream** - accuracy sim passed but ~9 invasive edit clusters for ~8% prefill-only; wrong direction under accuracy-first. **splitk-hgemm** - deferred; reopen only if the 451 tok/s accuracy-mode prefill becomes a real pain point. **Q4_0** - quality-failed. **TP2 dual-replica** - only if multi-client throughput becomes a goal. **IRQ-affinity tail patch** - only if P4 lands and the p99 tail persists.

## 5. Sequencing and expected outcome

**P0 today** (deployed decode 33 -> ~53, +60% at zero code risk) -> **P1** (attribution; re-ranks the rest) -> **P2** (quick, high-confidence kernel win) -> **P3/P4** in the order P1 dictates (P3's endgame = the SGLang-style overlap scheduler) -> **P5** (harvest + robustness) -> **P6** spike. **P9's modeling pre-step** (1/2 day, log analysis only) can run any time after P1; its build only on the model's green light. **P7/P8 only on explicit approval. P10 parked** (single-agent workflow today).

Realistic compound on the benched config (excluding opt-ins): 53 -> **~56-62 tok/s lossless** (P2 +2-6%, P3 +0-25% wide by design, P4 +1-4%, P5 >=0 + robustness); **P9, if its model holds, stacks +8-12%** (-> ~60-68). With P7 also approved and passing its accuracy bar: **~70-75 tok/s best case**. Workflow-level, the guaranteed piece is P0's 33->53 on real sessions plus SSE/stability hardening from P5.

Engine-alternatives scan (2026-07-14): vLLM/TensorRT-LLM/ExLlama do not support Pascal, so importing ideas is the only route - and after this revision every importable idea is either already adopted (Megatron TP, graphs, speculation), already refuted here (custom P2P allreduce, KV-cache quantization - Pascal's vec-FA makes it 6x redundant), or now in-plan (P3.4 overlap scheduler, P4 whole-token capture, P6d epilogue fusion, P9 tree-MTP, P10 batching). P100-unique fp16x2 and full-rate FP64 were both tested and lost to the issue/bandwidth bound - no unexploited hardware feature remains.

Every phase carries its own kill criterion so a miss costs a day, not a week - the pattern that separated the 9 adopted wins from the ~25 clean rejections so far.
