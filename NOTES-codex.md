# Coding-serving implementation

Start: 45fccdd5665d6c6c2f0fb669c1f630af84d0630e, qualified source 36bad6bb3.
Objective: serving-capable build, approximately 2912 tok/s long prefill,
improved short fresh and cached coding turns. No deployment replacement.
Accuracy: byte identity or paired PPL +/-0.003. All-Q8 restriction relaxed
by user; initial implementation keeps Q8 to isolate execution changes.

## Plan

1. Reproduce immutable long/short baseline under the scoped GPU runner.
2. Selectively port the existing arbitrary-length lifecycle implementation
   from the read-only affinitywave-openai-sass worktree behind AW_SERVE.
   Keep the legacy qualified execution unchanged when AW_SERVE is off.
3. Build isolated server/bench/perplexity binaries; validate startup, odd
   prompt lengths, decode, continued recurrent state, and prefix reuse.
4. Measure short-shape crossover options and graph setup. Accept only
   accuracy-gated, repeatable latency gains without material long regression.
5. Provide reproducible localhost launch and measured acceptance report.

Prior lifecycle work is tested precedent, not newly authored work: its
notes report 128/512/1024 and remainder PPL gates plus HTTP smoke, but only
about 2454 tok/s on the older baseline. Its precision-dependent fallback
warning is binding evidence: ordinary T64-weight prefill is not native
normal arithmetic. Do not implement a blind small-token fallback.

## Initial execution

- Sandboxed tools do not expose /dev/nvidia*. Approved host read-only checks
  show four idle P100s; GPU jobs need approved unsandboxed execution.
- Current host memory is approximately 30 GiB, not the older memory note's
  48 GiB. Use --no-mmap and serialize GPU work; do not compile during timing.
- Original qualified binary, same documented environment, current host:
  pp8128 r4 mean 2892.678 tok/s (samples 2887.36-2897.21), pp512 r4 mean
  1275.875 tok/s. Full llama-bench timing, not internal wave timing.
- Lifecycle port is behind GGML_CUDA_AW_SERVE=1. It includes per-ubatch
  token scope, aligned wave/normal-tail splitting, bounded shape/pointer/
  parameter pre-capture signatures, continued-state warmup suppression,
  and canonical state export. It deliberately does not adopt the older
  patch's partial-format auto policy without renewed accuracy measurements.
- The copied signature logic was tightened to hash only current graph
  nodes, not the backend vector's historical capacity, and include op
  parameters and source presence. Signature markers are capped at 64.
- Isolated CUDA 12.8 build requires explicit g++-14 for both C++ and CUDA
  host compiler; system default gcc is now 15 and was rejected by nvcc.
- Nine CPU harness tests pass (environment isolation, scoped shutdown,
  streaming token/timing extraction, and error handling). GPU tests pending.

## Initial server smoke (not qualified)

- Isolated server builds and starts. Fresh 128 and 513 token HTTP requests
  complete, but transition after the 1020-token portion of a 1025-token
  request segfaults. Reproduced under GDB in meta_graph_compute. Debugging
  underway; this prototype is not ready to serve user workloads.
- First-use prompt rates are approximately 75 and 240 tok/s, respectively;
  these include state setup and precapture, unlike steady-state bench.
- Server checkpointing splits off four-token tails. Canonical state export
  currently transfers whole KV roots: approximately 2296.9 MiB of corridor
  traffic per wave at context 16384, including tiny four-token waves.
  Investigate bounded updated-range export after lifecycle correctness.
- GDB v2 localized the crash to cgraph_ij->n_nodes in subgraph rebuilding.
  Resetting the context rebuilt only current n_subgraphs while retaining
  max_subgraphs, leaving dangling graph pointers for later transitions.
  Fix recreates all max_subgraphs with max_nnodes capacity on reset.
- serve-lifecycle-smoke-v2 passes 128, 513, 1025, 128 fresh requests,
  2048-token seed, 64/129-token cached suffixes, and OpenAI chat API.
  Cached suffix TTFT is still approximately 858/933 ms. Functional success
  is not an accuracy or speed qualification. Chat 64-token decode measured
  50.17 tok/s after cold prompt setup; initial 4-token decode is slower.

## Prefix-copy / no-extra-warmup experiment

- Opt-in GGML_CUDA_AW_SERVE_KV_PREFIX=1 crops token-major single-stream
  KV root copy views to the padded prefix actually visible to attention.
  Unsupported layouts retain full copies. No stored values or arithmetic
  are changed. GGML_CUDA_AW_PRECAPTURE=0 avoids two redundant fresh-state
  passes; continued-state passes were already disabled.
- Both serve-c512-logits-v2 (control) and serve-prefix-nowarm-c512-v3
  produce accepted hash 47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5.
- First smoke comparison, not a repeated performance qualification:
  fresh513 prompt rate 234.61 -> 501.46 tok/s; fresh1025 348.56 -> 816.87.
  Cached64 TTFT 857.81 -> 633.17 ms; cached129 932.58 -> 662.99 ms.
  Four-token tail corridor traffic falls from 2296.9 to 406.9 MiB at the
  first short prompt. Generated tokens match the preceding smoke run.
- Full-vocabulary teacher-forced boundary validation is implemented in
  bench/coding-serve/validate.cpp. It covers 24 boundaries across changing
  fresh sizes, four-token tails, cached additions and 8128-token prompts.
  Its byte comparison is stronger than matching sampled HTTP text.
  Eleven CPU harness/comparison tests pass.
- Full boundary validation: prefix-only v3 is byte-identical to control
  across all 24 boundaries / 23,838,720 logit bytes. Removing all warmups
  also matched the first 18 boundaries, but differed at the six long-context
  boundaries. Reject unconditional PRECAPTURE=0 as an exact serving policy.
  Added opt-in SERVE_PRECAPTURE_MIN_TOKENS to retain long fresh warmups while
  skipping short ones; threshold 2048 is next to validate.
- Native-reference short gates with FP32 partials: c128 reproduces prior
  PPL 5.0340 / ln ratio -0.00015 (approximately -0.0008 PPL delta). Adaptive
  c1024 reproduces delta +0.001769 and KLD 0.000590, passing +/-0.003.
- Added PARTIAL=serve-auto: BF16 at lane 128 and lane >=512, FP32 otherwise.
  FP32 short service uses a separate scratch-state bank so a preceding long
  panel run cannot inflate its legacy buffers to historical long capacity.
  Legacy qualified behavior is unchanged for PARTIAL=bf16.
- Long native-reference quality test evaluates the final 512 positions of
  an 8124-token WikiText prefix, requesting only those logits to avoid OOM.
  Native ub2031 PPL 6.185228676; warmup-free serving PPL 6.183474105,
  paired delta -0.001754571 PASS, mean KL(base||candidate) 0.000601446.
  Thus unconditional no-warmup is not byte-exact but passes this long suffix
  PPL gate. This is not a claim of full-corpus long-context qualification.
- Opt-in SERVE_TINY_MMVQ routes only four-token-and-smaller batches through
  the existing T64 MMVQ path. The prior rejected general short fallback is
  not being retried: this is restricted to the existing <=4 MMVQ dispatch.
  Native paired validation of this tail optimization is pending.
- Tiny MMVQ gate passed: native ub4 PPL 5.331306527, T64 serving ub4
  5.331732811, delta +0.000426285 over all 512 evaluated positions.
- Combined v5 coding matrix passes. Real fresh HTTP pp8128 is only
  2190.96 tok/s / 3718.38 ms TTFT, despite internal wave about 2770 tok/s.
  This gap is not hidden or claimed as retained 2.9k HTTP performance.
  Cached suffix64 after an 8k prefix is 654.49 ms TTFT; suffix128 759.01 ms.
- Nsight Systems v5 trace (sample=none, no counters) captured CUDA/NVTX/OSRT
  successfully. The runner sets a clean environment on nsys, which then
  injects its own child; it does not strip the injection with env -i.
  At the long-prompt transition, cudaMallocHost costs 157.93 ms and
  cudaFreeHost 14.93 ms within an approximately 410 ms pre-wave interval.
  Approximately 300 ms follows the main wave for checkpoint/tiny-tail work.
  Profile timings are diagnostic only, not performance acceptance samples.
- Testing SERVE_RESERVE_FULL: reserve token-lane activations at full ubatch
  size with one output per sequence instead of the old unsplit reserve.
  This avoids the old full-logit OOM shape; runtime larger output requests
  remain able to grow. Off by default pending validation. Token scopes now
  restore prior TLS values safely across nested reservation calls.
- Added optional SERVE_TIMING phase logs to separate graph build, allocation,
  input setup, and graph compute. No profiling of hardware counters attempted.

## Final serving candidate / v6-v7 validation

- The full-reserve server matrix (`serve-full-reserve-matrix-v6`) passed fresh
  and cached coding requests plus the OpenAI-compatible chat probe. It measured
  fresh prompt rates of 73.44, 178.22, 345.21, 579.50, 919.42, 1417.21,
  2024.84 and 2312.37 tok/s for 64 through 8128 input tokens. Cached suffix
  rates were 106.10, 174.17, 294.25, 529.34 and 706.57 tok/s for suffixes
  64 through 1024. The 8128-token request's HTTP TTFT was 3523.23 ms.
- Full-reserve c512 paired logits (`serve-full-reserve-c512-v7`) are byte
  identical to the accepted control hash
  `47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5`.
- The final launcher enables full reserve together with prefix-only KV export,
  adaptive short/long partial precision, the restricted <=4-token MMVQ path,
  and the warmup-free serving policy. It remains isolated under
  `bench/coding-serve/serve.sh`; no existing service is replaced.
- CPU harness (12 tests), shell syntax, diff whitespace, long suffix PPL and
  tiny-tail PPL gates all pass. The protected fixed-shape qualified path is
  unchanged and remains the ~2892.7 tok/s pp8128 reference; the serving HTTP
  path is intentionally reported separately because graph/checkpoint/output
  host work lowers its first fresh 8k request to ~2312 tok/s.
- Final candidate fixed-shape pp8128 benchmark (`qualified-candidate-8128-v9`)
  measured 2885.253 tok/s (samples 2879.83--2890.10), versus the earlier
  2892.678 tok/s reference. The serving-only guards therefore retain the
  approximately 2.9k qualified long-prompt rate within normal run variance.
- A timing-enabled matrix (`serve-full-reserve-timing-v8`) passed all fresh,
  cached and chat probes; its fresh 8128 request measured 2374.05 tok/s in
  that run, with an internal serving wave of 2772.3 tok/s. This confirms that
  the remaining HTTP gap is request/state overhead rather than a loss in the
  fixed-shape kernel path.
- Exact final-environment long quality gate (`serve-full-reserve-long-quality-v10`)
  passed: base PPL 6.185228676, candidate PPL 6.187218821, paired delta
  +0.001990145, mean KLD 0.000604380, same-top-token fraction 0.994140625.

## Short-prompt serving follow-up / v28-v36

- Added `--prewarm` to the isolated server runner. It sends non-cached,
  four-token-output requests for common prompt sizes after health becomes ready,
  retaining exact graph/capture signatures without seeding the user prompt
  cache. The persistent launcher prewarms 64/128/256/512/1024-token shapes.
- Final prewarmed matrix (`final-prewarm-matrix-v35`) passed transport and chat:
  fresh rates were 101.57, 193.03, 349.59, 596.26, 924.90, 1399.12,
  2000.41 and 2348.06 tok/s for 64 through 8128 tokens. Against v6 this is
  the largest gain at 64 tokens (73.44 -> 101.57, +38%), with smaller gains
  at 128-1024; the long HTTP path remains overhead-limited while the fixed
  kernel path is unchanged.
- Attempts to skip all or selected fresh canonical exports were rejected:
  all-export skip truncated output; recurrent-state-only skip did not improve
  rates; KV-only skip changed chat output. These experimental controls were
  removed from the final source.
- An actual-position KV-root crop was rejected: although it reduced one
  corridor from roughly 407 MiB to 391 MiB, the 8124-token PPL moved to
  6.1493. The final source retains only the previously validated view-based
  prefix crop (`GGML_CUDA_AW_SERVE_KV_PREFIX=1`).
- Final accuracy gates: c512 logits remain byte-identical to
  `47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a`; the
  established 8124-token quality run (`final-safe-long-quality-v33`) reports
  PPL 6.186618350 versus 6.185228676 (+0.001389674), within the +/-0.003 gate.
- Final fixed-shape benchmark (`final-qualified-8128-v36`) reports
  2885.23 tok/s (samples 2881.26-2889.37). CPU harness has 12/12 tests pass;
  shell syntax and diff whitespace checks pass. No commits, pushes, or service
  replacement were performed.
