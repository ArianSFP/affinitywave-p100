# AffinityWave prefill backend

## Goal and measured baseline

Raise Qwen3.6-35B-A3B Q8_0 pp8192 from the measured 1697 tok/s toward
4000 tok/s without changing weight precision or decode semantics.
AffinityWave replaces replicated tensor-parallel execution with a four-chunk
diagonal wavefront, replicated dense layers, and activation-routed expert
service.

## Architecture

- Split an 8192-token prompt into four contiguous 2048-token chunks. Chunk `c`
  remains on GPU `c`.
- Schedule cell `(chunk, layer)` only after `(chunk, layer-1)` and
  `(chunk-1, layer)` complete. The 40x4 grid has 43 diagonals versus a 40-cell
  ideal, a 7.5% structural fill cost.
- Replicate the approximately 2.313 GiB non-expert trunk on every GPU. Keep one
  primary copy of each expert, 64 experts per GPU per layer. Keep
  embedding/output tensors sharded.
- Initially replicate each layer's 16 most frequent code-domain experts on
  every GPU, adding approximately 1.494 GiB/GPU. Test hot-32 only when the
  complete `-c 32768 -ub 4096` configuration retains at least 768 MiB free on
  every GPU.
- Each home GPU computes its cell's attention/GDN, router, shared expert, and
  locally available routed experts.
- Pack each token activation once per required cold-expert owner. The remote
  owner evaluates every selected expert it owns and returns one route-weighted,
  summed hidden vector per token.
- Default transport is F32 requests and BF16 owner partials. Round local
  partials identically and accumulate owners in fixed rank order. BF16 requests
  are a separate optional accuracy-gated experiment.
- Pool expert work from all active cells on a diagonal into one cross-layer
  grouped service per projection and GPU. Tile descriptors carry layer,
  expert, input, output, and row offsets. Never overlap two throughput-bound
  compute kernels on the same P100; only P2P copies overlap compute.
- Pass GDN state and cumulative attention KV state from chunk home `c` to
  `c+1` using peer DMA and CUDA events. Normalize final state/KV into the
  existing production layout before decode.
- Bypass the stock cyclic scheduler: use one persistent submission thread, one
  compute stream, and separate inbound/outbound copy streams per GPU.

## Interfaces and implementation sequence

Add no public llama.cpp API changes. Use opt-in environment controls:

- `GGML_CUDA_AFFINITY_WAVE=1`
- `GGML_CUDA_AW_MAP=<placement-manifest>`
- `GGML_CUDA_AW_WIRE=f32|bf16`, default `f32`
- `GGML_CUDA_AW_CHECK=1`

The placement manifest records the GGUF hash, 256 primary-owner assignments
and replicated expert IDs for every layer. Generate it deterministically from
token-level top-8 code-routing traces.

### Phase 0: feasibility gate

Capture token-level routing, build a discrete-event simulator using measured
dense, grouped-GEMM, P2P, and state-copy costs, and optimize placement by
deterministic expert swaps against held-out code traces. Do not build the
backend unless pessimistic simulated pp8192 time is at most 2.05 seconds.

### Phase 1: isolated service

Implement replicated-dense/hot-expert loading and isolated cross-layer expert
service in a separate `build-p100-affinitywave` build. Require at least 5.5
effective TFLOP/s including packing and at least 10 GB/s sustained peer
transport.

### Phase 2: cold-prefill wave

Implement cold-prefill wave execution, state corridors, sparse
request/response exchange, and existing embedding/head adapters. Continue only
if hot-16 reaches at least 3200 tok/s and measured stage timings remain within
10% of simulation.

### Phase 3: append and normalization

Add append-prefill support and production decode-state normalization. Evaluate
hot-32 and BF16 requests independently, only after their memory and accuracy
gates pass.

Unsupported architectures, multi-sequence batches, short prompts, invalid
manifests, or disabled mode use the existing EP4 path. Explicitly requested
AffinityWave with incompatible hardware or insufficient memory fails loudly.

## Validation and acceptance

- Accuracy: byte identity where possible; otherwise perplexity within +/-0.003
  and KLD-pair behavior no worse than control self-variance. Run full-position
  perplexity because greedy smoke previously missed collective corruption.
- Correctness scenarios: cold and appended prompts, all four chunk boundaries,
  attention and GDN state handoffs, non-multiple lengths, code prompts,
  repeated requests, tool calls, and transition to ordinary decode.
- Performance: one warm-up plus at least three interleaved runs at pp512,
  pp2048, pp8192 and pp16384; record average, range, standard deviation, VRAM,
  stage timings, remote-owner counts, and copy/compute overlap.
- Milestones: R=0 prototype >=2700 tok/s; hot-16 >=3200; breakthrough target
  >=4000 pp8192. Results below 4000 remain experimental and must not be
  presented as meeting the breakthrough target.
- Decode must remain on the current production backend and stay within 2% of
  approximately 74.6 tok/s after state normalization.
- Run only one watchdog-protected four-GPU job at a time. Do not modify
  protected builds, commit, push, or alter production deployment without
  explicit approval.

## Guardrails

- Ignore MTP entirely.
- Never use or evaluate sub-Q8 weights.
- Do not retry stock cyclic PascalWave, same-SM stream overlap/TBO, all-weight
  AsyncEP, HFMA2/FP16 expert GEMM, TCCL, NCCL Simple, context parallelism, or
  lossy expert skipping.
- The novelty claim is the synthesis: fixed chunk affinity, diagonal
  hybrid-state execution, replicated dense skeleton, hot-expert replication,
  owner-aggregated sparse exchange, and cross-layer grouped expert service on
  PCIe Pascal.

