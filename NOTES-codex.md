# Codex notes

- 2026-09-07: Ported the serving harness and measured P100 decode changes onto
  upstream `0cae43063` instead of transplanting the July branch wholesale.
- Upstream already contains the current DSpark/SpecForge implementation; the
  downloaded Qwen3.6-35B-A3B DSpark draft is
  `/home/arian/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-DSPARK.gguf`, SHA-256
  `5b81226a453eb9898f810ad9939d6c6cbe3d8994912d5443d771ed99a48a0ded`.
- The first DSpark launch reached draft loading but failed when the draft
  shared the target LM head while the draft scheduler was restricted to
  `CUDA0`; the target head is a split four-GPU `Meta(...)` tensor. A same-device
  retry exposed that DSpark's vocabulary-wide in-graph `argmax` also cannot
  consume a split vocabulary tensor. The runner now uses a derived,
  self-contained draft GGUF on `CUDA0`, leaving the target transformer and
  vocabulary tensors tensor-parallel and untouched.
- The target-context sharing route also rejected same-GPU buffers. The derived
  `Qwen3.6-35B-A3B-DSPARK-P100.gguf` draft contains byte-for-byte Q8_0 copies
  of the target's two vocabulary tensors, so the draft context owns them.
- Matched `ubatch=512`, `ctx=16384` server probes were byte-identical on all
  fresh greedy completions. DSpark n=2 was the best P100 setting measured:
  32.6/33.6/33.0 tok/s at fresh 64/128/256-token prompts versus
  21.1/21.0/20.9 baseline, and 77.3 tok/s on the chat probe versus 52.8
  baseline. n=3 fell to 33.4 tok/s on chat; n=1 reached 65.7 tok/s. At
  512/1024-token fixture prompts, draft acceptance was zero, so speculative
  decoding remains workload-dependent and can be slower there.
- CUDA3 placement was a near tie on chat (78.8 tok/s) but noisier on short
  turns; the stable default remains `CUDA0`. The runner and `serve.sh` now
  default to DSpark n=2.
- Regular target MTP was also tested at n=2 and n=3 with the same short suite.
  MTP n=2 reached 27.1/23.1/26.5 tok/s at fresh 64/128/256-token prompts,
  16.5/17.5 tok/s at 512/1024, and 76.0 tok/s on chat. MTP n=3 reached
  29.7/29.4/28.2, 15.0/15.6, and 61.9 tok/s respectively. MTP therefore
  beats DSpark on longer fresh and cached turns, while DSpark n=2 wins the
  short turns and is slightly faster on chat; all MTP outputs were byte-
  identical to the non-speculative baseline.
- No commit, push, or PR was made.
- The decode follow-up rejected three additional knobs. DSpark n=2 with
  `ubatch=1024` improved fresh 64-256-token turns to roughly 35 tok/s but
  reduced the chat turn to 72.7 tok/s, so `ubatch=512` remains the balanced
  default. Ngram+MTP n=8 completed at 47.2 tok/s chat and 12.5-13.8 tok/s on
  512-1024-token fresh turns; n=16 could not initialize because the Qwen
  recurrent-state metadata arena exhausted at `n_rs_seq=16`. DSpark n=2 with
  confidence pruning `p_min=0.5` remained exact but reached only 75.3 tok/s
  chat and was not adopted. All three tests used the same watched four-P100
  server probe and Q8_0 target weights.
