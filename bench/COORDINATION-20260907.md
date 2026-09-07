# Coding-serving latest-port coordination - 2026-09-07

This worktree is a detached forward-port from upstream llama.cpp master
0cae43063 (2026-09-07), with the measured P100 MMVQ/Q8_1 decode changes and
the coding-serving harness. Upstream already contains DSpark and SpecForge
checkpoint support.

GPU policy: one four-P100 job at a time, guarded by
/tmp/affinitywave-4gpu.lock and the runner's .xsession-errors watchdog. No
commits, pushes, or PRs are being made.

ACTIVE: Self-contained DSpark n=2 on `CUDA0` is selected as the default: 77.3
tok/s chat (1.46x matched baseline), byte-identical fresh outputs, with the
derived draft’s vocabulary payloads verified byte-identical to the target.

2026-09-07: Regular-MTP comparison is active. The target GGUF contains four
`blk.40.nextn.*` MTP-head tensors; it will be tested with `draft-mtp` at n=2
under the same serialized four-P100 benchmark conditions.

MTP n=2 is faster than DSpark on 512/1024-token fresh turns and cached
suffixes, while DSpark wins 64/128/256-token fresh turns and is nearly tied on
chat. MTP n=3 is slower than MTP n=2 on long turns and chat; DSpark n=2
remains the default for this serving profile.

2026-09-07: Decode optimization follow-up is testing the existing
`ngram-mod,draft-mtp` stack at n=4, then DSpark batch geometry. One four-GPU job
at a time remains enforced by the runner lock and xsession watchdog.

The ngram+MTP n=4 probe was slower than regular MTP/DSpark on this workload;
next probe is regular MTP n=2 with the target's large `ubatch=8128` geometry.

Follow-up: ngram+MTP chat was 50.3 tok/s and MTP n=2/ubatch8128 was 68.1
tok/s, both below the selected DSpark n=2/ubatch512. DSpark fresh requests
currently report zero graph reuse; testing DSpark n=2/ubatch1024 for a safe
geometry improvement.

DSpark n=2/ubatch1024 improved short fresh turns (about 35 tok/s at 64-256
tokens) but reduced the chat turn to about 72.7 tok/s, so it is not adopted.
The next serialized probe is ngram+MTP with a larger draft window (n=16) to
see whether the repeated-code fixture can amortize verification better.

The n=16 ngram+MTP startup failed before inference because the Qwen recurrent
state metadata arena exhausted while creating the MTP context; no benchmark
data was produced. A narrower n=8 probe is in progress.

The n=8 ngram+MTP probe completed without a crash but was rejected: chat was
47.2 tok/s, short fresh turns were about 29.0-29.6 tok/s at 64-256 tokens and
12.5-13.8 tok/s at 512-1024, below the selected DSpark n=2 profile. Testing
DSpark confidence pruning next.

DSpark n=2 with `p_min=0.5` completed and remained exact, but chat was 75.3
tok/s with no balanced-suite gain; it is rejected. The current safe winner is
therefore unchanged: DSpark n=2, CUDA0 draft, ubatch=512.

2026-09-07: Implemented the selected P100 patch priorities 1-5 in order. The
final build passed 525/525 CUDA0 TOP_K cases, including large vocab and
multi-row cases. The watched four-P100 all-patches DSpark n=2 probe passed with
byte-identical tokens/content against the prior DSpark run; coding chat was
80.6 tok/s. No commit, push, or PR.
