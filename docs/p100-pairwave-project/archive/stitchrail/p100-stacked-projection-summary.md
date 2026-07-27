# P100 exact stacked-projection probe

Date: 2026-07-26

Probe:

- CUDA 12.8, sm_60, cuBLAS GemmEx, `CUBLAS_OP_T/CUBLAS_OP_N`.
- Half A and B, float C, `CUBLAS_COMPUTE_32F`, TF32 math mode.
- Two contiguous `K x M` A regions and one shared `K x N` B region.
- Baseline: two separate calls with the retained algorithm.
- Candidate: one `2M x N` call.
- Sweep: DEFAULT, algorithms 0-23, and algorithms 99-115.
- Five warmups and 30 timed repeats in each of three independent processes.
- Both complete `M x N` output regions were compared by float bits.

## M512 + M512 -> M1024

Signature: base M512, N2032, K2048; separate baseline algorithm 3.

- All three processes tested 42 algorithms and supported 28.
- Twenty-three candidates were exact in both output regions in all processes.
- The best three-process paired median was algorithm 112:
  - median speedup: 0.997310495x
  - median candidate time: 1.282949320 ms
  - median paired saving: -0.003447469 ms
  - mismatch count: 0 in the first region and 0 in the second region
- Per-process speedups for algorithm 112 were 0.999907726x,
  0.997310495x, and 0.899646531x.

Decision: close M512 stacking. It misses the 2% continuation threshold and
has no positive paired median.

## M32 + M32 -> M64

Signature: base M32, N2032, K2048; separate baseline algorithm 7.

- All three processes tested 42 algorithms and supported 28.
- Algorithms 7, 8, 9, 10, and 11 were exact in both output regions in all
  processes.
- Algorithm 7 was the exact winner:
  - median speedup: 1.671574345x
  - median candidate time: 0.123452799 ms
  - median baseline pair time: 0.206360531 ms
  - median paired saving: 0.082907732 ms
  - time reduction: 40.176158%
  - mismatch count: 0 in the first region and 0 in the second region
- Per-process speedups for algorithm 7 were 1.671574345x, 1.637776984x,
  and 1.694486637x.
- Algorithm 8 was the exact control winner at 1.582053536x median.

Decision: retain M64 stacking with algorithm 7 for graph integration and
full valid-model qualification. This probe does not authorize a production
change by itself.

All six error logs are empty. The watchdog did not fire,
`.xsession-errors` remained 22,762 bytes, and the GPU lock was released
after every run.
