# Final P100 exact static audit

Date: 2026-07-26

## Identity

- linked `libggml-cuda.so` SHA-256:
  `44d0db6d838953a57fe5dc18d9045f1d3274f933c94d7e5494816a0d254403b7`;
- tracked binary source diff SHA-256:
  `82d5d9826e31e7f238b9d33fca0483ac08094121d66df38f86e15cfb864c624f`;
- target: CUDA 12.8, cuBLAS 12.8.3, `sm_60`;
- the archive contains toolchain versions, source and cubin hashes, linked
  resources, full SASS, selected cubins, nvdisasm, and fresh verbose ptxas
  objects and logs.

## Cleanup

The development-only `aw_live_check_plan64` and `aw_r44_check_plan`
comparators are absent from source, linked resources, ELF symbols, SASS,
nvdisasm, and ptxas logs. Their runtime environment plumbing is also gone.
The qualified parallel planners and their original serial fallbacks remain.

The removed Q8_0 N=5-8 MMVQ candidate and its read-only-load helpers are
absent. Decode and MTP are outside the final campaign.

## Retained target resources

All 28 retained target kernels have `STACK:0 LOCAL:0` in the linked resource
table and zero stack frame, spill stores, and spill loads in their matching
ptxas records.

| Kernel | Registers | Shared bytes |
| --- | ---: | ---: |
| CohortRail P2 F32/BF16 | 96 / 96 | 20480 |
| CohortRail P3 F32/BF16 | 94 / 95 | 22528 |
| CohortRail P4 F32/BF16 | 96 / 96 | 24576 |
| M16 singleton F32/BF16 | 95 / 96 | 8192 |
| retained M32 F32/BF16 | 144 / 144 | 24576 |
| retained M64 F32/BF16 | 122 / 122 | 32768 |
| cohort builder | 26 | 4096 |
| diagonal planner | 32 | 1792 |
| PairWave planner | 32 | 7168 |
| W4 owner reducer | 39 | 64 |
| diagonal W2 sums, nonrot/rot | 32 / 31 | 0 |
| diagonal W4 sums, nonrot/rot | 32 / 32 | 0 |
| PairWave reducer W2/W4 | 27 / 28 | 136 |
| PairWave W2 peer sums, nonrot/rot | 19 / 23 | 0 |
| PairWave W4 peer sums, nonrot/rot | 21 / 37 | 0 |
| F32-to-F16 converter | 30 | 0 |
| GDN pre-exp producer | 8 | 0 |

## Arithmetic audit

- M64 F32/BF16: 1024 FFMA, 32 FADD, 4 static BAR each;
- retained M32 F32/BF16: 512 FFMA, 192 FADD, 18 BAR each;
- M16 singleton F32/BF16: 512 FFMA, 32 FADD, 3 BAR each;
- every P2/P3/P4 F32/BF16 specialization:
  512 FFMA, 32 FADD, 16 BAR;
- W4 owner reducer: 32 FFMA, 0 FADD, 2 BAR;
- diagonal canonical W2/W4 sums: 8/16 FADD per instantiation;
- PairWave W2/W4 reducers: 4/8 FFMA and one BAR;
- PairWave W2/W4 peer sums: 8/16 FADD per instantiation;
- planners, cohort builder, converter, and GDN pre-exp producer:
  zero FFMA and FADD.

The retained target blocks contain no LDL, STL, HFMA, HMMA, FP64 arithmetic,
or FP64 conversion. `HADD2.F32` and integer `DEPBAR.LE` are present and are
not half-FMA or FP64 instructions.

## Scope

The full AffinityWave translation unit still contains imported dormant
experimental and fallback kernels with nonzero stack or spills. They are not
reachable from the retained exact-P100 production selectors. Claims in this
audit apply to the 28 retained target kernels, not every compiled historical
variant in the translation unit.
