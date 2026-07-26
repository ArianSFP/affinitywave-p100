# Diagonal bounded-service audit

## Scope and isolation

This is the live record for the bounded-memory diagonal service work started
on 2026-07-25. The immutable source reference is
`ae2b41d682e0c18f5c7277860dd0566244413fcb`. All source, builds, traces, and
results are confined to the detached
`affinitywave-frontier-20260724` worktree. The production worktree and its
untracked captures have not been modified.

The objective is to preserve the accurate production diagonal scheduler while
removing approximately 1 GiB/GPU of context-sized service materializations.
The user's clarified acceptance boundary is approximately 1 GiB/GPU rather
than an exact minimum of 1 GiB. The mode is default-off:

`GGML_CUDA_AW_DIAGONAL_SERVICE=panel512`

## First implementation

The first exact implementation retains per-group route metadata but replaces
the large per-group tensors with one device-shared service arena:

- one FP32 input tensor;
- one full-width FP32 middle tensor;
- one routed FP32 N512 output panel;
- two BF16 owner-partial panel slots;
- two four-owner BF16 receive panel slots;
- a high-priority publication stream and CUDA-event dependencies.

Gate and up use the existing exact T64 projection kernels. The existing exact
SwiGLU kernel overwrites the gate result in place. Four N512 down projections
immediately perform the original rank-ordered weighted reduction and BF16
owner-boundary rounding. Each completed panel is copied to the logical home
and summed in the original rotated owner order.

The selector requires the accurate legacy service, FP32 request transport,
BF16 owner partials, deterministic route construction, and the production
one-cell group pattern. It does not alter public APIs or default behavior.

At 512 tokens total, the reported arena is 6,553,600 bytes/GPU. At 2,048
tokens total, the arena is 26,214,400 bytes/GPU. The measured allocator delta
is 27,262,976 bytes on GPU 0 and 29,360,128 bytes on GPUs 1-3. Linear scaling
to the production 8,128-token prompt predicts about 104.9 MB of major arena
storage, versus 1,166,661,424 bytes/GPU for the existing live state. This is
approximately a 0.99 GiB/GPU reduction before allocator granularity.

## Accuracy

The pp512 smoke test completes at 609.195 ms and 840.453 tok/s.

All four layer-0 reduced FP32 outputs are byte-identical to the accurate
legacy service:

| home | SHA-256 |
| ---: | --- |
| 0 | `3f19b36cb3d7004a49c4a68e48e84a0716e0e4eb3cdd0f718f34090a682dadee` |
| 1 | `95eca717bc0967387a268aedae3de37410956ff22bbcdbcd25653d84691c11fa` |
| 2 | `dd32d8d0d38f7d1c4320cc1313217ddcfaa4e8847eb8a4cc0d5f824dd3c7213c` |
| 3 | `727f15358e0e1b958def0c9cbbc458e22f341b6f84cc9be151a85aadeaabbae4` |

Matched pp512 KLD reports are also identical:

- mean PPL(Q): 4.078325;
- mean PPL(base): 4.082694;
- PPL difference: -0.004369;
- mean KLD: 0.000689;
- same-top-token agreement: 99.216%.

The absolute PPL difference is inherited from the matched accurate legacy
path. The candidate introduces no measured accuracy change.

## First performance result and falsification

Matched pp2048 timings:

| mode | time | rate |
| --- | ---: | ---: |
| accurate legacy BroadWave | 922.441505 ms | 2,220.195 tok/s |
| first panel512 prototype | 1,132.844840 ms | 1,807.838 tok/s |

The 210.403 ms wall regression is much larger than the N512 arithmetic tax
measured by the isolated service probe. A matched Nsight Systems comparison
explains the difference.

On critical GPU 0, the legacy path has 403.549 ms of exact-Q8 projection work:

- M64: 158.354 ms, 480 launches;
- M32: 73.189 ms, 480 launches;
- M16: 172.006 ms, 480 launches.

The first panel512 path has 451.064 ms:

- M64: 184.226 ms, 960 launches;
- M32: 83.008 ms, 960 launches;
- M16: 183.830 ms, 960 launches.

Gate plus up plus four down panels double the projection launch count from
three to six projection waves. This adds 47.515 ms of measured Q8 work.
Memcpy volume is unchanged at 1,067.816 MiB on GPU 0 and memcpy duration
changes only from 82.666 to 86.349 ms. The new owner-sum kernels add 4.594 ms.

Within the timed-pass NVTX interval, kernel-plus-memcpy busy/idle union is:

| mode | window | GPU 0 busy | GPU 0 idle |
| --- | ---: | ---: | ---: |
| accurate legacy | 918.079 ms | 847.958 ms | 70.121 ms |
| first panel512 | 1,138.875 ms | 897.233 ms | 241.642 ms |

The other three GPUs show the same result. The candidate adds approximately
49 ms of busy work but approximately 172 ms of idle time. The dominant
regression is therefore not transfer volume or down-panel arithmetic.

The cause is the single shared input slot. Before copying group `g + 1`, the
home copy stream waits for `input_free` from every owner of group `g`. This
turns scratch reuse into a slowest-owner gang barrier and destroys the
diagonal scheduler's cross-group skew. The legacy scheduler queues all four
group inputs before compute and does not introduce this boundary.

## Next experiments

1. Add two bounded input slots, selected by group parity. This permits the
   next group to stage while the current group computes and removes the
   adjacent-group all-owner wait without persistent replication.
2. Re-run pp2048 and a matched trace. Retain only if the added idle time
   collapses.
3. Compare N512 with N1024. The isolated N-panel audit measured N1024 at
   16.9017 ms/service versus 17.1291 ms for N512. N1024 halves down-panel
   launches while still predicting about 0.92 GiB/GPU of savings with two
   input slots.
4. Do not run pp8128 until the pp2048 trace demonstrates that bounded
   storage preserves the production diagonal wave.

## Scheduler repair and N1024 boundary

The two-input-slot experiment falsified input capacity alone as the cause.
It measured 1,130.811853 ms at pp2048, only 2.033 ms faster than the original
single-slot prototype.

The actual ordering defect was broader. The first helper enqueued every
group's output copies and home-stream publication wait before enqueueing the
next group's compute. The production scheduler instead stages all group
inputs and queues all owner compute before inserting home publication waits.
Because input and output share each device's copy stream, the first helper
also placed group-0 output traffic ahead of group-1 input traffic.

The repaired schedule uses four bounded input shards, one for each diagonal
group. It stages all four inputs first, queues group compute and panel
publication second, and inserts the four home-stream output waits only after
all group compute has been submitted. Shared middle and route-output storage
remains serial on each owner stream.

At N512, this reduces pp2048 from 1,130.811853 to 999.828028 ms. A matched
trace has a 1,001.828 ms timed window, 904.973 ms GPU-0 busy union, and
96.855 ms idle. Relative to legacy, the remaining differences are about
57.0 ms busy and 26.7 ms idle. Restoring production ordering therefore
recovers approximately 145 ms of the original trace regression.

N1024 is the retained boundary:

| mode | pp2048 time | rate | major arena |
| --- | ---: | ---: | ---: |
| accurate legacy | 922.441505 ms | 2,220.195 tok/s | 1,166,661,424 B/GPU at pp8128 |
| repaired N512 | 999.828028 ms | 2,048.352 tok/s | 38,797,312 B/GPU at pp2048 |
| repaired N1024 | 949.117039 ms | 2,157.795 tok/s | 52,428,800 B/GPU at pp2048 |

N1024 uses two down panels instead of four. It halves down projection,
owner-reduction, owner-copy, and owner-sum launch counts while retaining the
same exact arithmetic boundaries. Scaling its measured 50 MiB pp2048 major
arena from 512 to 2,032 tokens/cell predicts 208,076,800 bytes/GPU at
pp8128. Relative to the measured legacy live state, the predicted saving is
958,584,624 bytes/GPU, or approximately 0.893 GiB/GPU. This is within the
user's clarified approximately-1-GiB boundary.

The N1024 layer-0 outputs match the legacy hashes listed above byte for byte.
Its pp512 KLD report is also identical to both legacy and N512:

- mean PPL(Q): 4.078325;
- mean PPL(base): 4.082694;
- PPL difference: -0.004369;
- mean KLD: 0.000689;
- same-top-token agreement: 99.216%.

The next acceptance gate is a complete pp8128 timing and actual allocation
measurement. N512 remains available only as a higher-capacity diagnostic.

## Full-context results and N2048 boundary

N1024 passed the full-context gate. Three BroadWave pp8128 trials measured
3,017.434627, 3,019.532890, and 3,016.756432 ms, for a mean of
3,017.907983 ms or 2,693.256 tok/s. The reported major arena is
208,076,800 bytes/GPU and the measured CUDA allocation delta is
211.8-213.9 MB/GPU. Relative to the 1,166,661,424-byte legacy live state,
this reclaims approximately 0.887 GiB/GPU after allocator granularity.

The matching N1024 production trace has a 2,998.725 ms window on critical
GPU 3. Its busy union falls by 138.129 ms relative to the BroadWave
production trace while idle time changes by only about 5 ms. Exact-Q8 work
is slightly higher, 841.000 ms versus 824.097 ms, and copy volume is
unchanged. The gain comes from publishing bounded panels early enough to
hide service output traffic behind later work, not from reducing arithmetic
or communication bytes.

N2048 uses one full-width down panel. The first BroadWave trial measured
3,009.408393 ms; HalfPipe measured 2,994.274284 ms. Dynamic ring sizing now
allocates one partial/receive slot for N2048 instead of two. The resulting
major arena is 274,661,376 bytes/GPU and the measured allocation is
276.8-281.0 MB/GPU. This reclaims 892,000,048 bytes, or approximately
0.831 GiB/GPU, from the legacy major state. The user explicitly accepted
approximately 1 GiB/GPU rather than an exact 1.000 GiB requirement, so this
is a valid speed/capacity point.

The lane-stagger controls did not expose another scheduling gain:

| N1024 lane stagger | pp8128 time |
| ---: | ---: |
| 1 | 3,017.908 ms mean |
| 2 | 3,105.972 ms |
| 3 | 3,262.706 ms |
| 4 | 3,177.771 ms |

Stagger 1 remains required.

## Exact dense selector composition

The prior complete cuBLAS selector sweep identified bitwise-identical faster
algorithms for each live N=2032 dense shape:

| shape | production default | retained exact selector |
| --- | ---: | ---: |
| M8192, K2048 | 10.195888 ms | algorithm 6, 8.140240 ms |
| M2048, K4096 | 5.217104 ms | algorithm 10, 4.218656 ms |
| M4096, K2048 | 4.695648 ms | algorithm 5, 4.151168 ms |
| M2048, K512 | 0.636512 ms | algorithm 6, 0.566016 ms |

`GGML_CUDA_AW_DENSE_SELECTORS=exact` applies only those four
shape-and-device-specific choices on Pascal. It remains default-off and
leaves all other cuBLAS calls unchanged. The isolated probes compare every
output value against production algorithm 99 and report zero bitwise
mismatches.

Composed with N2048 and `halfpipe_sync`, three pp8128 trials measured:

| trial | time | rate |
| ---: | ---: | ---: |
| 1 | 2,928.068508 ms | 2,775.891 tok/s |
| 2 | 2,927.837806 ms | 2,776.110 tok/s |
| 3 | 2,926.757472 ms | 2,777.135 tok/s |
| mean | 2,927.554595 ms | 2,776.379 tok/s |

This is 230.275 ms faster than the accurate 3,157.830 ms production mean,
a 7.29% time reduction and a 7.87% token-rate increase. It remains
218.221 ms short of the 2,709.333 ms required for 3,000 tok/s.

All four intended selectors were observed in the live run. A layer-0
N2048-plus-HalfPipe service dump matches all four legacy reduced FP32 output
files byte for byte. The HalfPipe kernel had already passed its full
component bitwise gate, and the dense-selector probes are bitwise exact.

One launch attempted the shorthand engine value `halfpipe`; configuration
validation rejected it before model loading. The correct selector is
`halfpipe_sync`; no measurement was attributed to the rejected launch.

The accepted trace
`diagonal-panel2048-halfpipe-sync-dense-exact-p8128-trace-v1` has a
2,958.189 ms timed window. Critical GPU 3 records:

- 2,513.303 ms busy union and 444.887 ms idle;
- 814.727 ms exact-Q8 projections;
- 440.055 ms FlashAttention;
- 285.338 ms memcpy union;
- 144.675 ms GDN;
- 37.956 ms owner reduction and 9.727 ms owner sum.

The perfect-balance device-work floor is 2,382.213 ms. Reaching 3,000 tok/s
therefore remains physically possible, but requires removing exposed
dependency gaps as well as work. More isolated Q8 tile tuning is not a
sufficient next step.

## Evidence

- `diagonal-panel512-smoke-v1-bench-p512-ub512.{out,err}`
- `diagonal-panel512-dump-legacy-v1/`
- `diagonal-panel512-dump-candidate-v1/`
- `diagonal-panel512-kld-legacy-v1-ppl-aw-p512-ub512.{out,err}`
- `diagonal-panel512-kld-v1-ppl-aw-p512-ub512.{out,err}`
- `diagonal-panel512-legacy-p2048-v1-bench-p2048-ub2048.{out,err}`
- `diagonal-panel512-p2048-v1-bench-p2048-ub2048.{out,err}`
- `route-scatter-det-live-p2048-v1.{nsys-rep,sqlite}`
- `diagonal-panel512-p2048-trace-v1.{nsys-rep,sqlite,analysis.md,analysis.json}`
- `diagonal-panel512-input2-p2048-v1-bench-p2048-ub2048.{out,err}`
- `diagonal-panel512-phased4-p2048-v1-bench-p2048-ub2048.{out,err}`
- `diagonal-panel512-phased4-p2048-trace-v1.{nsys-rep,sqlite}`
- `diagonal-panel1024-phased4-p2048-v1-bench-p2048-ub2048.{out,err}`
- `diagonal-panel1024-dump-candidate-v1/`
- `diagonal-panel1024-kld-v1-ppl-aw-p512-ub512.{out,err}`
- `diagonal-panel1024-p8128-v{1,2,3}-bench-p8128-ub8128.{out,err}`
- `diagonal-panel1024-p8128-trace-v1.{nsys-rep,sqlite}`
- `diagonal-panel2048-halfpipe-p8128-v1-bench-p8128-ub8128.{out,err}`
- `temporal-dense-selector-*-v1.jsonl`
- `diagonal-panel2048-halfpipe-sync-dense-exact-p8128-v{1,2,3}-bench-p8128-ub8128.{out,err}`
- `diagonal-panel2048-halfpipe-dump-candidate-v1/`
- `diagonal-panel2048-halfpipe-sync-dense-exact-p8128-trace-v1.{nsys-rep,sqlite,analysis.md,analysis.json}`
