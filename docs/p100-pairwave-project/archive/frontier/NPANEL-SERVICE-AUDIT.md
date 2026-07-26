# Exact N-panel MoE service audit

## Mechanism

The down projection was decomposed along its 2,048 output columns. Each
panel uses a panel-adjusted exact-T64 descriptor, writes only one
route-output panel, and immediately performs the original rank-ordered FP32
weighted reduction with the same BF16 owner-boundary rounding. Panel-major
BF16 output is suitable for direct peer transfer without a transpose.

The service oracle checks the complete 8,192-token by 2,048-column result.
Every tested panel size has zero mismatches across 16,777,216 values on each
of four GPUs.

## Compute and memory sweep

The matched control is four-cell coalesced BroadWave on the edge-route mix.
The down column panels include the reduction after every panel.

| panels | columns/panel | down plus reduce | full service stage | device bytes | bytes removed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,048 | 5.6815 ms | 16.6461 ms | 688,088,708 | 0 |
| 2 | 1,024 | 5.9048 ms | 16.9017 ms | 620,985,988 | 67,102,720 |
| 4 | 512 | 6.1417 ms | 17.1291 ms | 587,437,700 | 100,651,008 |
| 8 | 256 | 6.6181 ms | 17.5428 ms | 570,672,772 | 117,415,936 |
| 16 | 128 | 8.1645 ms | 19.1169 ms | 562,308,740 | 125,779,968 |

N128 is closed on Pascal. Sixteen launches repeatedly expose only 16 M32 and
42 M16 tiles, destroying the output-column parallelism that kept the tail
kernels resident.

N512 is the best down-only memory boundary. It removes 96 MiB from the
isolated service and costs 0.4603 ms in down plus reduction. Replacing the
original eight tile-local N128 down slots with one full-route N512 panel
changes the bounded arena from 97,725,440 to 131,017,728 bytes/GPU. It
remains below the 128 MiB cap by 3,200,000 bytes.

The complete gate/up/down composition changes the retained down panel to
N256. N256 gate/up plus N256 down fits in 123,087,872 bytes/GPU and is faster
than N128 gate/up plus N512 down. See
`MATERIALIZATION-FREE-SERVICE-AUDIT.md`.

## Four-GPU copy pipeline falsification

The copy probe sends exactly the remote BF16 owner partials produced by the
four synchronized owners:

- four panels;
- one 2,048-token home slice per destination;
- three remote destinations per source;
- 96 MiB total remote traffic per service across each source GPU;
- compute of panel p+1 overlaps copies of panel p;
- the final compute event waits for every peer copy.

Two schedules were measured:

| schedule | down/reduce/copy mean | maximum | exposed over N512 compute |
| --- | ---: | ---: | ---: |
| simultaneous destination streams | 15.1327 ms | 16.3269 ms | 8.9910 ms mean |
| one source stream, rotated destinations | 21.9331 ms | 23.0024 ms | 15.7914 ms mean |

Both remain bitwise exact. The PCIe switch, not panel readiness, is the hard
resource. The proposal required at most 3 ms of exposed dispatch plus combine
per synchronized layer; the measured copy exposure alone is three to five
times that gate. Same-layer row pooling and panel-copy service are closed on
this rig.

The N512 compute layout remains useful only as a low-memory implementation
inside the existing diagonal wave, where owner-output copies can overlap
work from other layers. It is not credited with a token-rate gain until a
production trace demonstrates that overlap.

Evidence:

- `n128-service-base-v1-service-p2048-ub2048.out`
- `n128-service-panel2-v1-service-p2048-ub2048.out`
- `n128-service-panel4-v2-service-p2048-ub2048.out`
- `n128-service-panel8-v1-service-p2048-ub2048.out`
- `n128-service-panel16-v1-service-p2048-ub2048.out`
- `n128-service-panel4-copy-v1-service-p2048-ub2048.out`
- `n128-service-panel4-copy-ring-v1-service-p2048-ub2048.out`
