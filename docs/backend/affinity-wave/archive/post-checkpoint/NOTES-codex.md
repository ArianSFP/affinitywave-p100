# AffinityWave experimental notes - 2026-07-23

## Starting point

- Immutable retained checkpoint:
  `9d3983b89952c7fc1c6aa38fc1a7bd3182992382`.
- Experimental worktree:
  `/home/arian/llama.cpp-qwen36/affinitywave-experimental`.
- Experimental build:
  `/home/arian/llama.cpp-qwen36/build-p100-affinitywave-experimental`.
- Banked exact pp8128 mean: 2874.531 ms, or 2827.6 tok/s.
- Production sources, retained source, and retained build are out of scope.

## Trace-derived hypothesis

The retained Phase-2a trace has recurrent 28-67 ms whole-device idle gaps.
Singleton expert groups are enqueued on one owner compute stream and one copy
stream per GPU. A late `input_ready` wait for the last cell in diagonal `d`
therefore blocks ready work from diagonal `d+1`.

The first experiment gives each singleton group an independent persistent
compute and copy channel. This is not intended to overlap throughput-bound
kernels for extra arithmetic throughput. It lets CUDA select a ready channel
when another channel is waiting on a wave dependency.

### Unconstrained result

- Disabled-feature control: 2874.462 ms, 2827.7 tok/s.
- Four unconstrained channels: 3129.528 ms, 2597.2 tok/s.
- Two paired channels: 3163.889 ms, 2569.0 tok/s.

The control reproduces the checkpoint. Independent channels remove
head-of-line ordering but allow persistent Q8 grids to time-slice, losing more
arithmetic throughput than they recover. Both unconstrained variants are
closed and removed from the experimental source.

The follow-up retains independent readiness queues but uses a one-word
device-side admission lock. At most one expert cell per GPU can execute its
gate/up/down sequence at a time; waiting channels consume only one polling
thread each.

### Admission-lock result

- Unfair CAS admission: 3015.671 ms, 2695.3 tok/s.
- FIFO ticket admission: 3002.328 ms, 2707.2 tok/s.

Serialization recovered about half of the unconstrained regression, and FIFO
admission was slightly better than unfair reacquisition, but both remained
well below the 2874.462 ms control. Polling admission and independent dense
streams still impose too much Pascal time-slicing and critical-path drift.
This path is closed and removed from the experimental source.

## Tail-slack experiment

The retained trace shows the last lane arriving roughly one long attention
cell behind the first three lanes. `GGML_CUDA_AW_TAIL_GAP=1` changes the static
lane offsets from `[0,1,2,3]` to `[0,1,2,4]`. It preserves a single compute
stream per GPU and every original dependency, but gives lane 3 one extra
diagonal of pipeline slack. The work and arithmetic are unchanged; only one
fill/drain diagonal is added.

Result: 2914.783 ms, 2788.5 tok/s. Tail-only slack is 40.321 ms slower than
the experimental control and is closed.

The adjacent follow-up shifts the late pair together, using `[0,1,3,4]`.
This retains lane 2 -> lane 3 spacing while giving both late arrivals one
diagonal of slack.

Result: 2910.398 ms, 2792.7 tok/s. Pair slack is 35.936 ms slower than the
control. Both nonuniform offset variants are closed and removed.

## Trace-fitted token balance

The retained profile gives per-pass home-lane kernel costs:

| Lane | Attention | Other fixed plus recurrent |
|---:|---:|---:|
| 0 | 25.615 ms | 381.541 ms |
| 1 | 65.528 ms | 388.815 ms |
| 2 | 105.844 ms | 387.832 ms |
| 3 | 147.011 ms | 389.395 ms |

Fitting `linear*x + attention*x^2` to those measured totals gives cumulative
token boundaries near `[0,.287,.545,.781,1]`, rather than the old optional
balance `[0,.270,.525,.768,1]`. The experiment changes only those four static
split constants and enables the existing exact token-split path.

Result: 3009.239 ms, 2701.0 tok/s. The model omitted the approximately
207 ms/lane expert-service cost from the linear term. Including that cost moves
the fitted boundaries back near the old optional balance. The trace-fitted
split is closed and the original constants are restored; equal quarters remain
the measured winner.

## Attention lookahead

The benchmark scheduler now has an experimental topological schedule. At a
lane-3 attention frontier it:

1. services the other ready lanes;
2. services the newly ready frontier including the deferred attention cell;
3. advances lane 3 once alone to restore the normal diagonal.

The deferred attention pre-graph runs while the first service wave occupies
the owner GPUs. The catch-up wave prevents the permanent nonuniform offset
that lost in the static slack tests. All 160 cells execute once, dependencies
are asserted while the schedule is built, and the retained single-stream CUDA
service remains unchanged.

Result: 3262.119 ms, 2491.6 tok/s across 52 waves. The extra frontier
fragmentation destroys the beneficial pooling/order of fixed and expert work;
the catch-up policy is substantially worse and is removed.

## Asynchronous state corridor

The generic CUDA cross-backend copy runs P2P transfer on the source main
stream. The destination wait is necessary, but keeping the source main stream
behind the copy is not. The experimental path records source readiness, moves
the transfer to one nonblocking copy stream per source GPU, and leaves the
existing destination event wait intact. This targets only the 668.4 MiB/pass
state corridor; live-service transport is unchanged.

Confirmed pp8128:

- run 1: 2869.115 ms, 2832.9 tok/s;
- run 2: 2867.066 ms, 2835.0 tok/s;
- mean: 2868.091 ms, 2834.0 tok/s.

This is 6.371 ms (0.222%) faster than the 2874.462 ms experimental control.
The change is retained in the experimental source, default off.

## Early state corridor

The next variant records state readiness immediately after each pre-graph and
copies to the next lane on a dedicated source copy stream. The destination
wait is inserted only when that layer is consumed in the following diagonal.
This preserves the same bytes and dependency but exposes the full expert and
post interval for copy-engine overlap.

Confirmed pp8128:

- run 1: 2836.625 ms, 2865.4 tok/s;
- run 2: 2835.983 ms, 2866.0 tok/s;
- mean: 2836.304 ms, 2865.7 tok/s.

This is 38.158 ms (1.327%) faster than the 2874.462 ms experimental control,
and 38.227 ms (1.330%) faster than the 2874.531 ms immutable-checkpoint mean.
The source retains the change behind `GGML_CUDA_AW_CORRIDOR_EARLY`, default
off, while attention-only and recurrent-only treatment are measured.

Type isolation at pp8128:

- recurrent-only: 2867.301 ms, 2834.7 tok/s;
- attention-only: 2846.505 ms, 2855.4 tok/s;
- all layers: 2836.304 ms mean, 2865.7 tok/s.

Attention-state transfer contributes most of the improvement. Recurrent-state
transfer supplies a smaller additional gain, and enabling both is another
10.201 ms faster than attention alone. The next target is the earliest safe
attention K/V state-write boundary, so the transfer can overlap attention
compute rather than only expert and post work.

## Attention state-boundary split (paused before measurement)

Graph inspection identifies the attention layers as 3, 7, ..., 39. Their K/V
state writes are the two `SET_ROWS` nodes immediately before flash attention.
The prototype splits only source lanes 0-2 after the V-cache `SET_ROWS`:

1. the source main stream submits through both K/V state writes;
2. the corridor copy stream waits on that boundary and copies K/V;
3. the source main stream submits the remaining pre-graph, including flash
   attention, without waiting for the transfer;
4. the destination main stream retains the original next-diagonal wait.

This preserves the dependency order while exposing flash attention itself as
copy-overlap time. Lane 3 remains unsplit because it has no outgoing state
corridor. The path is gated by
`GGML_CUDA_AW_CORRIDOR_STATE_SPLIT=1`, requires early corridor mode, and is
default off.

At the user's request, GPU experiments were paused before any smoke or timing
run. A CPU-side build completed successfully with `git diff --check` clean.
Paused build identity:

- `llama-bench`: `aa66ecb19dc7107dc0c88f04842d194600c1733394f864a221cf7a0e168b5689`;
- `libggml-base.so.0.15.3`: `c9d236ea92f1ca35b6a1dfb9b396e39f9803b5b92e26c0a455dbf07333665fa4`;
- `libggml-cuda.so.0.15.3`: `1166f2209ae7c8ba78e1bbae9e5d5e4b9493695c32325a7b861ffe159fcd832c`;
- source diff SHA-256: `64c45fffe2a99762964b60afac752f5b9b68e487a0941a04d03be984860e75f0`.

The first resume gate is a watchdog-protected p1024 smoke:

```sh
bash benchmark-group-streams.sh 1024 0 corridor-state-split-p1024 1111 all 0 1
```

Only after that succeeds should pp8128 be measured:

```sh
bash benchmark-group-streams.sh 8128 0 corridor-state-split-pp8128-run1 1111 all 0 1
```

No correctness or throughput claim is attached to this unmeasured prototype.

## Host-submission starvation

Offline analysis of
`phase2a-service-p8128-gdn2w4-bf16-gp1111.sqlite` changes the experiment
priority. In the timed pass, the host reaches the final stream synchronization
about 2.06 seconds after beginning submission, then spends about 0.82 seconds
draining queued GPU work. The GPU timeline contains repeated 10-30 ms
all-device gaps.

The timed-pass CUDA API breakdown identifies one avoidable source:

- 640 `cudaGetDeviceProperties` calls;
- 395.305 ms total API time;
- 617.664 us mean per call and 25.626 ms maximum.

Across the 2066.858 ms timed-pass submission window, at least one GPU is active
for only 1171.362 ms. All four devices are simultaneously idle for 864.784 ms;
44 gaps of at least 10 ms account for 822.473 ms of that idle time. The longest
25.626 ms property query consumes the available compute slack immediately
before one of the approximately 20 ms global gaps. The query time is therefore
not merely hidden CPU bookkeeping: it is on the starvation path that prevents
the next diagonal from reaching the devices.

`ggml_cuda_affinity_wave_live_service` queried the same immutable owner
properties inside every `(group, owner)` iteration. The experimental source now
caches each validated device's SM count during the existing four-P100 hardware
check and reuses it for all live-service launch geometry. No kernel, arithmetic,
wire format, ordering, or weight format changes.

The change builds cleanly but is unmeasured because the GPU reservation is
released. Current identity:

- `llama-bench`: `aa66ecb19dc7107dc0c88f04842d194600c1733394f864a221cf7a0e168b5689`;
- `libggml-cuda.so.0.15.3`: `d83e2d3f36ddef2f178f37c6bbd747ce36763c8b8f57296d7200f13bda2bd59e`;
- source diff SHA-256: `900257d13fdbfead1855e6dba7be0e253a240d70b801ce9584a5473161ad91ad`.

When measurements resume, isolate this cache before enabling the attention
state-boundary split:

```sh
bash benchmark-group-streams.sh 1024 0 host-cache-p1024 1111 all 0 0
bash benchmark-group-streams.sh 8128 0 host-cache-pp8128-run1 1111 all 0 0
```

The trace also bounds the attention K/V corridor itself at about 14.4 ms per
source edge per pass (two 8.389 MB copies per attention layer). State-boundary
splitting remains useful, but host starvation is the only current experiment
with enough measured headroom to materially close the roughly 514 ms gap to
3500 tok/s.

### Redundant event and empty-tail dispatch removal

Two smaller CPU-only changes stack on the SM-count cache:

1. With `GGML_CUDA_AW_SHARED_SERVICE=0`, `compute_done` and `scratch_free`
   were recorded consecutively at the same compute-stream position. Scratch
   reuse now waits on `compute_done`, and the duplicate `scratch_free` record
   is omitted. This removes 640 event records from a full pass while preserving
   the same dependency.
2. The retained trace contains 5,427 M32 and 5,427 M16 Q8 kernel instances
   across three passes. Every one completed in 2-21 us; none performed a
   short-row GEMM. The M64 kernel already clamps replicated input rows and
   guards output stores with `tile.rows`, so it supports 1-64 rows without
   dropping work. The default-off `GGML_CUDA_AW_M64_ALL=1` path assigns every
   nonempty tail to M64 and omits the M32/M16 launches. On the traced pp8128
   routing this removes 3,618 empty kernel launches per pass and about 14.9 ms
   of empty GPU kernel time. For other routing distributions it trades padded
   M64 arithmetic for fewer launches and therefore requires a separate
   correctness/performance gate.

The updated CPU-only build is clean:

- `llama-affinity-wave-bench`: `c66312b4dd4340d44f72d6b7ba67a6bcbc7924bfbcd04d73f66af65889c0df5e`;
- `llama-bench`: `aa66ecb19dc7107dc0c88f04842d194600c1733394f864a221cf7a0e168b5689`;
- `libggml-cuda.so.0.15.3`: `cf8a7e1a629b3119bc6a5727ec88a87eb9e1fda55b114460c4e4b6ed8da4387a`;
- source diff SHA-256: `75f7ead3a5983bfc6c97e2039e01cd93e9651018185ec4de3c27de637a15e8e6`.

Resume order remains deliberately isolated:

```sh
# SM cache plus redundant-event removal, M64-all disabled
bash benchmark-group-streams.sh 1024 0 host-dispatch-p1024 1111 all 0 0 0
bash benchmark-group-streams.sh 8128 0 host-dispatch-pp8128-run1 1111 all 0 0 0

# Then test M64-all separately
bash benchmark-group-streams.sh 1024 0 m64-all-p1024 1111 all 0 0 1
bash benchmark-group-streams.sh 8128 0 m64-all-pp8128-run1 1111 all 0 0 1
```

No GPU run or correctness claim has been made for these paused experiments.

### Resumed host-dispatch and M64-all measurements

The user released the four GPUs for this work. Every run used the isolated
experimental build, the clean `env -i` stack, CPU affinity 0-11, and the
`.xsession-errors` watchdog.

The cached-SM-count plus redundant-event build, with retained early corridors
enabled and M64-all disabled, measured:

- p1024: 576.606 ms, 1775.9 tok/s;
- pp8128 run 1: 2837.538 ms, 2864.5 tok/s;
- pp8128 run 2: 2837.630 ms, 2864.4 tok/s;
- pp8128 mean: 2837.584 ms, 2864.4 tok/s.

The previous early-corridor mean was 2836.304 ms, so these host-only changes
are neutral within run variance. The 395.305 ms of profiled
`cudaGetDeviceProperties` API time was mostly driver backpressure rather than
recoverable wall time. The validated SM-count cache is safe to retain, but it
is closed as a throughput lever.

`GGML_CUDA_AW_M64_ALL=1` measured:

- p1024: 1011.501 ms, 1012.4 tok/s;
- pp8128 run 1: 2831.989 ms, 2870.1 tok/s;
- pp8128 run 2: 2833.283 ms, 2868.8 tok/s;
- pp8128 mean: 2832.636 ms, 2869.4 tok/s.

M64-all improves pp8128 by only 4.948 ms (0.175%) relative to the concurrent
host-dispatch control while regressing p1024 by 75%. It is not suitable as a
general retained configuration and remains default-off for diagnostic use.
The ordinary output-accuracy path remains withheld in this service prototype;
no new exactness claim is attached to M64-all.

### Attention state-boundary split

`GGML_CUDA_AW_CORRIDOR_STATE_SPLIT=1` publishes the attention K/V state after
the K/V writes and before the local flash-attention suffix. The destination
still waits on the same event before its next dependent diagonal.

Measurements with retained early corridors enabled and M64-all disabled:

- p1024: 576.650 ms, 1775.8 tok/s;
- pp8128 run 1: 2825.944 ms, 2876.2 tok/s;
- pp8128 run 2: 2826.250 ms, 2875.9 tok/s;
- pp8128 mean: 2826.097 ms, 2876.0 tok/s.

The pp8128 mean improves by 11.487 ms (0.406%) relative to the concurrent
host-dispatch control, while p1024 is neutral. This is retained for the next
experiment. Its gain confirms that the K/V corridor was slightly late, but
the corridor is too small to explain most of the remaining gap to 3500 tok/s.

### Persistent owner submission pool: closed and removed

A default-off persistent pool assigned owners 1-3 to stable worker threads
while the caller submitted owner 0. Only the compute submission phase was
parallelized; GPU stream order, events, arithmetic, and transport were
unchanged.

- p1024: 576.565 ms, 1776.0 tok/s;
- pp8128 run 1: 2825.306 ms, 2876.9 tok/s.

Relative to the 2826.097 ms state-split control mean, the pp8128 difference is
0.791 ms (0.028%), well inside run variance. The result disproves the
hypothesis that serial per-owner launch API calls are the critical wall-time
path: the long API durations in the old trace mainly represent driver
backpressure while other devices are working.

Per the complexity gate, no second pp8128 run was warranted. The pool class,
threading includes, compute-loop refactor, environment control, and benchmark
script argument were removed. The source again uses the original serial
group/owner loop and builds cleanly. This experiment is documentation only.

GPU reservation was then released at the user's request before the planned
current-best nsys capture began. No profiling process was launched. Offline
analysis may continue, but no further GPU measurement will run without fresh
user clearance.

### `CUDA_SCALE_LAUNCH_QUEUES=4x`: closed

After fresh GPU clearance, the immutable checkpoint binary and complete
retained environment were tested in ABBA order. Each arm ran in a new process;
the environment variable was completely omitted for control runs and set
before the first CUDA call for treatment runs.

| Order | Queue setting | pp8128 elapsed | Throughput |
|---|---|---:|---:|
| A1 | omitted | 2874.676 ms | 2827.4 tok/s |
| B1 | `4x` | 2875.897 ms | 2826.2 tok/s |
| B2 | `4x` | 2876.923 ms | 2825.2 tok/s |
| A2 | omitted | 2874.680 ms | 2827.4 tok/s |
| A mean | omitted | 2874.678 ms | 2827.447 tok/s |
| B mean | `4x` | 2876.410 ms | 2825.745 tok/s |

The treatment is 1.732 ms slower, a -0.0602% throughput change. A p1024
`4x` smoke run completed safely at 574.329 ms / 1783.0 tok/s. No CUDA error,
watchdog trip, OOM, or hang occurred.

This disproves launch-command-buffer capacity as the cause of the retained
trace bubbles on the CUDA 12.8 / driver 580 / four-P100 configuration. The
long CUDA API durations are backpressure from already-scheduled device work or
other dependencies, not a queue-depth limit recoverable by the runtime knob.
No backend or retained command change is justified.

Artifacts:

- `benchmark-launch-queues.sh`;
- `launch-queue-smoke-4x-p1024.err`;
- `launch-queue-abba-a1-omit-p8128.err`;
- `launch-queue-abba-b1-4x-p8128.err`;
- `launch-queue-abba-b2-4x-p8128.err`;
- `launch-queue-abba-a2-omit-p8128.err`.

A current-best nsys capture was started after the A/B, but the fixed profiling
window ended before the service printed a timed result. It produced
`current-best-p8128.nsys-rep` (SHA-256
`1ca0c50161303e3772defedba0d2a6746ae6fa8877f233d1218e0ece580ce52a`).
Treat it as an incomplete diagnostic capture, not a benchmark or performance
claim. No analysis or implementation followed it.

GPU experiments then stopped at the user's request. The reservation is
released, with no llama, nsys, benchmark, or watchdog process remaining.

## Next structural candidate: per-owner host submission

The remaining timed-pass host cost is dominated by `cudaLaunchKernel`:
9280 calls consume 1336.753 ms of API time. The live-service compute loop
submits owner 0, 1, 2, and 3 serially even though they target independent CUDA
devices, streams, scratch states, and weight pointers. This recreates the same
host serialization that the existing meta-backend submission pool already
removed from ordinary multi-GPU graph submission.

A dependency-preserving persistent owner pool would use three workers plus the
caller:

1. input publication is submitted independently by each home device, followed
   by a host-only barrier ensuring every `input_ready` record has been issued;
2. each owner thread walks all groups in the existing order on exactly one
   device, followed by a host-only barrier;
3. each owner thread submits its output copies in group order, followed by a
   barrier ensuring every `output_ready` record has been issued;
4. home reductions retain their existing event waits and stream order.

There is no device synchronization in those barriers and no arithmetic,
grouping, or event dependency changes. Thread affinity is stable across calls,
matching the proven `submit_pool` pattern in `ggml-backend-meta.cpp`.
Idealized launch-API overlap has more than enough headroom to close the
remaining gap to 3500 tok/s, but this is a larger scheduling change. It is
documented rather than implemented while GPU experiments are paused.
