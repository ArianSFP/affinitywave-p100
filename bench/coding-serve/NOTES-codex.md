# Codex findings

2026-09-07: MTP is not enabled by `server_probe.py` or `run.py`. The probe's
`n_predict` values only limit generated output. The runner's server command
contains no `--spec-type draft-mtp`, `--mtp`, or MTP environment variable, and
the recorded `webui-live-20260907.meta.json` launch has no speculative settings.

2026-09-07: Host-level P100 qualification (`p100-qualification-8128-20260907`)
completed on all four GPUs after the managed sandbox recovered no device nodes.
Fixed-shape 8128-token `llama-bench`, 4 repeats, reported 2900.109901 +/-
3.095575 tok/s (samples 2899.7, 2897.03, 2904.4, 2899.3). This is 0.417% below
the 2912.264 checkpoint. The run exited cleanly and left no compute clients.
