#!/usr/bin/env python3
"""AffinityWave Phase-0 placement and discrete-event feasibility model.

The tool accepts either the compact token-routing format documented in README.md
or the older aggregate GGML_CUDA_MOE_HIST format. Aggregate input is deliberately
marked as a proxy: it cannot recover per-token remote-owner co-occurrence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


N_GPU = 4
N_LAYER = 40
N_EXPERT = 256
N_USED = 8
HIDDEN = 2048
EXPERT_FF = 512
AWTR_MAGIC = b"AWTRV001"
AWTR_HEADER = struct.Struct("<HIH")  # layer, token count, experts/token


@dataclass(frozen=True)
class RouteObservation:
    layer: int
    n_tokens: int
    counts: tuple[int, ...]
    routes: tuple[tuple[int, ...], ...] | None = None


@dataclass(frozen=True)
class TraceCosts:
    passes: int
    span_s: float
    kernel_s_per_gpu: tuple[float, ...]
    nccl_s_per_gpu: tuple[float, ...]
    expert_s_per_gpu: tuple[float, ...]
    get_rows_s_per_gpu: tuple[float, ...]
    plan_s_per_gpu: tuple[float, ...]
    fixed_lower_s_per_gpu: tuple[float, ...]
    top_kernels: tuple[tuple[str, int, float], ...]


def sha256_file(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _validate_observations(observations: Sequence[RouteObservation]) -> None:
    if not observations:
        raise ValueError("routing input contained no gate-expert observations")
    for obs in observations:
        if not 0 <= obs.layer < N_LAYER:
            raise ValueError(f"layer {obs.layer} is outside [0,{N_LAYER})")
        if len(obs.counts) != N_EXPERT:
            raise ValueError(f"layer {obs.layer} has {len(obs.counts)} expert counts")
        if sum(obs.counts) != obs.n_tokens * N_USED:
            raise ValueError(
                f"layer {obs.layer}: counts sum {sum(obs.counts)}, expected "
                f"{obs.n_tokens * N_USED}"
            )


def load_hist(path: Path) -> list[RouteObservation]:
    observations: list[RouteObservation] = []
    layer_re = re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$")
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            fields = line.split()
            if len(fields) < 3:
                continue
            match = layer_re.match(fields[0])
            if match is None:
                continue
            layer = int(match.group(1))
            n_tokens = int(fields[1])
            counts = [int(value) for value in fields[2:2 + N_EXPERT]]
            if len(counts) < N_EXPERT:
                counts.extend([0] * (N_EXPERT - len(counts)))
            if any(value < 0 for value in counts):
                raise ValueError(f"{path}:{line_no}: negative expert count")
            observations.append(RouteObservation(layer, n_tokens, tuple(counts)))
    _validate_observations(observations)
    return observations


def load_awtr(path: Path) -> list[RouteObservation]:
    observations: list[RouteObservation] = []
    with path.open("rb") as handle:
        magic = handle.read(len(AWTR_MAGIC))
        if magic != AWTR_MAGIC:
            raise ValueError(f"{path}: bad AffinityWave trace magic {magic!r}")
        while header := handle.read(AWTR_HEADER.size):
            if len(header) != AWTR_HEADER.size:
                raise ValueError(f"{path}: truncated record header")
            layer, n_tokens, n_used = AWTR_HEADER.unpack(header)
            if n_used != N_USED:
                raise ValueError(f"{path}: record uses top-{n_used}, expected top-{N_USED}")
            payload_size = n_tokens * n_used * 2
            payload = handle.read(payload_size)
            if len(payload) != payload_size:
                raise ValueError(f"{path}: truncated layer {layer} payload")
            flat = struct.unpack(f"<{n_tokens * n_used}H", payload)
            routes = tuple(tuple(flat[i:i + n_used]) for i in range(0, len(flat), n_used))
            counts = [0] * N_EXPERT
            for token_routes in routes:
                if len(set(token_routes)) != n_used:
                    raise ValueError(f"{path}: duplicate expert in layer {layer} token route")
                for expert in token_routes:
                    if expert >= N_EXPERT:
                        raise ValueError(f"{path}: expert {expert} is out of range")
                    counts[expert] += 1
            observations.append(RouteObservation(layer, n_tokens, tuple(counts), routes))
    _validate_observations(observations)
    return observations


def load_routes(path: Path) -> tuple[list[RouteObservation], str]:
    with path.open("rb") as handle:
        magic = handle.read(len(AWTR_MAGIC))
    if magic == AWTR_MAGIC:
        return load_awtr(path), "token_routes"
    return load_hist(path), "aggregate_histogram_proxy"


def by_layer(observations: Iterable[RouteObservation]) -> dict[int, list[RouteObservation]]:
    result = {layer: [] for layer in range(N_LAYER)}
    for obs in observations:
        result[obs.layer].append(obs)
    missing = [layer for layer, values in result.items() if not values]
    if missing:
        raise ValueError(f"routing input is missing layers: {missing}")
    return result


def split_calibration_holdout(
    observations: Sequence[RouteObservation],
) -> tuple[dict[int, list[RouteObservation]], dict[int, list[RouteObservation]]]:
    grouped = by_layer(observations)
    calibration: dict[int, list[RouteObservation]] = {}
    holdout: dict[int, list[RouteObservation]] = {}
    for layer, values in grouped.items():
        midpoint = max(1, len(values) // 2)
        calibration[layer] = values[:midpoint]
        holdout[layer] = values[midpoint:] or values[-1:]
    return calibration, holdout


def aggregate_counts(observations: Sequence[RouteObservation]) -> list[int]:
    return [sum(obs.counts[e] for obs in observations) for e in range(N_EXPERT)]


def placement_score(observations: Sequence[RouteObservation], owners: Sequence[int], hot: set[int]) -> int:
    score = 0
    for obs in observations:
        loads = [0] * N_GPU
        for expert, count in enumerate(obs.counts):
            if expert not in hot:
                loads[owners[expert]] += count
        score += max(loads)
    return score


def placement_skew(
    observations: Sequence[RouteObservation], owners: Sequence[int], hot: set[int],
) -> dict[str, float]:
    values: list[float] = []
    for obs in observations:
        loads = [0] * N_GPU
        for expert, count in enumerate(obs.counts):
            if expert not in hot:
                loads[owners[expert]] += count
        mean = sum(loads) / N_GPU
        values.append(max(loads) / mean if mean else 1.0)
    values.sort()
    p95 = values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)]
    return {"mean": sum(values) / len(values), "p95": p95, "max": values[-1]}


def optimize_layer_placement(
    calibration: Sequence[RouteObservation], hot_count: int, swap_rounds: int,
) -> tuple[list[int], list[int]]:
    weights = aggregate_counts(calibration)
    hot = sorted(range(N_EXPERT), key=lambda e: (-weights[e], e))[:hot_count]
    hot_set = set(hot)

    # Capacity-constrained longest-processing-time placement. Primary copies include
    # replicas, so every rank retains exactly 64 primary experts.
    owners = [-1] * N_EXPERT
    rank_weight = [0] * N_GPU
    rank_count = [0] * N_GPU
    rank_hot_count = [0] * N_GPU
    for expert in hot:
        rank = min(range(N_GPU), key=lambda value: (rank_hot_count[value], rank_weight[value], value))
        owners[expert] = rank
        rank_weight[rank] += weights[expert]
        rank_count[rank] += 1
        rank_hot_count[rank] += 1
    for expert in sorted((e for e in range(N_EXPERT) if e not in hot_set), key=lambda e: (-weights[e], e)):
        candidates = [rank for rank in range(N_GPU) if rank_count[rank] < N_EXPERT // N_GPU]
        rank = min(candidates, key=lambda value: (rank_weight[value], rank_count[value], value))
        owners[expert] = rank
        rank_weight[rank] += weights[expert]
        rank_count[rank] += 1

    # Deterministic best-improvement swaps use calibration observations only. The
    # second half remains untouched for the reported held-out score.
    effective = [
        tuple(0 if expert in hot_set else obs.counts[expert] for expert in range(N_EXPERT))
        for obs in calibration
    ]
    loads = []
    for row in effective:
        rank_loads = [0] * N_GPU
        for expert, count in enumerate(row):
            rank_loads[owners[expert]] += count
        loads.append(rank_loads)
    current = sum(max(rank_loads) for rank_loads in loads)
    for _ in range(swap_rounds):
        best_score = current
        best_pair: tuple[int, int] | None = None
        for left in range(N_EXPERT):
            for right in range(left + 1, N_EXPERT):
                left_rank, right_rank = owners[left], owners[right]
                if left_rank == right_rank or (left in hot_set) != (right in hot_set):
                    continue
                candidate = 0
                for row, rank_loads in zip(effective, loads):
                    left_load = rank_loads[left_rank] - row[left] + row[right]
                    right_load = rank_loads[right_rank] - row[right] + row[left]
                    other_load = max(
                        rank_loads[rank]
                        for rank in range(N_GPU)
                        if rank != left_rank and rank != right_rank
                    )
                    candidate += max(left_load, right_load, other_load)
                if candidate < best_score:
                    best_score = candidate
                    best_pair = (left, right)
        if best_pair is None:
            break
        left, right = best_pair
        left_rank, right_rank = owners[left], owners[right]
        for row, rank_loads in zip(effective, loads):
            rank_loads[left_rank] += row[right] - row[left]
            rank_loads[right_rank] += row[left] - row[right]
        owners[left], owners[right] = right_rank, left_rank
        current = best_score

    assert [owners.count(rank) for rank in range(N_GPU)] == [N_EXPERT // N_GPU] * N_GPU
    assert [sum(owners[e] == rank for e in hot) for rank in range(N_GPU)] == [hot_count // N_GPU] * N_GPU
    return owners, hot


def build_manifest(
    calibration: dict[int, list[RouteObservation]],
    holdout: dict[int, list[RouteObservation]],
    gguf_sha256: str,
    routing_source: str,
    hot_count: int,
    swap_rounds: int,
) -> tuple[dict, dict[int, list[int]], dict[int, set[int]]]:
    if not re.fullmatch(r"[0-9a-f]{64}", gguf_sha256):
        raise ValueError("GGUF SHA-256 must be 64 lowercase hexadecimal characters")
    if hot_count < 0 or hot_count > N_EXPERT or hot_count % N_GPU != 0:
        raise ValueError("replicated expert count must be in [0,256] and divisible by four")
    layers = []
    owner_map: dict[int, list[int]] = {}
    hot_map: dict[int, set[int]] = {}
    for layer in range(N_LAYER):
        owners, hot = optimize_layer_placement(calibration[layer], hot_count, swap_rounds)
        owner_map[layer] = owners
        hot_map[layer] = set(hot)
        layers.append({
            "layer": layer,
            "primary_owner": owners,
            "replicated_experts": hot,
            "calibration_skew": placement_skew(calibration[layer], owners, hot_map[layer]),
            "holdout_skew": placement_skew(holdout[layer], owners, hot_map[layer]),
        })
    manifest = {
        "schema": "ggml.cuda.affinity_wave.placement.v1",
        "gguf_sha256": gguf_sha256,
        "architecture": "qwen35moe",
        "layers": N_LAYER,
        "experts_per_layer": N_EXPERT,
        "experts_per_token": N_USED,
        "gpu_count": N_GPU,
        "primary_experts_per_gpu": N_EXPERT // N_GPU,
        "wire_default": "f32",
        "routing_source": routing_source,
        "placement": layers,
    }
    return manifest, owner_map, hot_map


def extract_trace_costs(path: Path, passes: int, pfold_savings_ms: float) -> TraceCosts:
    if passes <= 0:
        raise ValueError("trace pass count must be positive")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        devices = [row[0] for row in connection.execute(
            "SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY deviceId"
        )]
        if devices != list(range(N_GPU)):
            raise ValueError(f"trace devices are {devices}, expected [0, 1, 2, 3]")
        span_ns = connection.execute(
            "SELECT MAX(end)-MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()[0]
        rows = list(connection.execute(
            """
            SELECT k.deviceId, s.value, COUNT(*), SUM(k.end-k.start)/1e9
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON s.id=k.shortName
            GROUP BY k.deviceId, s.value
            """
        ))
    finally:
        connection.close()

    per_device: dict[int, dict[str, tuple[int, float]]] = {device: {} for device in devices}
    for device, name, count, seconds in rows:
        per_device[device][name] = (count, seconds)

    def category(predicate) -> tuple[float, ...]:
        return tuple(
            sum(seconds for name, (_, seconds) in per_device[device].items() if predicate(name)) / passes
            for device in devices
        )

    kernel = category(lambda _name: True)
    nccl = category(lambda name: name.startswith("ncclDevKernel_"))
    expert = category(lambda name: name.startswith("moe_gemm_q8_"))
    get_rows = category(lambda name: name == "k_get_rows_float")
    plan = category(lambda name: name.startswith("moe_plan_"))
    fixed = tuple(
        max(0.0, kernel[d] - nccl[d] - expert[d] - get_rows[d] - plan[d] - pfold_savings_ms / 1000.0)
        for d in devices
    )
    totals: dict[str, tuple[int, float]] = {}
    for device in devices:
        for name, (count, seconds) in per_device[device].items():
            old_count, old_seconds = totals.get(name, (0, 0.0))
            totals[name] = (old_count + count, old_seconds + seconds)
    top = tuple(
        (name, count, seconds / (passes * N_GPU))
        for name, (count, seconds) in sorted(totals.items(), key=lambda item: -item[1][1])[:20]
    )
    return TraceCosts(
        passes=passes,
        span_s=span_ns / 1e9,
        kernel_s_per_gpu=kernel,
        nccl_s_per_gpu=nccl,
        expert_s_per_gpu=expert,
        get_rows_s_per_gpu=get_rows,
        plan_s_per_gpu=plan,
        fixed_lower_s_per_gpu=fixed,
        top_kernels=top,
    )


def _scaled_counts(obs: RouteObservation, chunk_tokens: int) -> tuple[float, ...]:
    scale = chunk_tokens / obs.n_tokens
    return tuple(value * scale for value in obs.counts)


def select_cell_routes(
    holdout: dict[int, list[RouteObservation]], chunk_tokens: int,
) -> tuple[dict[tuple[int, int], tuple[float, ...]], dict[tuple[int, int], tuple[tuple[int, ...], ...] | None]]:
    counts: dict[tuple[int, int], tuple[float, ...]] = {}
    routes: dict[tuple[int, int], tuple[tuple[int, ...], ...] | None] = {}
    for layer in range(N_LAYER):
        exact_chunks: list[tuple[tuple[int, ...], ...]] = []
        for obs in holdout[layer]:
            if obs.routes is None:
                continue
            for start in range(0, len(obs.routes), chunk_tokens):
                part = obs.routes[start:start + chunk_tokens]
                if len(part) == chunk_tokens:
                    exact_chunks.append(part)
        for chunk in range(N_GPU):
            if exact_chunks:
                selected = exact_chunks[chunk % len(exact_chunks)]
                cell_counts = [0.0] * N_EXPERT
                for token_routes in selected:
                    for expert in token_routes:
                        cell_counts[expert] += 1.0
                counts[layer, chunk] = tuple(cell_counts)
                routes[layer, chunk] = selected
            else:
                source = holdout[layer][chunk % len(holdout[layer])]
                counts[layer, chunk] = _scaled_counts(source, chunk_tokens)
                routes[layer, chunk] = None
    return counts, routes


def _remote_pairs(
    token_routes: tuple[tuple[int, ...], ...] | None,
    counts: Sequence[float],
    home: int,
    owners: Sequence[int],
    hot: set[int],
) -> tuple[list[float], str]:
    pairs = [0.0] * N_GPU
    if token_routes is not None:
        for route in token_routes:
            remote = {owners[expert] for expert in route if expert not in hot and owners[expert] != home}
            for owner in remote:
                pairs[owner] += 1.0
        return pairs, "exact"

    # Aggregate histograms lack co-occurrence. remote_routes/top_k is a rigorous
    # lower bound on token-owner pairs, allocated by destination owner.
    for owner in range(N_GPU):
        if owner == home:
            continue
        remote_routes = sum(
            counts[expert]
            for expert in range(N_EXPERT)
            if expert not in hot and owners[expert] == owner
        )
        pairs[owner] = remote_routes / N_USED
    return pairs, "aggregate_lower_bound"


def simulate_wave(
    holdout: dict[int, list[RouteObservation]],
    owners: dict[int, list[int]],
    hot: dict[int, set[int]],
    trace: TraceCosts,
    chunk_tokens: int,
    service_tflops: float,
    peer_gbps: float,
    request_bytes: int,
    response_bytes: int,
    gdn_state_bytes: int,
    kv_key_length: int,
    kv_value_length: int,
) -> dict:
    if service_tflops <= 0 or peer_gbps <= 0:
        raise ValueError("service TFLOP/s and peer GB/s must be positive")
    cell_counts, cell_routes = select_cell_routes(holdout, chunk_tokens)
    fixed_total = max(trace.fixed_lower_s_per_gpu)
    fixed_cell_s = fixed_total / N_LAYER
    route_flops = 3 * 2 * HIDDEN * EXPERT_FF
    stages = []
    transport_modes: set[str] = set()
    total_expert_flops = 0.0
    total_request_pairs = 0.0

    for diagonal in range(N_LAYER + N_GPU - 1):
        compute = [0.0] * N_GPU
        copy_bytes = [0.0] * N_GPU
        cells = []
        for chunk in range(N_GPU):
            layer = diagonal - chunk
            if not 0 <= layer < N_LAYER:
                continue
            counts = cell_counts[layer, chunk]
            compute[chunk] += fixed_cell_s
            for expert, count in enumerate(counts):
                destination = chunk if expert in hot[layer] else owners[layer][expert]
                flops = count * route_flops
                compute[destination] += flops / (service_tflops * 1e12)
                total_expert_flops += flops

            pairs, mode = _remote_pairs(
                cell_routes[layer, chunk], counts, chunk, owners[layer], hot[layer]
            )
            transport_modes.add(mode)
            for destination, pair_count in enumerate(pairs):
                if not pair_count:
                    continue
                wire_bytes = pair_count * (request_bytes + response_bytes)
                copy_bytes[chunk] += wire_bytes
                copy_bytes[destination] += wire_bytes
                total_request_pairs += pair_count

            if chunk < N_GPU - 1:
                if (layer + 1) % 4 == 0:  # full-attention blocks 3,7,...,39
                    state_bytes = (
                        (chunk + 1) * chunk_tokens * (kv_key_length + kv_value_length) * 2
                    )
                else:
                    state_bytes = gdn_state_bytes
                copy_bytes[chunk] += state_bytes
                copy_bytes[chunk + 1] += state_bytes
            cells.append({"chunk": chunk, "layer": layer})

        compute_s = max(compute)
        copy_s = max(copy_bytes) / (peer_gbps * 1e9)
        duration_s = max(compute_s, copy_s)
        stages.append({
            "diagonal": diagonal,
            "cells": cells,
            "compute_s": compute_s,
            "copy_s": copy_s,
            "duration_s": duration_s,
            "gpu_compute_s": compute,
            "gpu_copy_bytes": copy_bytes,
        })

    total_s = sum(stage["duration_s"] for stage in stages)
    ideal_expert_s_per_gpu = total_expert_flops / N_GPU / (service_tflops * 1e12)
    return {
        "seconds": total_s,
        "tokens_per_second": N_GPU * chunk_tokens / total_s,
        "diagonals": len(stages),
        "fixed_trace_lower_bound_s_per_gpu": fixed_total,
        "fixed_wave_fill_lower_bound_s": fixed_cell_s * (N_LAYER + N_GPU - 1),
        "expert_flops": total_expert_flops,
        "ideal_expert_s_per_gpu": ideal_expert_s_per_gpu,
        "remote_owner_pairs": total_request_pairs,
        "transport_accounting": sorted(transport_modes),
        "request_bytes": request_bytes,
        "response_bytes": response_bytes,
        "service_tflops": service_tflops,
        "peer_gbps": peer_gbps,
        "stages": stages,
    }


def mean_layer_skew(manifest: dict, key: str) -> dict[str, float]:
    return {
        metric: sum(layer[key][metric] for layer in manifest["placement"]) / N_LAYER
        for metric in ("mean", "p95", "max")
    }


def run_analysis(args: argparse.Namespace) -> int:
    observations, routing_source = load_routes(args.routes)
    calibration, holdout = split_calibration_holdout(observations)
    gguf_sha256 = args.gguf_sha256 or sha256_file(args.gguf)
    manifest, owners, hot = build_manifest(
        calibration, holdout, gguf_sha256, routing_source, args.hot, args.swap_rounds
    )
    trace = extract_trace_costs(args.nsys, args.trace_passes, args.pfold_savings_ms)
    simulation = simulate_wave(
        holdout, owners, hot, trace,
        chunk_tokens=args.chunk_tokens,
        service_tflops=args.service_tflops,
        peer_gbps=args.peer_gbps,
        request_bytes=HIDDEN * 4,
        response_bytes=HIDDEN * 2,
        gdn_state_bytes=args.gdn_state_bytes,
        kv_key_length=args.kv_key_length,
        kv_value_length=args.kv_value_length,
    )

    physical_peak_expert_s = simulation["expert_flops"] / N_GPU / (args.physical_peak_tflops * 1e12)
    measured_service_bound = (
        simulation["fixed_wave_fill_lower_bound_s"] + simulation["ideal_expert_s_per_gpu"]
    )
    physical_peak_bound = simulation["fixed_wave_fill_lower_bound_s"] + physical_peak_expert_s
    remaining_gate_s = args.gate_seconds - simulation["fixed_wave_fill_lower_bound_s"]
    required_ideal_tflops = (
        simulation["expert_flops"] / N_GPU / remaining_gate_s / 1e12
        if remaining_gate_s > 0 else math.inf
    )
    sensitivity = []
    for service_tflops in sorted({args.service_tflops, 6.0, 6.5, 7.0, args.physical_peak_tflops}):
        candidate = simulate_wave(
            holdout, owners, hot, trace,
            chunk_tokens=args.chunk_tokens,
            service_tflops=service_tflops,
            peer_gbps=args.peer_gbps,
            request_bytes=HIDDEN * 4,
            response_bytes=HIDDEN * 2,
            gdn_state_bytes=args.gdn_state_bytes,
            kv_key_length=args.kv_key_length,
            kv_value_length=args.kv_value_length,
        )
        sensitivity.append({
            "service_tflops": service_tflops,
            "seconds": candidate["seconds"],
            "tokens_per_second": candidate["tokens_per_second"],
        })
    gate_pass = simulation["seconds"] <= args.gate_seconds and routing_source == "token_routes"
    report = {
        "schema": "ggml.cuda.affinity_wave.phase0.v1",
        "decision": "GO" if gate_pass else "NO_GO",
        "backend_implementation_authorized": gate_pass,
        "gate_seconds": args.gate_seconds,
        "routing_source": routing_source,
        "routing_gate_complete": routing_source == "token_routes",
        "model": {
            "path": str(args.gguf),
            "sha256": gguf_sha256,
            "prompt_tokens": N_GPU * args.chunk_tokens,
            "layers": N_LAYER,
            "experts": N_EXPERT,
            "top_k": N_USED,
        },
        "baseline": {
            "tokens_per_second": args.baseline_tps,
            "seconds": N_GPU * args.chunk_tokens / args.baseline_tps,
        },
        "trace": {
            "path": str(args.nsys),
            "passes": trace.passes,
            "span_s": trace.span_s,
            "kernel_s_per_gpu": trace.kernel_s_per_gpu,
            "nccl_s_per_gpu": trace.nccl_s_per_gpu,
            "expert_s_per_gpu": trace.expert_s_per_gpu,
            "get_rows_s_per_gpu": trace.get_rows_s_per_gpu,
            "plan_s_per_gpu": trace.plan_s_per_gpu,
            "pfold_savings_ms": args.pfold_savings_ms,
            "fixed_lower_s_per_gpu": trace.fixed_lower_s_per_gpu,
            "top_kernels": trace.top_kernels,
        },
        "placement": {
            "hot_experts_per_layer": args.hot,
            "swap_rounds": args.swap_rounds,
            "calibration_skew_layer_mean": mean_layer_skew(manifest, "calibration_skew"),
            "holdout_skew_layer_mean": mean_layer_skew(manifest, "holdout_skew"),
        },
        "bounds": {
            "fixed_wave_fill_s": simulation["fixed_wave_fill_lower_bound_s"],
            "expert_at_measured_service_s_per_gpu": simulation["ideal_expert_s_per_gpu"],
            "fixed_plus_expert_measured_service_s": measured_service_bound,
            "expert_at_physical_peak_s_per_gpu": physical_peak_expert_s,
            "fixed_plus_expert_physical_peak_s": physical_peak_bound,
            "physical_peak_tflops": args.physical_peak_tflops,
            "ideal_service_tflops_required_before_imbalance_or_transport": required_ideal_tflops,
        },
        "simulation": simulation,
        "service_sensitivity": sensitivity,
        "limitations": [
            "The A6 trace predates p-fold; its measured p-fold wall saving is subtracted in full.",
            "All get_rows and plan kernels are excluded from fixed work before new packing is added.",
            "P2P copies are allowed to overlap compute completely.",
            "Aggregate histograms provide only a lower bound on remote token-owner pairs."
            if routing_source != "token_routes" else
            "Token-owner pairs are computed exactly from token-level routes.",
        ],
    }

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "placement-hot16.json"
    report_path = args.output / "phase0-report.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"routing source: {routing_source}")
    print(f"GGUF SHA-256: {gguf_sha256}")
    print(f"fixed wave-fill lower bound: {simulation['fixed_wave_fill_lower_bound_s']:.4f} s")
    print(f"ideal expert time at {args.service_tflops:.2f} TFLOP/s/GPU: "
          f"{simulation['ideal_expert_s_per_gpu']:.4f} s")
    print(f"fixed+expert lower bound: {measured_service_bound:.4f} s")
    print(f"discrete-event result: {simulation['seconds']:.4f} s "
          f"({simulation['tokens_per_second']:.1f} tok/s)")
    print(f"Phase-0 decision: {report['decision']}")
    print(f"wrote {manifest_path}")
    print(f"wrote {report_path}")
    return 0 if gate_pass else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True,
                        help="AWTRV001 token trace or legacy aggregate histogram")
    parser.add_argument("--nsys", type=Path, required=True,
                        help="nsys-exported SQLite trace for the A6 pp8192 run")
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--gguf-sha256", help="precomputed lowercase SHA-256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-passes", type=int, default=2,
                        help="warm-up plus measured pp8192 passes present in trace")
    parser.add_argument("--baseline-tps", type=float, default=1697.0)
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    parser.add_argument("--hot", type=int, default=16)
    parser.add_argument("--swap-rounds", type=int, default=2)
    parser.add_argument("--service-tflops", type=float, default=5.5)
    parser.add_argument("--physical-peak-tflops", type=float, default=9.3)
    parser.add_argument("--peer-gbps", type=float, default=10.0)
    parser.add_argument("--gate-seconds", type=float, default=2.05)
    parser.add_argument("--pfold-savings-ms", type=float, default=147.5,
                        help="full measured 2.8%% wall saving, conservatively subtracted from fixed work")
    parser.add_argument("--gdn-state-bytes", type=int, default=65536)
    parser.add_argument("--kv-key-length", type=int, default=256)
    parser.add_argument("--kv-value-length", type=int, default=256)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_analysis(args)
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
