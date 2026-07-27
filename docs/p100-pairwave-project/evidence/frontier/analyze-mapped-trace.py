#!/usr/bin/env python3
"""Attribute an AffinityWave Nsight trace to logical operations."""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sqlite3
import subprocess
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Range:
    start: int
    end: int
    name: str
    tid: int | None


@dataclass(frozen=True)
class Activity:
    start: int
    end: int
    device: int
    stream: int
    kind: str
    name: str
    correlation: int | None
    graph_node: int | None


def table_exists(db: sqlite3.Connection, name: str) -> bool:
    return db.execute(
        "select 1 from sqlite_master where type='table' and name=?", (name,)
    ).fetchone() is not None


def columns(db: sqlite3.Connection, name: str) -> set[str]:
    return {row[1] for row in db.execute(f'pragma table_info("{name}")')}


def export_sqlite(source: Path, output: Path) -> Path:
    if source.suffix == ".sqlite":
        return source
    if output.exists() and output.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return output
    subprocess.run(
        [
            "nsys",
            "export",
            "--type",
            "sqlite",
            "--force-overwrite=true",
            "--output",
            str(output),
            str(source),
        ],
        check=True,
    )
    return output


def merge(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return [(start, end) for start, end in result]


def union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in merge(intervals))


def strings(db: sqlite3.Connection) -> dict[int, str]:
    if not table_exists(db, "StringIds"):
        return {}
    return dict(db.execute("select id, value from StringIds"))


def load_ranges(db: sqlite3.Connection, names: dict[int, str]) -> list[Range]:
    if not table_exists(db, "NVTX_EVENTS"):
        return []
    result = []
    for start, end, text, text_id, tid in db.execute(
        "select start, end, text, textId, globalTid "
        "from NVTX_EVENTS where end is not null and end > start"
    ):
        name = text or names.get(text_id)
        if name:
            result.append(
                Range(
                    int(start),
                    int(end),
                    str(name),
                    int(tid) if tid is not None else None,
                )
            )
    return result


def choose_window(
    ranges: list[Range], start: int | None, end: int | None
) -> tuple[int, int, list[Range]]:
    if (start is None) != (end is None):
        raise RuntimeError("--window-start-ns and --window-end-ns must be paired")
    if start is not None and end is not None:
        if end <= start:
            raise RuntimeError("invalid explicit time window")
        return start, end, []
    timed = [item for item in ranges if item.name == "timed-pass"]
    if len(timed) != 1:
        raise RuntimeError(
            f"expected one timed-pass NVTX range, found {len(timed)}; "
            "use an explicit window for an unannotated capture"
        )
    tails = [item for item in ranges if item.name == "output-tail"]
    scoped = [timed[0]]
    if len(tails) == 1:
        scoped.append(tails[0])
    return min(item.start for item in scoped), max(item.end for item in scoped), scoped


def load_runtime(
    db: sqlite3.Connection, names: dict[int, str]
) -> dict[int, dict[str, int | str | None]]:
    result = {}
    if not table_exists(db, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return result
    for start, end, tid, corr, name_id in db.execute(
        "select start, end, globalTid, correlationId, nameId "
        "from CUPTI_ACTIVITY_KIND_RUNTIME where correlationId is not null"
    ):
        result[int(corr)] = {
            "start": int(start),
            "end": int(end),
            "tid": int(tid) if tid is not None else None,
            "name": names.get(int(name_id), str(name_id)),
        }
    return result


def load_activities(
    db: sqlite3.Connection,
    names: dict[int, str],
    start_ns: int,
    end_ns: int,
) -> list[Activity]:
    result: list[Activity] = []

    def add(table: str, kind: str) -> None:
        if not table_exists(db, table):
            return
        table_cols = columns(db, table)
        corr = "correlationId" if "correlationId" in table_cols else "null"
        node = "graphNodeId" if "graphNodeId" in table_cols else "null"
        if kind == "kernel":
            extra = "shortName"
        elif kind == "memcpy":
            extra = "bytes"
        else:
            extra = "null"
        query = (
            f"select start,end,deviceId,streamId,{corr},{node},{extra} "
            f'from "{table}" where end > ? and start < ?'
        )
        for row in db.execute(query, (start_ns, end_ns)):
            start, end, device, stream, correlation, graph_node, value = row
            clipped_start = max(start_ns, int(start))
            clipped_end = min(end_ns, int(end))
            if clipped_end <= clipped_start:
                continue
            if kind == "kernel":
                name = names.get(int(value), str(value))
            elif kind == "memcpy":
                name = f"memcpy/{int(value)}B"
            else:
                name = "memset"
            result.append(
                Activity(
                    clipped_start,
                    clipped_end,
                    int(device),
                    int(stream),
                    kind,
                    name,
                    int(correlation) if correlation is not None else None,
                    int(graph_node) if graph_node is not None else None,
                )
            )

    add("CUPTI_ACTIVITY_KIND_KERNEL", "kernel")
    add("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy")
    add("CUPTI_ACTIVITY_KIND_MEMSET", "memset")
    return result


def family(activity: Activity) -> str:
    name = activity.name.lower()
    if activity.kind != "kernel":
        return activity.kind
    if "maxwell_sgemm_fp16_32x128_tn" in name:
        return "target-fp16-sgemm"
    if "flash_attn" in name or "flashattention" in name:
        return "flash-attention"
    if "aw_q8_service" in name:
        return "exact-q8-projection"
    if "aw_live_owner_reduce" in name or "aw_owner_reduce" in name:
        return "owner-reduction"
    if "aw_live_sum_owners" in name:
        return "owner-sum"
    if "gated_delta_net" in name:
        return "gdn"
    if "dequant" in name:
        return "dequant"
    if "convert" in name or "cpy_f32_f16" in name or "cpy_f16_f32" in name:
        return "convert"
    if name.startswith("aw_"):
        return "other-affinitywave"
    return "other-graph"


def parse_label(label: str) -> dict[str, str]:
    fields = {}
    for item in label.split("|")[1:]:
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value
    return fields


def layer_from(fields: dict[str, str]) -> int | None:
    for key in ("weight", "dst"):
        match = re.search(r"(?:blk|layer)[.-](\d+)", fields.get(key, ""))
        if match:
            return int(match.group(1))
    return None


def operation_family(fields: dict[str, str]) -> str:
    text = " ".join((fields.get("dst", ""), fields.get("weight", ""))).lower()
    if "router" in text or "ffn_gate_inp" in text:
        return "router"
    if "attn_qkv" in text:
        return "recurrent-qkv"
    if "attn_z" in text:
        return "recurrent-z"
    if "attn_ba" in text or "beta" in text or "alpha" in text:
        return "recurrent-beta-alpha"
    if "attn_out" in text:
        return "recurrent-output"
    if "attn" in text:
        return "attention"
    if "shared" in text:
        return "shared-expert"
    return "other"


def graph_index(
    db: sqlite3.Connection,
) -> tuple[dict[int, list[int]], dict[int, list[tuple[int, int | None]]]]:
    children: dict[int, list[int]] = defaultdict(list)
    by_tid: dict[int, list[tuple[int, int | None]]] = defaultdict(list)
    if not table_exists(db, "CUDA_GRAPH_NODE_EVENTS"):
        return children, by_tid
    for start, tid, node, original in db.execute(
        "select start, globalTid, graphNodeId, originalGraphNodeId "
        "from CUDA_GRAPH_NODE_EVENTS"
    ):
        node = int(node)
        original_int = int(original) if original is not None else None
        if original_int is not None:
            children[original_int].append(node)
        if tid is not None:
            by_tid[int(tid)].append((int(start), node))
    for items in by_tid.values():
        items.sort()
    return children, by_tid


def descendants(roots: Iterable[int], children: dict[int, list[int]]) -> set[int]:
    seen = set(int(root) for root in roots)
    todo = deque(seen)
    while todo:
        node = todo.popleft()
        for child in children.get(node, []):
            if child not in seen:
                seen.add(child)
                todo.append(child)
    return seen


def range_nodes(
    item: Range, by_tid: dict[int, list[tuple[int, int | None]]]
) -> list[int]:
    if item.tid is None:
        return []
    events = by_tid.get(item.tid, [])
    times = [event[0] for event in events]
    begin = bisect.bisect_left(times, item.start)
    end = bisect.bisect_right(times, item.end)
    return [node for _, node in events[begin:end] if node is not None]


def logical_operations(
    ranges: list[Range],
    activities: list[Activity],
    children: dict[int, list[int]],
    by_tid: dict[int, list[tuple[int, int | None]]],
) -> tuple[list[dict], set[int], int]:
    by_node: dict[int, list[tuple[int, Activity]]] = defaultdict(list)
    for index, activity in enumerate(activities):
        if activity.graph_node is not None:
            by_node[activity.graph_node].append((index, activity))
    operations = []
    mapped_activity_indices: set[int] = set()
    assignment_counts: dict[int, int] = defaultdict(int)
    for item in ranges:
        if not item.name.startswith("aw-gemm-map|"):
            continue
        fields = parse_label(item.name)
        roots = range_nodes(item, by_tid)
        nodes = descendants(roots, children)
        mapped: list[tuple[int, Activity]] = []
        for node in nodes:
            mapped.extend(by_node.get(node, []))
        unique = {index: activity for index, activity in mapped}
        mapped_activity_indices.update(unique)
        for index in unique:
            assignment_counts[index] += 1
        mapped_activities = list(unique.values())
        kernel_groups: dict[str, dict[str, float | int]] = {}
        for group in sorted({family(activity) for activity in mapped_activities}):
            selected = [activity for activity in mapped_activities if family(activity) == group]
            kernel_groups[group] = {
                "count": len(selected),
                "sum_ms": sum(a.end - a.start for a in selected) / 1e6,
                "union_ms": union_ns((a.start, a.end) for a in selected) / 1e6,
            }
        m = int(fields.get("M", "0"))
        n = int(fields.get("N", "0"))
        k = int(fields.get("K", "0"))
        flops = 2*m*n*k
        sgemm_ms = float(kernel_groups.get("target-fp16-sgemm", {}).get("sum_ms", 0.0))
        q8_bytes = m*k*34/32 if fields.get("A") == "q8_0" else m*k*2
        estimated_bytes = q8_bytes + 4*m*k + 6*k*n + 4*m*n
        operations.append(
            {
                "label": item.name,
                "device": int(fields.get("dev", "-1")),
                "layer": layer_from(fields),
                "block_type": (
                    "full-attention"
                    if layer_from(fields) is not None and layer_from(fields) % 4 == 3
                    else "recurrent"
                    if layer_from(fields) is not None
                    else "unknown"
                ),
                "operation_family": operation_family(fields),
                "fields": fields,
                "root_graph_nodes": roots,
                "descendant_graph_node_count": len(nodes),
                "mapped_activity_count": len(mapped_activities),
                "kernel_groups": kernel_groups,
                "flops": flops,
                "estimated_pipeline_bytes": estimated_bytes,
                "sgemm_tflops": flops/(sgemm_ms*1e9) if sgemm_ms else None,
            }
        )
    collisions = sum(count > 1 for count in assignment_counts.values())
    return operations, mapped_activity_indices, collisions


def layer_attribution(
    ranges: list[Range],
    activities: list[Activity],
    runtime: dict[int, dict[str, int | str | None]],
    children: dict[int, list[int]],
    by_tid: dict[int, list[tuple[int, int | None]]],
) -> list[dict]:
    by_node: dict[int, list[tuple[int, Activity]]] = defaultdict(list)
    by_correlation: dict[int, list[tuple[int, Activity]]] = defaultdict(list)
    for index, activity in enumerate(activities):
        if activity.graph_node is not None:
            by_node[activity.graph_node].append((index, activity))
        if activity.correlation is not None:
            by_correlation[activity.correlation].append((index, activity))

    attributed: dict[tuple[int, int, str], dict[int, Activity]] = defaultdict(dict)
    graph_pattern = re.compile(
        r"diag=\d+/lane=(\d+)/layer=(\d+)/stage=([^/]+)$"
    )
    service_pattern = re.compile(
        r"service/(input|compute|output|reduce|nccl-sum)/.*"
    )
    for item in ranges:
        graph_match = graph_pattern.match(item.name)
        service_match = service_pattern.match(item.name)
        if graph_match:
            lane = int(graph_match.group(1))
            layer = int(graph_match.group(2))
            stage = graph_match.group(3)
        elif service_match and (
            "layers=" in item.name or "layer=" in item.name
        ):
            lane_match = re.search(r"(?:owner|home)=(\d+)", item.name)
            lane = int(lane_match.group(1)) if lane_match else -1
            one_layer = re.search(r"(?:^|/)layer=(\d+)", item.name)
            layer_range = re.search(r"layers=(\d+)-(\d+)", item.name)
            if one_layer:
                layer = int(one_layer.group(1))
            elif layer_range and layer_range.group(1) == layer_range.group(2):
                layer = int(layer_range.group(1))
            else:
                continue
            stage = f"service-{service_match.group(1)}"
        else:
            continue

        selected: dict[int, Activity] = {}
        roots = range_nodes(item, by_tid)
        for node in descendants(roots, children):
            for index, activity in by_node.get(node, []):
                selected[index] = activity
        if item.tid is not None:
            for correlation, api in runtime.items():
                if api["tid"] != item.tid:
                    continue
                midpoint = (int(api["start"]) + int(api["end"]))//2
                if item.start <= midpoint <= item.end:
                    for index, activity in by_correlation.get(correlation, []):
                        selected[index] = activity
        attributed[(layer, lane, stage)].update(selected)

    result = []
    for (layer, lane, stage), selected in sorted(attributed.items()):
        values = list(selected.values())
        family_totals = {}
        for name in sorted({family(activity) for activity in values}):
            group = [activity for activity in values if family(activity) == name]
            family_totals[name] = {
                "count": len(group),
                "sum_ms": sum(activity.end - activity.start for activity in group)/1e6,
                "union_ms": union_ns(
                    (activity.start, activity.end) for activity in group
                )/1e6,
            }
        result.append(
            {
                "layer": layer,
                "lane": lane,
                "stage": stage,
                "activity_count": len(values),
                "sum_ms": sum(activity.end - activity.start for activity in values)/1e6,
                "union_ms": union_ns(
                    (activity.start, activity.end) for activity in values
                )/1e6,
                "families": family_totals,
            }
        )
    return result


def summarize(
    source: Path,
    start_ns: int,
    end_ns: int,
    ranges: list[Range],
    activities: list[Activity],
    operations: list[dict],
    mapped_indices: set[int],
    mapping_collisions: int,
    layer_rows: list[dict],
) -> dict:
    pass_ms = (end_ns - start_ns)/1e6
    devices = sorted({activity.device for activity in activities})
    device_rows = {}
    for device in devices:
        selected = [activity for activity in activities if activity.device == device]
        families = {}
        for group in sorted({family(activity) for activity in selected}):
            group_items = [activity for activity in selected if family(activity) == group]
            families[group] = {
                "count": len(group_items),
                "sum_ms": sum(a.end - a.start for a in group_items)/1e6,
                "union_ms": union_ns((a.start, a.end) for a in group_items)/1e6,
            }
        device_rows[str(device)] = {
            "busy_union_ms": union_ns((a.start, a.end) for a in selected)/1e6,
            "idle_in_window_ms": pass_ms -
                union_ns((a.start, a.end) for a in selected)/1e6,
            "families": families,
        }
    critical = max(devices, key=lambda device: device_rows[str(device)]["busy_union_ms"])
    target = [
        (index, activity)
        for index, activity in enumerate(activities)
        if family(activity) == "target-fp16-sgemm"
    ]
    mapped_target = [activity for index, activity in target if index in mapped_indices]
    target_sum = sum(activity.end - activity.start for _, activity in target)/1e6
    mapped_target_sum = sum(activity.end - activity.start for activity in mapped_target)/1e6
    operation_totals: dict[str, dict[str, float | int]] = {}
    operation_keys = {
        f"{op['block_type']}/{op['operation_family']}" for op in operations
    }
    for op_key in sorted(operation_keys):
        selected = [
            op
            for op in operations
            if f"{op['block_type']}/{op['operation_family']}" == op_key
        ]
        operation_totals[op_key] = {
            "count": len(selected),
            "flops": sum(op["flops"] for op in selected),
            "sgemm_ms": sum(
                float(op["kernel_groups"].get("target-fp16-sgemm", {}).get("sum_ms", 0.0))
                for op in selected
            ),
            "dequant_ms": sum(
                float(op["kernel_groups"].get("dequant", {}).get("sum_ms", 0.0))
                for op in selected
            ),
            "convert_ms": sum(
                float(op["kernel_groups"].get("convert", {}).get("sum_ms", 0.0))
                for op in selected
            ),
        }
    target_per_device = {}
    for device in devices:
        device_target = [
            (index, activity)
            for index, activity in target
            if activity.device == device
        ]
        target_per_device[str(device)] = {
            "count": len(device_target),
            "mapped_count": sum(index in mapped_indices for index, _ in device_target),
            "sum_ms": sum(activity.end - activity.start for _, activity in device_target)/1e6,
        }
    duration_attribution = mapped_target_sum/target_sum if target_sum else None
    mapping_passed = (
        len(target) == 600
        and all(value["count"] == 150 for value in target_per_device.values())
        and len(mapped_target) == len(target)
        and duration_attribution is not None
        and duration_attribution >= 0.99
        and mapping_collisions == 0
    )
    return {
        "source": str(source),
        "window_start_ns": start_ns,
        "window_end_ns": end_ns,
        "window_ms": pass_ms,
        "devices": device_rows,
        "critical_device": critical,
        "aggregate_device_busy_ms": sum(
            values["busy_union_ms"] for values in device_rows.values()
        ),
        "perfect_balance_floor_ms": sum(
            values["busy_union_ms"] for values in device_rows.values()
        )/len(device_rows),
        "target_sgemm": {
            "count": len(target),
            "sum_ms": target_sum,
            "mapped_count": len(mapped_target),
            "mapped_sum_ms": mapped_target_sum,
            "duration_attribution": duration_attribution,
            "per_device": target_per_device,
        },
        "mapping_collision_count": mapping_collisions,
        "mapping_acceptance_passed": mapping_passed,
        "gemm_map_range_count": sum(
            item.name.startswith("aw-gemm-map|") for item in ranges
        ),
        "logical_operation_count": len(operations),
        "operation_totals": operation_totals,
        "layer_attribution": layer_rows,
        "operations": operations,
    }


def f3(value: float | int) -> str:
    return f"{float(value):.3f}"


def markdown(result: dict) -> str:
    target = result["target_sgemm"]
    lines = [
        f"# Mapped trace: `{Path(result['source']).name}`",
        "",
        f"Window: {f3(result['window_ms'])} ms. Critical GPU: "
        f"{result['critical_device']}. Perfect-balance device-work floor: "
        f"{f3(result['perfect_balance_floor_ms'])} ms.",
        "",
        "## Device work",
        "",
        "| GPU | Busy union (ms) | Idle in window (ms) |",
        "|---:|---:|---:|",
    ]
    for device, values in result["devices"].items():
        lines.append(
            f"| {device} | {f3(values['busy_union_ms'])} | "
            f"{f3(values['idle_in_window_ms'])} |"
        )
    attribution = target["duration_attribution"]
    lines += [
        "",
        "## Dense mapping acceptance",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Target SGEMM launches | {target['count']} |",
        f"| Mapped target launches | {target['mapped_count']} |",
        f"| Target SGEMM summed time | {f3(target['sum_ms'])} ms |",
        f"| Mapped SGEMM summed time | {f3(target['mapped_sum_ms'])} ms |",
        f"| Duration attribution | "
        f"{f3(100*attribution) if attribution is not None else 'n/a'}% |",
        f"| Logical GEMM ranges | {result['gemm_map_range_count']} |",
        f"| Mapping collisions | {result['mapping_collision_count']} |",
        f"| Acceptance | {'PASS' if result['mapping_acceptance_passed'] else 'FAIL'} |",
        "",
        "| GPU | Target launches | Mapped | Summed time (ms) |",
        "|---:|---:|---:|---:|",
    ]
    for device, values in target["per_device"].items():
        lines.append(
            f"| {device} | {values['count']} | {values['mapped_count']} | "
            f"{f3(values['sum_ms'])} |"
        )
    lines += [
        "",
        "## Logical operation families",
        "",
        "| Family | Count | SGEMM (ms) | Dequant (ms) | Convert (ms) | TFLOP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in sorted(
        result["operation_totals"].items(),
        key=lambda item: item[1]["sgemm_ms"],
        reverse=True,
    ):
        lines.append(
            f"| {name} | {values['count']} | {f3(values['sgemm_ms'])} | "
            f"{f3(values['dequant_ms'])} | {f3(values['convert_ms'])} | "
            f"{f3(values['flops']/1e12)} |"
        )
    lines += [
        "",
        "## Critical-GPU families",
        "",
        "| Family | Count | Union (ms) | Summed time (ms) |",
        "|---|---:|---:|---:|",
    ]
    critical = result["devices"][str(result["critical_device"])]
    for name, values in sorted(
        critical["families"].items(),
        key=lambda item: item[1]["union_ms"],
        reverse=True,
    ):
        lines.append(
            f"| {name} | {values['count']} | {f3(values['union_ms'])} | "
            f"{f3(values['sum_ms'])} |"
        )
    return "\n".join(lines) + "\n"


def self_test() -> None:
    label = (
        "aw-gemm-map|dev=3|dst=linear_attn_qkv_mixed-12"
        "|weight=blk.12.attn_qkv.weight|input=attn_norm-12"
        "|M=8192|N=2032|K=2048|A=q8_0|B=f32|C=f32"
        "|lda=2048|ldb=2048|ldc=8192|split=0|precision=f16-f32"
    )
    ranges = [Range(10, 20, label, 1)]
    activity = Activity(
        100,
        200,
        3,
        7,
        "kernel",
        "maxwell_sgemm_fp16_32x128_tn",
        9,
        200,
    )
    operations, mapped, collisions = logical_operations(
        ranges,
        [activity],
        {100: [200]},
        {1: [(15, 100)]},
    )
    assert mapped == {0}
    assert collisions == 0
    assert operations[0]["layer"] == 12
    assert operations[0]["operation_family"] == "recurrent-qkv"
    assert operations[0]["kernel_groups"]["target-fp16-sgemm"]["count"] == 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--window-start-ns", type=int)
    parser.add_argument("--window-end-ns", type=int)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return 0
    if args.source is None:
        parser.error("source is required unless --self-test is used")
    output_prefix = args.output_prefix or args.source.with_suffix("")
    sqlite_path = export_sqlite(args.source, output_prefix.with_suffix(".sqlite"))
    db = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    name_map = strings(db)
    ranges = load_ranges(db, name_map)
    start_ns, end_ns, _ = choose_window(
        ranges, args.window_start_ns, args.window_end_ns
    )
    activities = load_activities(db, name_map, start_ns, end_ns)
    runtime = load_runtime(db, name_map)
    children, by_tid = graph_index(db)
    operations, mapped, mapping_collisions = logical_operations(
        ranges, activities, children, by_tid
    )
    layer_rows = layer_attribution(
        ranges, activities, runtime, children, by_tid
    )
    result = summarize(
        args.source,
        start_ns,
        end_ns,
        ranges,
        activities,
        operations,
        mapped,
        mapping_collisions,
        layer_rows,
    )
    json_path = output_prefix.with_suffix(".analysis.json")
    md_path = output_prefix.with_suffix(".analysis.md")
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    md_path.write_text(markdown(result))
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
