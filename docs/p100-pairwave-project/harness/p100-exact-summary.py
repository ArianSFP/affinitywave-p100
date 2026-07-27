#!/usr/bin/env python3
"""Read-only paired timing summary for a P100 exact campaign."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gate",
        action="store_true",
        help="exit nonzero unless exactness and paired performance gates pass",
    )
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("results_dir", type=Path)
    return parser.parse_args()


def read_metadata(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["key"]: row["value"]
            for row in csv.DictReader(handle, delimiter="\t")
        }


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing manifest: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def read_exactness(path: Path) -> tuple[str, list[dict[str, str]]]:
    if not path.is_file():
        return "MISSING", []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if rows and all(row.get("result") == "PASS" for row in rows):
        return "PASS", rows
    return "FAIL", rows


def read_timing(path: Path) -> tuple[int, float]:
    if not path.is_file():
        raise ValueError(f"missing timing log: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    status_matches = re.findall(r"^RUN END .* status=([0-9]+) ", text, re.MULTILINE)
    if not status_matches or status_matches[-1] != "0":
        raise ValueError(f"incomplete or failed timing log: {path}")
    ns_match = re.search(r'"avg_ns"\s*:\s*([0-9]+)', text)
    ts_match = re.search(r'"avg_ts"\s*:\s*([-+0-9.eE]+)', text)
    if ns_match is None or ts_match is None:
        raise ValueError(f"timing metrics not found: {path}")
    return int(ns_match.group(1)), float(ts_match.group(1))


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sample")
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_median_lower(
    values: list[float],
    seed: str,
    probability: float = 0.025,
) -> float:
    count = len(values)
    combinations = count**count
    medians: list[float]
    if combinations <= 100_000:
        medians = [
            statistics.median(values[index] for index in sample)
            for sample in itertools.product(range(count), repeat=count)
        ]
    else:
        generator = random.Random(seed)
        medians = [
            statistics.median(generator.choice(values) for _ in range(count))
            for _ in range(100_000)
        ]
    return percentile(medians, probability)


def main() -> int:
    args = parse_args()
    campaign_dir = args.campaign_dir.resolve()
    results_dir = args.results_dir.resolve()
    metadata = read_metadata(campaign_dir / "metadata.tsv")
    rows = read_manifest(campaign_dir / "manifest.tsv")
    exactness_status, exactness_rows = read_exactness(campaign_dir / "exactness.tsv")

    grouped: dict[tuple[int, int], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        if row["stage"] != "performance":
            continue
        tokens = int(row["tokens"])
        pair = int(row["pair"])
        arm = row["arm"]
        avg_ns, avg_ts = read_timing(results_dir / f"{row['tag']}.out")
        grouped[(tokens, pair)][arm] = {
            "position": int(row["position"]),
            "avg_ns": avg_ns,
            "avg_ts": avg_ts,
        }

    if not grouped:
        raise ValueError("manifest contains no performance rows")

    campaign_name = metadata.get("campaign", campaign_dir.name)
    seed = metadata.get("seed", "unknown")
    print("# P100 Exact Campaign Summary")
    print()
    print(f"- Campaign: `{campaign_name}`")
    print(f"- Randomization seed: `{seed}`")
    print(f"- Exactness: **{exactness_status}** ({len(exactness_rows)} checks)")
    print()
    print(
        "| Prompt | Pair | Order | OFF ms | ON ms | Saved ms | "
        "Wall gain | OFF tok/s | ON tok/s | Throughput gain |"
    )
    print("|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|")

    gains_by_tokens: dict[int, list[dict[str, float]]] = defaultdict(list)
    for (tokens, pair), arms in sorted(grouped.items()):
        if set(arms) != {"off", "on"}:
            raise ValueError(f"prompt {tokens} pair {pair} lacks an OFF/ON arm")
        off = arms["off"]
        on = arms["on"]
        off_ns = int(off["avg_ns"])
        on_ns = int(on["avg_ns"])
        off_ts = float(off["avg_ts"])
        on_ts = float(on["avg_ts"])
        saved_ms = (off_ns - on_ns) / 1_000_000.0
        wall_gain = (off_ns - on_ns) * 100.0 / off_ns
        throughput_gain = (on_ts / off_ts - 1.0) * 100.0
        order = "OFF/ON" if int(off["position"]) < int(on["position"]) else "ON/OFF"
        gains_by_tokens[tokens].append(
            {
                "saved_ms": saved_ms,
                "wall_gain": wall_gain,
                "throughput_gain": throughput_gain,
            }
        )
        print(
            f"| {tokens} | {pair} | {order} | {off_ns / 1e6:.3f} | "
            f"{on_ns / 1e6:.3f} | {saved_ms:+.3f} | {wall_gain:+.3f}% | "
            f"{off_ts:.3f} | {on_ts:.3f} | {throughput_gain:+.3f}% |"
        )

    print()
    print("| Prompt | Pairs | Median saved ms | Median wall gain | "
          "Bootstrap 95% lower | Median throughput gain | Gate |")
    print("|---:|---:|---:|---:|---:|---:|:---:|")
    performance_pass = True
    for tokens, gains in sorted(gains_by_tokens.items()):
        saved = [item["saved_ms"] for item in gains]
        wall = [item["wall_gain"] for item in gains]
        throughput = [item["throughput_gain"] for item in gains]
        median_saved = statistics.median(saved)
        median_wall = statistics.median(wall)
        median_throughput = statistics.median(throughput)
        lower = bootstrap_median_lower(wall, f"{seed}:{tokens}")
        gate = median_wall > 0.0 and lower > 0.0
        performance_pass = performance_pass and gate
        print(
            f"| {tokens} | {len(gains)} | {median_saved:+.3f} | "
            f"{median_wall:+.3f}% | {lower:+.3f}% | "
            f"{median_throughput:+.3f}% | {'PASS' if gate else 'FAIL'} |"
        )

    overall_pass = exactness_status == "PASS" and performance_pass
    print()
    print(f"Overall gate: **{'PASS' if overall_pass else 'FAIL'}**")
    if args.gate and not overall_pass:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
