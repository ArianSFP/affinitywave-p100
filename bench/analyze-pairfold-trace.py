#!/usr/bin/env python3

import argparse
import collections
import json
import re
import sqlite3


TASK = re.compile(
    r"pairfold/gen=\d+/layer=\d+/panel=\d+/pair=(\d+)$")


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if not result or start > result[-1][1]:
            result.append([start, end])
        elif end > result[-1][1]:
            result[-1][1] = end
    return [(start, end) for start, end in result]


def duration(intervals):
    return sum(end - start for start, end in intervals)


def intersection_duration(left, right):
    total = 0
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) -
                     max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def active_histogram(interval_sets, begin, end):
    events = [(begin, 0), (end, 0)]
    for intervals in interval_sets:
        for start, stop in intervals:
            start = max(start, begin)
            stop = min(stop, end)
            if start < stop:
                events.append((start, 1))
                events.append((stop, -1))
    events.sort(key=lambda value: (value[0], value[1]))
    result = collections.Counter()
    active = 0
    previous = begin
    for timestamp, delta in events:
        if timestamp > previous:
            result[active] += timestamp - previous
        active += delta
        previous = timestamp
    return result


def milliseconds(value):
    return round(value/1e6, 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    args = parser.parse_args()

    connection = sqlite3.connect(args.sqlite)
    strings = dict(connection.execute(
        "select id, value from StringIds"))
    tasks = []
    pair_tasks = {0: [], 1: []}
    for start, end, text, text_id in connection.execute(
            "select start, end, text, textId from NVTX_EVENTS"):
        text = text if text is not None else strings.get(text_id)
        match = TASK.fullmatch(text or "")
        if match is None or end is None:
            continue
        tasks.append((start, end))
        pair_tasks[int(match.group(1))].append((start, end))
    if len(tasks) != 80:
        raise SystemExit(f"expected 80 PairFold tasks, found {len(tasks)}")

    begin = min(start for start, _ in tasks)
    host_end = max(end for _, end in tasks)
    marker_ids = [
        identifier for identifier, value in strings.items()
        if "ggml_cuda_aw_pairfold_terminal_marker" in value]
    marker_ends = []
    if marker_ids:
        placeholders = ",".join("?" for _ in marker_ids)
        marker_ends = [
            end for end, in connection.execute(
                "select end from CUPTI_ACTIVITY_KIND_KERNEL "
                f"where demangledName in ({placeholders}) and start >= ?",
                (*marker_ids, begin))]
    if marker_ends and len(marker_ends) != 4:
        raise SystemExit(
            f"expected four PairFold terminal markers, found {len(marker_ends)}")
    kernel_limit = max(marker_ends) if marker_ends else host_end + 10_000_000
    raw_kernels = {device: [] for device in range(4)}
    stream_kernels = {device: [] for device in range(4)}
    kernel_names = collections.Counter()
    for start, end, device, stream, name_id in connection.execute(
            "select start, end, deviceId, streamId, demangledName "
            "from CUPTI_ACTIVITY_KIND_KERNEL "
            "where end > ? and start < ?",
            (begin, kernel_limit)):
        if device not in raw_kernels:
            continue
        raw_kernels[device].append((max(start, begin), end))
        stream_kernels[device].append((start, end, stream))
        kernel_names[strings.get(name_id, "")] += 1
    trace_end = (
        max(marker_ends) if marker_ends else
        max(
            host_end,
            max(end for kernels in raw_kernels.values()
                for _, end in kernels)))
    kernel_union = {
        device: merge((start, min(end, trace_end))
                      for start, end in kernels
                      if start < trace_end)
        for device, kernels in raw_kernels.items()
    }

    gpu = {}
    for device in range(4):
        busy = duration(kernel_union[device])
        overlap = (
            duration(raw_kernels[device]) - busy)
        cross_stream_overlap = 0
        kernels = sorted(stream_kernels[device])
        for index, (start, end, stream) in enumerate(kernels):
            for other_start, other_end, other_stream in kernels[index + 1:]:
                if other_start >= end:
                    break
                if stream != other_stream:
                    cross_stream_overlap += (
                        min(end, other_end) -
                        max(start, other_start))
        gpu[str(device)] = {
            "busy_ms": milliseconds(busy),
            "idle_ms": milliseconds(trace_end - begin - busy),
            "busy_percent": round(
                100.0*busy/(trace_end - begin), 3),
            "same_gpu_sm_overlap_ms": milliseconds(overlap),
            "cross_stream_sm_overlap_ms":
                milliseconds(cross_stream_overlap),
        }

    gpu_count = active_histogram(
        [kernel_union[device] for device in range(4)],
        begin, trace_end)
    pair_union = {
        0: merge(kernel_union[0] + kernel_union[1]),
        1: merge(kernel_union[2] + kernel_union[3]),
    }
    pair_count = active_histogram(
        [pair_union[0], pair_union[1]], begin, trace_end)
    host_pair_count = active_histogram(
        [merge(pair_tasks[0]), merge(pair_tasks[1])],
        begin, host_end)

    raw_copies = {device: [] for device in range(4)}
    copy_bytes = collections.Counter()
    for start, end, device, byte_count in connection.execute(
            "select start, end, deviceId, bytes "
            "from CUPTI_ACTIVITY_KIND_MEMCPY "
            "where end > ? and start < ?",
            (begin, trace_end)):
        if device not in raw_copies:
            continue
        raw_copies[device].append((
            max(start, begin), min(end, trace_end)))
        copy_bytes[device] += byte_count
    copy_union = {
        device: merge(raw_copies[device])
        for device in range(4)
    }
    all_copies = merge([
        interval for intervals in copy_union.values()
        for interval in intervals])
    all_kernels = merge([
        interval for intervals in kernel_union.values()
        for interval in intervals])
    copy_time = duration(all_copies)
    exposed_copy = copy_time - intersection_duration(
        all_copies, all_kernels)

    result = {
        "task_count": len(tasks),
        "window_ms": milliseconds(trace_end - begin),
        "gpu": gpu,
        "time_by_active_gpu_count_ms": {
            str(count): milliseconds(value)
            for count, value in sorted(gpu_count.items())
        },
        "pair": {
            str(pair): {
                "busy_ms": milliseconds(duration(pair_union[pair]))
            }
            for pair in range(2)
        },
        "time_by_active_pair_count_ms": {
            str(count): milliseconds(value)
            for count, value in sorted(pair_count.items())
        },
        "host_task_time_by_active_pair_count_ms": {
            str(count): milliseconds(value)
            for count, value in sorted(host_pair_count.items())
        },
        "copy": {
            "bytes": sum(copy_bytes.values()),
            "union_ms": milliseconds(copy_time),
            "outside_all_sm_work_ms": milliseconds(exposed_copy),
            "exposed_percent_of_copy_union": round(
                100.0*exposed_copy/copy_time, 3)
                if copy_time != 0 else 0.0,
        },
        "kernels": {
            "peer_named_count": sum(
                count for name, count in kernel_names.items()
                if "peer" in name.lower()),
            "pairfold_local_sum_count": sum(
                count for name, count in kernel_names.items()
                if "aw_pair_sum_groups_local_panel_p100" in name),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
