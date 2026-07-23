#!/usr/bin/env python3

import collections
import sqlite3
import sys


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: analyze-phase2a-trace.py TRACE.sqlite")

    db = sqlite3.connect(sys.argv[1])
    strings = dict(db.execute("select id, value from StringIds"))
    kernels = list(db.execute(
        "select deviceId, streamId, start, end, demangledName "
        "from CUPTI_ACTIVITY_KIND_KERNEL"))
    copies = list(db.execute(
        "select deviceId, streamId, start, end, bytes, copyKind, "
        "srcDeviceId, dstDeviceId from CUPTI_ACTIVITY_KIND_MEMCPY"))
    memsets = list(db.execute(
        "select deviceId, streamId, start, end, bytes "
        "from CUPTI_ACTIVITY_KIND_MEMSET"))

    aw_intervals = [
        (start, end)
        for _, _, start, end, name_id in kernels
        if "aw_live_" in strings[name_id] or "aw_q8_service_" in strings[name_id]
    ]
    aw_window = None
    if aw_intervals:
        aw_window = (min(row[0] for row in aw_intervals), max(row[1] for row in aw_intervals))
        print("affinity_window_ms %.3f" % ((aw_window[1] - aw_window[0])/1e6))

    for device in sorted({row[0] for row in kernels + copies + memsets}):
        activities = []
        stream_intervals = collections.defaultdict(list)
        kernel_sum = collections.defaultdict(int)
        kernel_count = collections.Counter()
        copy_sum = collections.defaultdict(int)
        copy_bytes = collections.defaultdict(int)

        for dev, stream, start, end, name_id in kernels:
            if dev != device:
                continue
            activities.append((start, end))
            stream_intervals[stream].append((start, end))
            name = strings[name_id]
            kernel_sum[name] += end - start
            kernel_count[name] += 1
        for dev, stream, start, end, nbytes, kind, src, dst in copies:
            if dev != device:
                continue
            activities.append((start, end))
            stream_intervals[stream].append((start, end))
            key = (kind, src, dst)
            copy_sum[key] += end - start
            copy_bytes[key] += nbytes
        for dev, stream, start, end, _ in memsets:
            if dev != device:
                continue
            activities.append((start, end))
            stream_intervals[stream].append((start, end))

        merged = merge_intervals(activities)
        span = merged[-1][1] - merged[0][0]
        busy = sum(end - start for start, end in merged)
        gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
        print("device", device)
        print("  span_ms %.3f busy_ms %.3f idle_ms %.3f busy_pct %.1f" % (
            span/1e6, busy/1e6, (span - busy)/1e6, 100.0*busy/span))
        for threshold_us in (20, 100, 1000):
            selected = [gap for gap in gaps if gap >= threshold_us*1000]
            print("  gaps_ge_%dus count %d total_ms %.3f max_ms %.3f" % (
                threshold_us, len(selected), sum(selected)/1e6,
                max(selected, default=0)/1e6))

        if aw_window is not None:
            aw_activities = [
                (max(start, aw_window[0]), min(end, aw_window[1]))
                for start, end in activities
                if start < aw_window[1] and end > aw_window[0]
            ]
            aw_merged = merge_intervals(aw_activities)
            aw_busy = sum(end - start for start, end in aw_merged)
            aw_span = aw_window[1] - aw_window[0]
            aw_gaps = [
                aw_merged[i + 1][0] - aw_merged[i][1]
                for i in range(len(aw_merged) - 1)
            ]
            print("  affinity_busy_ms %.3f idle_ms %.3f busy_pct %.1f" % (
                aw_busy/1e6, (aw_span - aw_busy)/1e6, 100.0*aw_busy/aw_span))
            for threshold_us in (20, 100, 1000):
                selected = [gap for gap in aw_gaps if gap >= threshold_us*1000]
                print("  affinity_gaps_ge_%dus count %d total_ms %.3f max_ms %.3f" % (
                    threshold_us, len(selected), sum(selected)/1e6,
                    max(selected, default=0)/1e6))

        print("  streams")
        stream_rows = []
        for stream, intervals in stream_intervals.items():
            merged_stream = merge_intervals(intervals)
            duration = sum(end - start for start, end in merged_stream)
            stream_rows.append((duration, stream, len(intervals)))
        for duration, stream, count in sorted(stream_rows, reverse=True)[:10]:
            print("    id %d busy_ms %.3f activities %d" % (
                stream, duration/1e6, count))

        print("  kernels")
        for name, duration in sorted(kernel_sum.items(), key=lambda item: item[1], reverse=True)[:24]:
            print("    %9.3f ms %6d %s" % (
                duration/1e6, kernel_count[name], name))

        print("  copies")
        for key, duration in sorted(copy_sum.items(), key=lambda item: item[1], reverse=True):
            kind, src, dst = key
            print("    %9.3f ms %9.3f MiB kind=%d src=%s dst=%s" % (
                duration/1e6, copy_bytes[key]/1048576.0, kind, src, dst))


if __name__ == "__main__":
    main()
