"""Compare complete FP32 distributions saved by validate.cpp."""
import argparse
import array
import json
import struct


def records(path):
    with open(path, 'rb') as source:
        while header := source.read(4):
            if len(header) != 4:
                raise ValueError('truncated label length')
            length, = struct.unpack('=I', header)
            if length > 1024:
                raise ValueError('invalid label length')
            label = source.read(length).decode('ascii')
            n_vocab, = struct.unpack('=I', source.read(4))
            if not 0 < n_vocab < 1000000:
                raise ValueError('invalid vocabulary length')
            data = source.read(4*n_vocab)
            if len(data) != 4*n_vocab:
                raise ValueError('truncated logits')
            yield label, data


def compare(left, right):
    left_rows = dict(records(left))
    right_rows = dict(records(right))
    if not left_rows or left_rows.keys() != right_rows.keys():
        raise ValueError('missing or mismatched boundaries')
    mismatches = []
    size = 0
    for name, data in left_rows.items():
        other = right_rows[name]
        if len(data) != len(other):
            raise ValueError('vocabulary mismatch')
        size += len(data)
        if data != other:
            a = array.array('f', data)
            b = array.array('f', other)
            mismatches.append(dict(boundary=name,
                changed_logits=sum(x != y for x, y in zip(a, b)),
                max_abs_delta=max(abs(x-y) for x, y in zip(a, b))))
    return dict(boundaries=len(left_rows), compared_bytes=size,
                byte_identical=not mismatches, mismatches=mismatches)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('control')
    parser.add_argument('candidate')
    args = parser.parse_args()
    result = compare(args.control, args.candidate)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['byte_identical'] else 1)
