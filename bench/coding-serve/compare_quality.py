"""Paired PPL and full-distribution KLD on a long conditioned suffix."""
import argparse
import json
import math
import struct

import numpy as np


def header(source):
    if source.read(8) != b'AWQLOG01':
        raise ValueError('invalid quality header')
    total, rows, vocab = struct.unpack('=III', source.read(12))
    if not 0 < rows <= total or not 0 < vocab < 1000000:
        raise ValueError('invalid quality dimensions')
    return total, rows, vocab


def row(source, vocab):
    target, = struct.unpack('=I', source.read(4))
    data = source.read(4*vocab)
    if len(data) != 4*vocab or target >= vocab:
        raise ValueError('invalid quality row')
    values = np.frombuffer(data, dtype=np.float32).astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError('non-finite quality logits')
    return target, data, values


def compare(control, candidate):
    with open(control, 'rb') as left, open(candidate, 'rb') as right:
        dimensions = header(left)
        if header(right) != dimensions:
            raise ValueError('quality dimensions differ')
        total, rows, vocab = dimensions
        losses = [[], []]
        kld = []
        exact_rows = 0
        same_top = 0
        max_delta = 0.0
        for _ in range(rows):
            target, raw_a, a = row(left, vocab)
            target_b, raw_b, b = row(right, vocab)
            if target != target_b:
                raise ValueError('teacher-forcing targets differ')
            exact_rows += raw_a == raw_b
            same_top += np.argmax(a) == np.argmax(b)
            max_delta = max(max_delta, float(np.max(np.abs(a-b))))
            log_probs = []
            for index, values in enumerate([a, b]):
                shifted = values - np.max(values)
                normalized = shifted - np.log(np.sum(np.exp(shifted)))
                losses[index].append(-float(normalized[target]))
                log_probs.append(normalized)
            kld.append(float(np.sum(np.exp(log_probs[0])*(log_probs[0]-log_probs[1]))))
        if left.read(1) or right.read(1):
            raise ValueError('unexpected trailing quality records')
    base_ppl, candidate_ppl = [math.exp(float(np.mean(x))) for x in losses]
    delta = candidate_ppl - base_ppl
    return dict(conditioned_tokens=total, evaluated_suffix_tokens=rows,
                base_ppl=base_ppl, candidate_ppl=candidate_ppl, paired_ppl_delta=delta,
                mean_kld_base_to_candidate=float(np.mean(kld)),
                max_abs_logit_delta=max_delta, identical_rows=exact_rows,
                same_top_token_fraction=float(same_top/rows),
                byte_identical=exact_rows == rows, ppl_gate_pass=abs(delta) <= 0.003)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('control')
    parser.add_argument('candidate')
    args = parser.parse_args()
    result = compare(args.control, args.candidate)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['ppl_gate_pass'] else 1)
