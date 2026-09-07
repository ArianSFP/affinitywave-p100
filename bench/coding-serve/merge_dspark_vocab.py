#!/usr/bin/env python3
"""Make a self-contained DSpark draft by copying the target vocab tensors.

The Qwen DSpark export intentionally omits token_embd/output and normally
shares them with the target context.  A tensor-parallel target cannot expose
those buffers to a separate one-GPU draft context, so this creates a derived
GGUF containing byte-for-byte copies of the target's existing vocab tensors.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'gguf-py'))

from gguf import GGUFReader, GGUFWriter  # noqa: E402


def copy_metadata(writer: GGUFWriter, reader: GGUFReader) -> None:
    for key, field in reader.fields.items():
        # These are generated from the writer's actual contents.
        if key.startswith('GGUF.') or key == 'general.architecture':
            continue
        if not field.types:
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type.name == 'ARRAY' else None
        writer.add_key_value(key, field.contents(), value_type, sub_type=sub_type)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f'refusing to overwrite existing output: {args.output}')

    draft = GGUFReader(args.draft)
    target = GGUFReader(args.target)
    target_tensors = {tensor.name: tensor for tensor in target.tensors}
    for name in ('token_embd.weight', 'output.weight'):
        if name not in target_tensors:
            raise SystemExit(f'target is missing required tensor: {name}')

    writer = GGUFWriter(args.output, 'dflash', use_temp_file=True)
    copy_metadata(writer, draft)

    for tensor in draft.tensors:
        writer.add_tensor(
            tensor.name,
            tensor.data,
            raw_shape=tensor.data.shape,
            raw_dtype=tensor.tensor_type,
        )

    for name in ('token_embd.weight', 'output.weight'):
        tensor = target_tensors[name]
        writer.add_tensor(
            name,
            tensor.data,
            raw_shape=tensor.data.shape,
            raw_dtype=tensor.tensor_type,
        )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f'wrote {args.output} ({args.output.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
