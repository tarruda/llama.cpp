#!/usr/bin/env python3
"""Locate the first observed difference between V4.1 reference and native captures."""

import argparse
import json
from pathlib import Path
import re

import numpy as np


MODULE_STAGES = {
    'attn_norm': 'attn_norm', 'attn.wq_a': 'q_a', 'attn.q_norm': 'q_norm',
    'attn.wq_b': 'q_b', 'attn': 'attention', 'ffn_norm': 'ffn_norm',
    'ffn.gate.1': 'expert_ids', 'ffn.gate.0': 'expert_weights',
    'ffn': 'moe', '0': 'layer', '1': 'ffn_pre_mix', 'engram': 'engram',
}


def native_name(name):
    if name == 'logits':
        return name
    match = re.fullmatch(r'layers\.(\d+)\.(.+)', name)
    if match:
        layer, suffix = match.groups()
        if suffix in MODULE_STAGES:
            return f'dsv41_{MODULE_STAGES[suffix]}-{layer}'
    if name.startswith('dsv41_') and '_mix_input-' not in name:
        return name.replace('dsv41_ffn_post-', 'dsv41_layer-')
    return None


def load_index(folder):
    records, duplicates = {}, 0
    for line in (folder / 'index.jsonl').read_text().splitlines():
        record = json.loads(line)
        key = record['position'], record['name']
        if key in records:
            if records[key] != record:
                raise ValueError(f'ambiguous native capture: {key}')
            duplicates += 1
        records[key] = record
    return records, duplicates


def load_native(folder, record, shape):
    dtype = {'f32': np.float32, 'i32': np.int32}[record['type']]
    value = np.ndarray(shape=record['shape'][::-1], dtype=dtype,
                       buffer=(folder / record['file']).read_bytes(), strides=record['strides'][::-1])
    if tuple(size for size in value.shape if size != 1) != tuple(size for size in shape if size != 1):
        raise ValueError(f"shape mismatch for {record['name']}: native {value.shape}, reference {shape}")
    return value.reshape(shape)


def number(value):
    value = float(value)
    return value if np.isfinite(value) else str(value)


def compare_arrays(actual, expected, atol, rtol, token_axis, start):
    if actual.shape != expected.shape or actual.size == 0:
        raise ValueError(f'invalid comparison shapes: {actual.shape}, {expected.shape}')
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    delta = np.zeros(a.shape, dtype=np.float64)
    np.subtract(a, b, out=delta, where=finite)
    error = np.abs(delta)
    unequal = (actual != expected) | ~finite
    if np.issubdtype(expected.dtype, np.integer):
        outside = unequal
    else:
        outside = ~finite | (error > atol + rtol * np.abs(b))

    def point(mask):
        indices = np.argwhere(mask)
        if not len(indices):
            return None
        index = tuple(indices[0])
        return dict(index=list(map(int, index)), token=start + (int(index[token_axis]) if token_axis is not None else 0),
                    native=number(actual[index]), reference=number(expected[index]),
                    abs_error=number(error[index]) if finite[index] else None)

    valid = bool(finite.all())
    return dict(elements=actual.size, nonexact=int(unequal.sum()), outside_tolerance=int(outside.sum()),
                nonfinite=int((~finite).sum()), max_abs=float(error.max()) if valid else None,
                relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-30)) if valid else None,
                first_nonexact=point(unequal), first_outside_tolerance=point(outside))


def token_layout(name, shape, tokens, position):
    if name == 'logits':
        return None, position + tokens - 1
    axis = 0 if name.startswith(('dsv41_expert_ids-', 'dsv41_expert_weights-')) else 1
    if len(shape) <= axis or shape[axis] != tokens or (axis == 1 and shape[0] != 1):
        raise ValueError(f'unsupported token layout for {name}: {shape}; expected {tokens} tokens')
    return axis, position


def first_event(stages, field, predicate=lambda row: True):
    for index, row in enumerate(stages):
        if row['metrics'][field] is not None and predicate(row):
            return dict(stage_index=index, position=row['position'], layer=row['layer'], stage=row['stage'],
                        reference_stage=row['reference_stage'], point=row['metrics'][field])
    return None


def compare(reference, native, atol, rtol, position=None, stage_pattern='.*'):
    records, duplicates = load_index(native)
    pattern = re.compile(stage_pattern)
    stages, missing, skipped, runs = [], [], [], []
    paths = sorted(reference.glob('stages-*.npz'), key=lambda path: int(path.stem.split('-')[1]))
    for path in paths:
        start = int(path.stem.split('-')[1])
        if position is not None and start != position:
            continue
        with np.load(path) as arrays:
            tokens = arrays['tokens']
            if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.size == 0:
                raise ValueError(f'expected one nonempty token sequence in {path}')
            runs.append(dict(position=start, tokens=tokens.size, reference_file=path.name))
            seen = set()
            for ref_name in arrays.files:
                name = native_name(ref_name)
                if name is None:
                    skipped.append(dict(position=start, reference_stage=ref_name))
                    continue
                if name in seen or not pattern.search(name):
                    continue
                seen.add(name)
                if (start, name) not in records:
                    missing.append(dict(position=start, stage=name, reference_stage=ref_name))
                    continue
                expected = arrays[ref_name]
                record = records[start, name]
                actual = load_native(native, record, expected.shape)
                axis, token_start = token_layout(name, expected.shape, tokens.size, start)
                layer_match = re.search(r'-(\d+)$', name)
                row = dict(position=start, layer=int(layer_match[1]) if layer_match else None,
                           stage=name, reference_stage=ref_name, reference_file=path.name,
                           native_file=record['file'], shape=list(expected.shape), token_axis=axis,
                           metrics=compare_arrays(actual, expected, atol, rtol, axis, token_start))
                if name.startswith('dsv41_expert_ids-'):
                    same_set = np.all(np.sort(actual, axis=-1) == np.sort(expected, axis=-1), axis=-1)
                    row['routing'] = dict(changed_token_sets=int((~same_set).sum()),
                                          first_changed_token=int(np.flatnonzero(~same_set)[0]) + start if not same_set.all() else None)
                stages.append(row)
    if not runs or not stages:
        raise ValueError('no matched stages for the selected position/filter')
    report = dict(reference=str(reference), native=str(native), atol=atol, rtol=rtol, stage_pattern=stage_pattern,
                  order='reference archive capture order, then row-major element order within each tensor',
                  runs=runs, stages=stages, missing=missing, unmapped_reference=skipped,
                  duplicate_native_records=duplicates,
                  first_nonexact=first_event(stages, 'first_nonexact'),
                  first_outside_tolerance=first_event(stages, 'first_outside_tolerance'),
                  first_residual_nonexact=first_event(stages, 'first_nonexact', lambda row: row['stage'].startswith(('dsv41_attn_post-', 'dsv41_layer-'))),
                  limitations=[
                      'First observed mismatch in matched forward stages, not proof of the first faulty instruction or its cause.',
                      'Unmapped reference stages and cache/state tensors are not accepted as compared.',
                      'Use the same weights, input tokens, batching, state and reference corrections in both runs; legacy native captures do not prove these inputs.',
                      'Duplicate identical index entries refer to the same stored file; overwritten historical values cannot be recovered.',
                      'This adapter expects full-forward captures with matching token spans; decoder-only CED replay requires a separate layout mapping.',
                  ])
    report['passed'] = not missing and report['first_outside_tolerance'] is None
    report['first_routing_change'] = next((dict(position=row['position'], layer=row['layer'], stage=row['stage'],
                                              token=row['routing']['first_changed_token'])
                                         for row in stages if row.get('routing', {}).get('changed_token_sets')), None)
    return report


def describe(event):
    if event is None:
        return 'none in compared stages'
    p = event['point']
    return (f"position={event['position']} layer={event['layer']} {event['stage']} token={p['token']} "
            f"index={p['index']} native={p['native']} reference={p['reference']} abs_error={p['abs_error']}")


def dump_first(report, output):
    event = report['first_outside_tolerance'] or report['first_nonexact']
    if event is None:
        return
    index = event['stage_index']
    payload, metadata = {}, []
    records, _ = load_index(Path(report['native']))
    for i in range(max(0, index - 1), index + 1):
        row = report['stages'][i]
        with np.load(Path(report['reference']) / row['reference_file']) as arrays:
            expected = arrays[row['reference_stage']]
            actual = load_native(Path(report['native']), records[row['position'], row['stage']], expected.shape)
            payload[f'native_{i}'] = actual
            payload[f'reference_{i}'] = expected
            metadata.append(dict(stage_index=i, **row))
    payload['metadata'] = np.array(json.dumps(metadata))
    np.savez(output, **payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path, help='directory containing stages-NNNNN.npz')
    parser.add_argument('native', type=Path, help='directory containing index.jsonl and tensor bytes')
    parser.add_argument('--output', type=Path, required=True, help='write the full comparison as JSON')
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--position', type=int, help='compare one captured call by its starting token position')
    parser.add_argument('--stage-regex', default='.*', help='filter native stage names')
    parser.add_argument('--dump-first', type=Path, help='save the first failing pair and preceding matched stage as NPZ')
    args = parser.parse_args()
    if not np.isfinite([args.atol, args.rtol]).all() or min(args.atol, args.rtol) < 0:
        parser.error('tolerances must be finite and nonnegative')
    try:
        report = compare(args.reference, args.native, args.atol, args.rtol, args.position, args.stage_regex)
    except (ValueError, KeyError, OSError, re.error) as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('First nonexact:', describe(report['first_nonexact']))
    print('First outside tolerance:', describe(report['first_outside_tolerance']))
    print('First residual nonexact:', describe(report['first_residual_nonexact']))
    print('First routed expert set change:', report['first_routing_change'] or 'none in compared stages')
    first = report['first_outside_tolerance'] or report['first_nonexact']
    if first:
        index = first['stage_index']
        print('Surrounding matched stages (reference capture order):')
        for row in report['stages'][max(0, index - 2):index + 3]:
            m = row['metrics']
            print(f"  {row['position']} {row['stage']}: changed={m['nonexact']}/{m['elements']} outside={m['outside_tolerance']} max_abs={m['max_abs']} rel_l2={m['relative_l2']}")
    print(f"Compared {len(report['stages'])} stages; missing {len(report['missing'])}; unmapped reference {len(report['unmapped_reference'])}.")
    if args.dump_first:
        dump_first(report, args.dump_first)
    print('PASS within tolerance for matched stages' if report['passed'] else 'FAIL: differences or missing stages; see JSON report')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
