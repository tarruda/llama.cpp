#!/usr/bin/env python3
import json
from pathlib import Path
import runpy
import tempfile
import unittest

import numpy as np
import torch

import cpu_kernel as kernel

stage_comparison = runpy.run_path(str(Path(__file__).with_name('compare-stages.py')))


class CPUKernelChecks(unittest.TestCase):
    def test_fp4_codes_and_midpoints(self):
        values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
        torch.testing.assert_close(kernel.fp4_codes(values), torch.arange(8, dtype=torch.uint8))
        torch.testing.assert_close(kernel.fp4_codes(-values), torch.arange(8, 16, dtype=torch.uint8))
        midpoints = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
        expected = torch.tensor([0, 2, 2, 4, 4, 6, 6], dtype=torch.uint8)
        torch.testing.assert_close(kernel.fp4_codes(midpoints), expected)
        torch.testing.assert_close(kernel.fp4_codes(-midpoints), expected | 8)
        lower = torch.nextafter(midpoints, torch.full_like(midpoints, -torch.inf))
        upper = torch.nextafter(midpoints, torch.full_like(midpoints, torch.inf))
        torch.testing.assert_close(kernel.fp4_codes(lower), torch.arange(7, dtype=torch.uint8))
        torch.testing.assert_close(kernel.fp4_codes(upper), torch.arange(1, 8, dtype=torch.uint8))

    def test_adjacent_nibble_packing(self):
        values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.] * 4).reshape(1, 32)
        packed, scales = kernel.fp4_act_quant(values)
        expected = torch.tensor([[0x10, 0x32, 0x54, 0x76] * 4], dtype=torch.uint8)
        torch.testing.assert_close(packed.view(torch.uint8), expected)
        torch.testing.assert_close(scales.float(), torch.ones(1, 1))
        torch.testing.assert_close(kernel.unpack_fp4(packed), values)

    def test_zero_scales_and_activation_scale_rule(self):
        zeros = torch.zeros(2, 64, dtype=torch.bfloat16)
        packed, scales = kernel.fp4_act_quant(zeros, 16, scale_dtype=torch.float8_e4m3fn)
        torch.testing.assert_close(scales.view(torch.uint8), torch.ones(2, 4, dtype=torch.uint8))
        self.assertEqual(torch.count_nonzero(packed.view(torch.uint8)), 0)
        _, scales = kernel.fp4_act_quant(zeros, 32)
        torch.testing.assert_close(scales.view(torch.uint8), torch.ones(2, 2, dtype=torch.uint8))
        values = torch.full((1, 32), 3.5, dtype=torch.bfloat16)
        _, scales = kernel.fp4_act_quant(values)
        torch.testing.assert_close(scales.float(), torch.ones(1, 1))
        torch.testing.assert_close(kernel.fp4_act_quant(values, inplace=True), torch.full_like(values, 4.))

    def test_fp8_scale_and_rounding(self):
        values = torch.zeros(1, 32)
        values[0, :4] = torch.tensor([448., -448., 1.0625, 1.1875])
        quantized, scales = kernel.act_quant(values, 32, 'ue8m0', torch.float8_e8m0fnu)
        torch.testing.assert_close(scales.float(), torch.ones(1, 1))
        torch.testing.assert_close(quantized.float()[0, :4], torch.tensor([448., -448., 1., 1.25]))
        _, scales = kernel.act_quant(torch.zeros_like(values), 32, 'ue8m0', torch.float8_e8m0fnu)
        torch.testing.assert_close(scales.float(), torch.full((1, 1), 2**-22))

    def test_block_scaled_gemm(self):
        activations = torch.ones(2, 64).to(torch.float8_e4m3fn)
        activation_scales = torch.tensor([[1., 2.], [4., 8.]])
        weight_values = torch.ones(32, 64)
        weight_scales = torch.tensor([[8., 16.]])
        expected = torch.full((2, 32), 1280.)
        expected[1] *= 4
        actual = kernel.fp8_gemm(activations, activation_scales, weight_values.to(torch.float8_e4m3fn), weight_scales, block_size=32)
        torch.testing.assert_close(actual.float(), expected)
        codes = torch.full((32, 32), 0x22, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
        actual = kernel.fp4_gemm(activations, activation_scales, codes, weight_scales.expand(32, -1), act_block_size=32)
        torch.testing.assert_close(actual.float(), expected)

    def test_sparse_attention_sink_mask_and_duplicate(self):
        query = torch.zeros(1, 2, 2, 4, dtype=torch.bfloat16)
        values = torch.ones(1, 1, 4, dtype=torch.bfloat16)
        ids = torch.tensor([[[0, 0, -1], [-1, -1, -1]]], dtype=torch.int32)
        sink = torch.tensor([0., float('inf')])
        output = kernel.sparse_attn(query, values, sink, ids, 0.5)
        expected = torch.zeros_like(output)
        expected[0, 0, 0] = 2 / 3
        torch.testing.assert_close(output, expected)

    def test_sinkhorn_constant_input(self):
        pre, post, comb = kernel.hc_split_sinkhorn(torch.zeros(1, 2, 24), torch.ones(3), torch.zeros(24))
        torch.testing.assert_close(pre, torch.full((1, 2, 4), .500001))
        torch.testing.assert_close(post, torch.ones(1, 2, 4))
        torch.testing.assert_close(comb.sum(-1), torch.ones(1, 2, 4), atol=2e-6, rtol=0)
        torch.testing.assert_close(comb.sum(-2), torch.ones(1, 2, 4), atol=2e-6, rtol=0)


class StageComparisonChecks(unittest.TestCase):
    def test_first_difference_and_tolerance(self):
        expected = np.ones((1, 2, 3), dtype=np.float32)
        actual = expected.copy()
        actual[0, 0, 2] = np.nextafter(np.float32(1), np.float32(2))
        actual[0, 1, 1] = 1.015625
        result = stage_comparison['compare_arrays'](actual, expected, 1e-6, 1e-5, 1, 33)
        self.assertEqual(result['nonexact'], 2)
        self.assertEqual(result['outside_tolerance'], 1)
        self.assertEqual(result['first_nonexact']['index'], [0, 0, 2])
        self.assertEqual(result['first_outside_tolerance']['index'], [0, 1, 1])
        self.assertEqual(result['first_outside_tolerance']['token'], 34)
        self.assertEqual(result['max_abs'], .015625)

    def test_ids_are_exact_and_nonfinite_fails(self):
        result = stage_comparison['compare_arrays'](np.array([2]), np.array([1]), 100, 100, 0, 0)
        self.assertEqual(result['outside_tolerance'], 1)
        for value in (np.nan, np.inf, -np.inf):
            result = stage_comparison['compare_arrays'](np.array([value]), np.array([1.]), 1e-6, 1e-5, 0, 0)
            self.assertEqual(result['nonfinite'], 1)
            self.assertEqual(result['outside_tolerance'], 1)
            json.dumps(result, allow_nan=False)

    def test_strides_and_shape_validation(self):
        base = np.arange(14, dtype=np.float32).reshape(2, 7)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'view.bin').write_bytes(base.tobytes()[8:])
            record = dict(name='view', type='f32', file='view.bin', shape=[3, 2, 1, 1], strides=[4, 28, 56, 56])
            actual = stage_comparison['load_native'](folder, record, (1, 2, 3))
            np.testing.assert_array_equal(actual, base[:, 2:5][None])
            with self.assertRaises(ValueError):
                stage_comparison['load_native'](folder, record, (1, 3, 2))

    def test_capture_order_routing_and_missing_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            native, reference = folder / 'native', folder / 'reference'
            native.mkdir()
            reference.mkdir()
            stages = {'layers.0.attn_norm': np.ones((1, 2, 3), dtype=np.float32),
                      'layers.0.ffn.gate.1': np.array([[1, 2], [2, 3]], dtype=np.int32)}
            np.savez(reference / 'stages-00033.npz', tokens=np.array([[4, 5]]), **stages)
            records = []
            for key, value in stages.items():
                name = stage_comparison['native_name'](key)
                data = value.copy()
                if key.endswith('gate.1'):
                    data[:] = [[2, 1], [2, 4]]
                else:
                    data[0, 1, 2] = 1.25
                filename = name + '.bin'
                (native / filename).write_bytes(data.tobytes())
                shape = list(data.shape[::-1])
                strides = list(data.strides[::-1])
                records.append(dict(position=33, name=name, file=filename, type='i32' if data.dtype == np.int32 else 'f32',
                                    shape=shape + [1] * (4 - len(shape)), strides=strides + [data.nbytes] * (4 - len(strides))))
            index = native / 'index.jsonl'
            index.write_text(''.join(json.dumps(row) + '\n' for row in records[::-1]))
            report = stage_comparison['compare'](reference, native, 1e-6, 1e-5)
            self.assertEqual(report['first_nonexact']['stage'], 'dsv41_attn_norm-0')
            self.assertEqual(report['first_nonexact']['point']['token'], 34)
            self.assertEqual(report['first_routing_change']['token'], 34)
            self.assertEqual(report['stages'][1]['routing']['changed_token_sets'], 1)
            index.write_text(json.dumps(records[0]) + '\n')
            report = stage_comparison['compare'](reference, native, 10, 10)
            self.assertIsNone(report['first_outside_tolerance'])
            self.assertFalse(report['passed'])
            self.assertEqual(report['missing'][0]['stage'], 'dsv41_expert_ids-0')
            index.write_text(json.dumps(records[0]) + '\n' + json.dumps(dict(records[0], file='different.bin')) + '\n')
            with self.assertRaises(ValueError):
                stage_comparison['load_index'](native)

    def test_capture_positions_sort_numerically(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            native, reference = folder / 'native', folder / 'reference'
            native.mkdir()
            reference.mkdir()
            records = []
            for position in (100000, 99999):
                np.savez(reference / f'stages-{position:05d}.npz', tokens=np.array([[4]]), logits=np.zeros((1, 3), dtype=np.float32))
                filename = f'{position}-logits.bin'
                (native / filename).write_bytes(np.ones(3, dtype=np.float32).tobytes())
                records.append(dict(position=position, name='logits', file=filename, type='f32', shape=[3, 1, 1, 1], strides=[4, 12, 12, 12]))
            (native / 'index.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
            report = stage_comparison['compare'](reference, native, 0, 0)
            self.assertEqual(report['first_nonexact']['position'], 99999)
            self.assertEqual(report['first_nonexact']['point']['token'], 99999)


if __name__ == '__main__':
    unittest.main()
