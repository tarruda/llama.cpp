#!/usr/bin/env python3
import unittest

import torch

import cpu_kernel as kernel


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


if __name__ == '__main__':
    unittest.main()
