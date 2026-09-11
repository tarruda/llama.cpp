#!/usr/bin/env python3
"""Compare native V4.1 packed activation rows with the CPU Python definitions."""

import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
import torch

import cpu_kernel


def inputs():
    generator = torch.Generator().manual_seed(4101)
    random = torch.randn(2048, 64, generator=generator)
    random *= torch.exp2((torch.arange(2048) % 40 - 25).float()).unsqueeze(-1)
    random = random.bfloat16().float()
    ties = torch.tensor([0., -0., .25, .75, 1.25, 1.75, 2.5, 3.5, 5., -.25, -.75, -1.25, -1.75, -2.5, -3.5, 6.])
    cases = [random, ties.repeat(4).reshape(1, 64), torch.zeros(1, 64)]
    for sign in (1., -1.):
        for maximum in (3.5, 6., 6.03125, 7.9375, 12., 12.0625, 0.01171875, 0.010986328125):
            cases.append(torch.full((1, 64), sign * maximum))
    for exponent in (-130, -127, -126, -20, -9, -6, 0, 6, 9):
        cases.append(ties.repeat(4).reshape(1, 64) * 2.0**exponent)
    fp8 = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    midpoints = (fp8[:-1] + fp8[1:]) / 2
    for midpoint in midpoints:
        row = torch.zeros(1, 64)
        row[:, ::32] = 448
        row[0, 1:4] = torch.stack((midpoint, torch.nextafter(midpoint, torch.tensor(-torch.inf)), torch.nextafter(midpoint, torch.tensor(torch.inf))))
        row[0, 33:36] = -row[0, 1:4]
        cases.append(row)
        # Each FP4 group's maximum makes the E4M3 scale land on this midpoint.
        cases.append(torch.full((1, 64), float(midpoint) * 6))
    return torch.cat(cases).contiguous()


def expected_bytes(x, kind):
    if kind == 'mxfp8':
        quantized, scales = cpu_kernel.act_quant(x, 32, 'ue8m0', torch.float8_e8m0fnu)
        raw = torch.cat((scales.view(torch.uint8).reshape(-1, 1), quantized.view(torch.uint8).reshape(-1, 32)), dim=-1)
        decoded = (quantized.float().unflatten(-1, (-1, 32)) * scales.float().unsqueeze(-1)).flatten(-2)
        return raw.numpy().flatten(), decoded.numpy()
    group = 32 if kind == 'mxfp4' else 16
    dtype = torch.float8_e8m0fnu if kind == 'mxfp4' else torch.float8_e4m3fn
    quantized, scales = cpu_kernel.fp4_act_quant(x, group, scale_dtype=dtype)
    packed = quantized.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).reshape(-1, group)
    packed = codes[:, :group//2] | (codes[:, group//2:] << 4)
    if kind == 'mxfp4':
        raw = torch.cat((scales.view(torch.uint8).reshape(-1, 1), packed), dim=-1)
    else:
        raw = torch.cat((scales.view(torch.uint8).reshape(-1, 4), packed.reshape(-1, 32)), dim=-1)
    decoded = (cpu_kernel.unpack_fp4(quantized).unflatten(-1, (-1, group)) * scales.float().unsqueeze(-1)).flatten(-2)
    return raw.numpy().flatten(), decoded.numpy()


class InitParams(ctypes.Structure):
    _fields_ = [('mem_size', ctypes.c_size_t), ('mem_buffer', ctypes.c_void_p), ('no_alloc', ctypes.c_bool)]


def graph_values(native, library, backend_name, x, kind):
    backend_library = ctypes.CDLL(str(library.resolve().with_name(library.name.replace('ggml-base', 'ggml-' + backend_name))))
    pointer, size, integer = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int

    def bind(owner, name, result, arguments):
        function = getattr(owner, name)
        function.restype, function.argtypes = result, arguments
        return function

    init = bind(backend_library, f'ggml_backend_{backend_name}_init', pointer, [])
    context = bind(native, 'ggml_init', pointer, [InitParams])
    tensor = bind(native, 'ggml_new_tensor_2d', pointer, [pointer, integer, ctypes.c_int64, ctypes.c_int64])
    quantize = bind(native, 'ggml_dsv41_act_quant', pointer, [pointer, pointer, integer])
    graph = bind(native, 'ggml_new_graph_custom', pointer, [pointer, size, ctypes.c_bool])
    expand = bind(native, 'ggml_build_forward_expand', None, [pointer, pointer])
    allocate = bind(native, 'ggml_backend_alloc_ctx_tensors', pointer, [pointer, pointer])
    set_tensor = bind(native, 'ggml_backend_tensor_set', None, [pointer, pointer, size, size])
    get_tensor = bind(native, 'ggml_backend_tensor_get', None, [pointer, pointer, size, size])
    compute = bind(native, 'ggml_backend_graph_compute', integer, [pointer, pointer])
    free_buffer = bind(native, 'ggml_backend_buffer_free', None, [pointer])
    free_context = bind(native, 'ggml_free', None, [pointer])
    free_backend = bind(native, 'ggml_backend_free', None, [pointer])

    backend = init()
    assert backend, f'{backend_name} backend is unavailable'
    ctx, buffer = None, None
    try:
        ctx = context(InitParams(1024*1024, None, True))
        assert ctx
        source = tensor(ctx, 0, x.shape[1], x.shape[0])  # GGML_TYPE_F32
        result = quantize(ctx, source, ('bf16', 'mxfp8', 'mxfp4', 'nvfp4').index(kind))
        forward = graph(ctx, 32, False)
        expand(forward, result)
        buffer = allocate(ctx, backend)
        assert buffer
        set_tensor(source, x.ctypes.data, 0, x.nbytes)
        assert compute(backend, forward) == 0
        actual = np.empty_like(x)
        get_tensor(result, actual.ctypes.data, 0, actual.nbytes)
        return actual
    finally:
        if buffer:
            free_buffer(buffer)
        if ctx:
            free_context(ctx)
        free_backend(backend)


def run(library, backend):
    native = ctypes.CDLL(str(library.resolve()))
    values = inputs()
    x = values.numpy()
    reports = []
    for kind in ('mxfp4', 'nvfp4', 'mxfp8'):
        expected, decoded = expected_bytes(values, kind)
        actual = np.empty_like(expected)
        quantize = getattr(native, f'quantize_row_{kind}_act_ref')
        quantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        quantize.restype = None
        quantize(x.ctypes.data, actual.ctypes.data, x.size)
        mismatch = np.flatnonzero(expected != actual)
        assert len(mismatch) == 0, (kind, len(mismatch), mismatch[:16], expected[mismatch[:16]], actual[mismatch[:16]])
        result = np.empty_like(x)
        dequantize = getattr(native, 'dequantize_row_mxfp8_act' if kind == 'mxfp8' else f'dequantize_row_{kind}')
        dequantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        dequantize.restype = None
        dequantize(actual.ctypes.data, result.ctypes.data, x.size)
        np.testing.assert_array_equal(result, decoded)
        reports.append(dict(format=kind, rows=x.shape[0], values=x.size, packed_bytes=actual.size, exact_packed_bytes=True, exact_decoded_values=True))
        if backend:
            np.testing.assert_array_equal(graph_values(native, library, backend, x, kind), decoded)
            reports[-1]['exact_graph_values'] = backend
    if backend:
        np.testing.assert_array_equal(graph_values(native, library, backend, x, 'bf16'), values.bfloat16().float().numpy())
        reports.append(dict(format='bf16', rows=x.shape[0], values=x.size, exact_graph_values=backend))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True, help='Built ggml-base shared library')
    parser.add_argument('--backend', choices=('cpu', 'metal'), help='Also compare the GGML graph operation with the Python values')
    args = parser.parse_args()
    run(args.library, args.backend)
