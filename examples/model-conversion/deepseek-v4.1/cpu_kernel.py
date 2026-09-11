"""CPU definitions of the numerical kernels used by the official V4.1 model.

These follow inference/kernel.py. They do not emulate CUDA reduction order.
"""

import torch


def round_scale(x):
    bits = x.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7fffff) != 0).int()
    return ((exponent + 127) << 23).view(torch.float32)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    assert x.shape[-1] % block_size == 0
    groups = x.float().unflatten(-1, (-1, block_size))
    amax = groups.abs().amax(-1).clamp_min(1e-4)
    scales = amax * (1.0 / 448.0)
    if scale_fmt is not None:
        scales = round_scale(scales)
    values = (groups / scales.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((values.float() * scales.unsqueeze(-1)).flatten(-2))
        return x
    return values.flatten(-2), scales.to(scale_dtype)


def fp4_codes(x):
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
    magnitude = x.abs().contiguous()
    codes = torch.bucketize(magnitude, midpoints)
    tie = magnitude == midpoints[codes.clamp_max(6)]
    codes += (tie & (codes < 7) & ((codes & 1) != 0)).int()
    return (codes | (torch.signbit(x).int() << 3)).to(torch.uint8)


def fp4_values(codes):
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    magnitude = values[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -magnitude, magnitude)


def unpack_fp4(x):
    packed = x.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    return fp4_values(codes)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert x.shape[-1] % block_size == 0
    groups = x.float().unflatten(-1, (-1, block_size))
    amax = groups.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        scales = (amax.clamp_min(6 * 2**-9) / 6).clamp_max(448).to(scale_dtype).float()
    else:
        assert scale_dtype == torch.float8_e8m0fnu
        scales = round_scale(amax.clamp_min(6 * 2**-126) * (1.0 / 6.0))
    codes = fp4_codes((groups / scales.unsqueeze(-1)).clamp(-6, 6))
    if inplace:
        x.copy_((fp4_values(codes) * scales.unsqueeze(-1)).flatten(-2))
        return x
    codes = codes.flatten(-2)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.view(torch.float4_e2m1fn_x2), scales.to(scale_dtype)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    del scale_dtype
    assert block_size in (32, 128) and a.shape[-1] % block_size == 0
    a, b, a_s, b_s = a.float(), b.float(), a_s.float(), b_s.float()
    output = torch.zeros(*a.shape[:-1], b.shape[0], dtype=torch.float32)
    row_blocks = torch.arange(b.shape[0]) // block_size
    for k in range(a.shape[-1] // block_size):
        begin, end = k * block_size, (k + 1) * block_size
        partial = a[..., begin:end] @ b[:, begin:end].T
        output += partial * a_s[..., k, None] * b_s[row_blocks, k]
    return output.to(torch.bfloat16)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    del scale_dtype
    assert act_block_size in (32, 128) and a.shape[-1] % act_block_size == 0
    a, b, a_s, b_s = a.float(), unpack_fp4(b), a_s.float(), b_s.float()
    output = torch.zeros(*a.shape[:-1], b.shape[0], dtype=torch.float32)
    for k in range(a.shape[-1] // 32):
        begin, end = k * 32, (k + 1) * 32
        partial = a[..., begin:end] @ b[:, begin:end].T
        output += partial * a_s[..., k // (act_block_size // 32), None] * b_s[:, k]
    return output.to(torch.bfloat16)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2*hc] * hc_scale[1] + hc_base[hc:2*hc])
    comb = (mixes[..., 2*hc:] * hc_scale[2] + hc_base[2*hc:]).unflatten(-1, (hc, hc))
    comb = (comb - comb.amax(-1, keepdim=True)).exp()
    comb = comb / comb.sum(-1, keepdim=True) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    assert q.dtype == kv.dtype == torch.bfloat16
    assert ((topk_idxs >= -1) & (topk_idxs < kv.shape[1])).all()
    batch, length, heads, width = q.shape
    output = torch.zeros(batch, length, heads, width, dtype=torch.float32)
    maximum = torch.full((batch, length, heads), -1e30, dtype=torch.float32)
    denominator = torch.zeros_like(maximum)
    batch_ids = torch.arange(batch)[:, None, None]
    for begin in range(0, topk_idxs.shape[-1], 64):
        ids = topk_idxs[..., begin:begin + 64].long()
        selected = kv[batch_ids, ids.clamp_min(0)].float()
        selected = selected.masked_fill((ids == -1).unsqueeze(-1), 0)
        scores = (q.float() @ selected.transpose(-1, -2)) * softmax_scale
        scores = scores.masked_fill((ids == -1).unsqueeze(-2), -torch.inf)
        new_maximum = torch.maximum(maximum, scores.amax(-1))
        correction = torch.exp(maximum - new_maximum)
        weights = torch.exp(scores - new_maximum.unsqueeze(-1))
        denominator = denominator * correction + weights.sum(-1)
        output = output * correction.unsqueeze(-1) + weights.to(torch.bfloat16).float() @ selected
        maximum = new_maximum
    denominator += torch.exp(attn_sink - maximum)
    return (output / denominator.unsqueeze(-1)).to(q.dtype)
