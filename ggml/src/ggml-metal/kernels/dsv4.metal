#include "common.h"
#include "dequantize.h"

constant int FC_dsv4_hc_n_hc [[function_constant(FC_DSV4_HC + 0)]];

// Metal arithmetic can flush subnormals. Apply power-of-two scales through the bits.
static float dsv4_scale_pow2(float x, int shift) {
    const uint bits = as_type<uint>(x);
    const uint sign = bits & 0x80000000;
    int exponent = (bits >> 23) & 255;
    uint mantissa = bits & 0x7fffff;
    if (exponent == 255 || (exponent == 0 && mantissa == 0)) {
        return x;
    }
    if (exponent == 0) {
        const int normalize = clz(mantissa) - 8;
        mantissa <<= normalize;
        exponent = 1 - normalize;
    } else {
        mantissa |= 0x800000;
    }
    exponent += shift;
    if (exponent >= 255) {
        return as_type<float>(sign | 0x7f800000);
    }
    if (exponent > 0) {
        return as_type<float>(sign | uint(exponent) << 23 | (mantissa & 0x7fffff));
    }
    const int right = 1 - exponent;
    if (right > 24) {
        return as_type<float>(sign);
    }
    uint rounded = mantissa >> right;
    const uint remainder = mantissa & ((1u << right) - 1);
    const uint halfway = 1u << (right - 1);
    rounded += remainder > halfway || (remainder == halfway && (rounded & 1));
    return as_type<float>(sign | rounded);
}

static float2 dsv4_pair_add(float2 a, float2 b) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    const float sum = a.x + b.x, v = sum - a.x;
    const float error = (a.x - (sum - v)) + (b.x - v) + (a.y + b.y);
    const float hi = sum + error;
    return float2(hi, error - (hi - sum));
}

static float2 dsv4_pair_mul(float2 a, float2 b) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    const float product = a.x*b.x;
    const float error = fma(a.x, b.x, -product) + (a.x*b.y + a.y*b.x);
    const float hi = product + error;
    return float2(hi, error - (hi - product));
}

// Extra precision keeps exp error from crossing BF16 boundaries.
static float dsv4_exp(float x) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    if (isnan(x)) { return x; }
    if (x > 89.0f) { return INFINITY; }
    if (x < -104.0f) { return 0.0f; }
    const float n = rint(x*0x1.715476p+0f);
    const float2 r = dsv4_pair_add(float2(x, 0), dsv4_pair_mul(float2(-n, 0), float2(0x1.62e4300000000p-1f, -0x1.05c6100000000p-29f)));
    constexpr float2 coefficients[] = {
        { 0x1.0000000000000p+0f, 0x0.0p+0f },
        { 0x1.0000000000000p+0f, 0x0.0p+0f },
        { 0x1.0000000000000p-1f, 0x0.0p+0f },
        { 0x1.5555560000000p-3f, -0x1.5555560000000p-28f },
        { 0x1.5555560000000p-5f, -0x1.5555560000000p-30f },
        { 0x1.1111120000000p-7f, -0x1.ddddde0000000p-32f },
        { 0x1.6c16c20000000p-10f, -0x1.27d27e0000000p-35f },
        { 0x1.a01a020000000p-13f, -0x1.7f97fa0000000p-39f },
        { 0x1.a01a020000000p-16f, -0x1.7f97fa0000000p-42f },
        { 0x1.71de3a0000000p-19f, 0x1.55b1cc0000000p-45f },
        { 0x1.27e4fc0000000p-22f, -0x1.10ec140000000p-47f },
        { 0x1.ae64560000000p-26f, 0x1.fd51380000000p-52f },
        { 0x1.1eed8e0000000p-29f, 0x1.ff1b120000000p-54f }
    };
    float2 value = coefficients[12];
    for (int i = 11; i >= 0; --i) { value = dsv4_pair_add(dsv4_pair_mul(value, r), coefficients[i]); }
    if (n <= -126) {
        const float2 scaled = float2(dsv4_scale_pow2(value.x, int(n) + 149), dsv4_scale_pow2(value.y, int(n) + 149));
        const float base = floor(scaled.x), fraction = scaled.x - base;
        uint bits = uint(base);
        bits += fraction > 0.5f || (fraction == 0.5f && (scaled.y > 0 || (scaled.y == 0 && (bits & 1))));
        return as_type<float>(bits);
    }
    return dsv4_scale_pow2(value.x, int(n));
}

kernel void kernel_dsv4_hc_split(
        constant ggml_metal_kargs_dsv4_hc_split & args,
        device const char  * mixes,
        device const float * scale,
        device const float * base,
        device       float * dst,
        uint3    tgpig[[threadgroup_position_in_grid]],
        ushort   tiisg[[thread_index_in_simdgroup]],
        ushort   sgitg[[simdgroup_index_in_threadgroup]],
        ushort3    ntg[[threads_per_threadgroup]]) {
#pragma clang fp contract(off)
    const int it = tgpig.x*ntg.y + sgitg;
    if (it >= args.n_tokens) { return; }
    device const float * x = (device const float *) (mixes + it*args.nb_m1);
    float v = tiisg < 24 ? x[tiisg]*scale[tiisg < 4 ? 0 : tiisg < 8 ? 1 : 2] + base[tiisg] : 0.0f;
    if (tiisg < 8) {
        const float p = precise::divide(1.0f, 1.0f + dsv4_exp(-v));
        dst[24*it + tiisg] = tiisg < 4 ? p + args.eps : 2*p;
    }
    const ushort first = tiisg & ~3;
    const ushort column = 8 + tiisg % 4;
    const float maximum = max(max(simd_shuffle(v, first), simd_shuffle(v, first + 1)), max(simd_shuffle(v, first + 2), simd_shuffle(v, first + 3)));
    v = dsv4_exp(v - maximum);
    float sum = ((simd_shuffle(v, first) + simd_shuffle(v, first + 1)) + simd_shuffle(v, first + 2)) + simd_shuffle(v, first + 3);
    v = precise::divide(v, sum) + args.eps;
    for (int iteration = 0; iteration < args.n_iter; ++iteration) {
        if (iteration) {
            sum = ((simd_shuffle(v, first) + simd_shuffle(v, first + 1)) + simd_shuffle(v, first + 2)) + simd_shuffle(v, first + 3);
            v = precise::divide(v, sum + args.eps);
        }
        sum = ((simd_shuffle(v, column) + simd_shuffle(v, column + 4)) + simd_shuffle(v, column + 8)) + simd_shuffle(v, column + 12);
        v = precise::divide(v, sum + args.eps);
    }
    if (tiisg >= 8 && tiisg < 24) { dst[24*it + tiisg] = v; }
}

kernel void kernel_dsv4_hc_pre_gated_f32(
        constant ggml_metal_kargs_dsv4_hc_pre & args,
        device const char * x,
        device const char * gate,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    if (i0 >= args.n_embd) {
        return;
    }

    device const char * xb = x    + i0*args.nb_x0 + it*args.nb_x2;
    device const char * gb = gate + i0*args.nb_w0 + it*args.nb_w2;
    float result = 0.0f;
    FOR_UNROLL (int ih = 0; ih < FC_dsv4_hc_n_hc; ++ih) {
        const float g  = 1.0f/(1.0f + exp(-*(device const float *) (gb + ih*args.nb_w1)));
        const float xv = *(device const float *) (xb + ih*args.nb_x1);
        result = fma(xv, g, result);
    }

    *(device float *) (dst + i0*args.nb_d0 + it*args.nb_d1) = args.scale*result;
}

kernel void kernel_dsv4_hc_up_q8_0_f32(
        constant ggml_metal_kargs_dsv4_hc_up & args,
        device const block_q8_0 * w,
        device const float * lo,
        device const float * x,
        device float * dst,
        uint tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NR = 2;
    const int i0 = (tgpig*4 + sgitg)*NR;
    if (i0 >= args.n_embd) {
        return;
    }

    const int nb = args.n_low/QK8_0;
    const short il = tiisg%4;
    float sums[4][NR] = {};
    for (int ib = tiisg/4; ib < nb; ib += 8) {
        float v[8];
        FOR_UNROLL (short j = 0; j < 8; ++j) {
            v[j] = lo[ib*QK8_0 + il*8 + j];
        }
        FOR_UNROLL (short ih = 0; ih < 4; ++ih) {
            FOR_UNROLL (short r = 0; r < NR; ++r) {
                if (i0 + r < args.n_embd) {
                    device const block_q8_0 * b = w + (uint64_t) (ih*args.n_embd + i0 + r)*nb + ib;
                    float sum = 0.0f;
                    FOR_UNROLL (short j = 0; j < 8; ++j) {
                        sum += b->qs[il*8 + j]*v[j];
                    }
                    sums[ih][r] += sum*b->d;
                }
            }
        }
    }

    float result[NR] = {};
    FOR_UNROLL (short ih = 0; ih < 4; ++ih) {
        FOR_UNROLL (short r = 0; r < NR; ++r) {
            const float gate = simd_sum(sums[ih][r]);
            if (tiisg == 0 && i0 + r < args.n_embd) {
                result[r] = fma(x[ih*args.n_embd + i0 + r], 1.0f/(1.0f + exp(-gate)), result[r]);
            }
        }
    }
    if (tiisg == 0) {
        FOR_UNROLL (short r = 0; r < NR; ++r) {
            if (i0 + r < args.n_embd) {
                dst[i0 + r] = args.scale*result[r];
            }
        }
    }
}

kernel void kernel_dsv4_hc_down_q8_0_f32(
        constant ggml_metal_kargs_dsv4_hc_down & args,
        device const float * x,
        device const float * gamma,
        device const block_q8_0 * wd,
        device const block_q8_0 * wi,
        device float * part,
        uint3 group [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (4*group.x + sg)*4;
    const uint stream = group.y;
    const uint rows = args.n_low + 4;
    if (row0 >= rows) {
        return;
    }
    const uint nb = args.n_embd/QK8_0;
    const uint n_parts = (nb + 19)/20;
    const uint il = lane%4;
    float sums[4] = {};
    float ss = 0.0f;
    for (uint ib = group.z*20 + lane/4; ib < min(nb, (group.z + 1)*20); ib += 8) {
        float values[8];
        FOR_UNROLL (uint j = 0; j < 8; ++j) {
            const uint col = stream*args.n_embd + ib*QK8_0 + il*8 + j;
            const float v = x[col];
            values[j] = v*gamma[col];
            if (row0 == 0) {
                ss += v*v;
            }
        }
        FOR_UNROLL (uint r = 0; r < 4; ++r) {
            const uint row = row0 + r;
            if (row < rows) {
                device const block_q8_0 * w = row < args.n_low ? wd + row*4*nb : wi + (row - args.n_low)*4*nb;
                w += stream*nb + ib;
                float v = 0.0f;
                FOR_UNROLL (uint j = 0; j < 8; ++j) {
                    v += values[j]*w->qs[il*8 + j];
                }
                sums[r] += v*float(w->d);
            }
        }
    }
    FOR_UNROLL (uint r = 0; r < 4; ++r) {
        const float sum = simd_sum(sums[r]);
        if (lane == 0 && row0 + r < rows) {
            part[(stream*n_parts + group.z)*rows + row0 + r] = sum;
        }
    }
    if (row0 == 0) {
        ss = simd_sum(ss);
        if (lane == 0) {
            part[4*n_parts*rows + group.z*4 + stream] = ss;
        }
    }
}

kernel void kernel_dsv4_hc_down_finish_f32(
        constant ggml_metal_kargs_dsv4_hc_down & args,
        device const float * x,
        device const float * gamma,
        device const float * part,
        device float * xn,
        device float * lo,
        device float * inject,
        uint i [[thread_position_in_grid]]) {
    const uint rows = args.n_low + 4;
    const uint n_parts = (args.n_embd + 639)/640;
    float4 ss = 0.0f;
    for (uint j = 0; j < n_parts; ++j) {
        ss += *(device const float4 *) (part + 4*n_parts*rows + j*4);
    }
    const float4 rs = rsqrt(ss/float(args.n_embd) + args.eps);
    if (i < 4*args.n_embd) {
        xn[i] = (x[i]*rs[i/args.n_embd])*gamma[i];
    }
    if (i < rows) {
        float sum = 0.0f;
        FOR_UNROLL (uint h = 0; h < 4; ++h) {
            float dot = 0.0f;
            for (uint j = 0; j < n_parts; ++j) {
                dot += part[(h*n_parts + j)*rows + i];
            }
            sum += dot*rs[h];
        }
        if (i < args.n_low) {
            const float v = sum*args.scale + args.bias;
            lo[i] = v/(1.0f + exp(-v));
        } else {
            inject[i - args.n_low] = sum;
        }
    }
}

kernel void kernel_dsv4_hc_post_gated_f32(
        constant ggml_metal_kargs_dsv4_hc_post & args,
        device const float * x,
        device const float * residual,
        device const float * gate,
        device       float * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float weight_lane = 0.0f;
    if (tiisg < hc) {
        weight_lane = 2.0f/(1.0f + exp(-args.scale*gate[it*hc + tiisg]));
    }

    float weight[hc];
    FOR_UNROLL (ushort ih = 0; ih < hc; ++ih) {
        weight[ih] = simd_shuffle(weight_lane, ih);
    }

    if (i0 >= args.n_embd) {
        return;
    }

    const float xv = x[it*args.n_embd + i0];
    const int offset = it*hc*args.n_embd + i0;
    FOR_UNROLL (ushort ih = 0; ih < hc; ++ih) {
        const int idx = offset + ih*args.n_embd;
        dst[idx] = fma(xv, weight[ih], residual[idx]);
    }
}

kernel void kernel_dsv4_hc_pre_norm_f32(
        constant ggml_metal_kargs_dsv4_hc_pre_norm & args,
        device const char * x,
        device const char * weights,
        device const char * norm,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint    tgpig[[threadgroup_position_in_grid]],
        ushort  tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]]) {
    constexpr ushort hc = 4;

    float weight_lane = 0.0f;
    if (tiisg < hc) {
        weight_lane = *(device const float *) (weights + tiisg*args.nb_w0 + tgpig*args.nb_w1);
    }

    float4 result = 0.0f;
    device const char * xb = x + 4*tpitg*args.nb_x0 + tgpig*args.nb_x2;
    FOR_UNROLL (ushort ih = 0; ih < hc; ++ih) {
        const float weight = simd_shuffle(weight_lane, ih);
        result = fma(*(device const float4 *) (xb + ih*args.nb_x1), weight, result);
    }

    float sumf = simd_sum(dot(result, result));

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = simd_sum(shmem_f32[tiisg]);
    const float scale = 1.0f/sqrt(sumf/args.n_embd + args.eps);

    const float4 norm_v = *(device const float4 *) (norm + 4*tpitg*args.nb_n0);
    *(device float4 *) (dst + 4*tpitg*args.nb_d0 + tgpig*args.nb_d1) = result*scale*norm_v;
}

static float dsv4_round_bf16(float x) {
    uint bits = as_type<uint>(x);
    bits = (bits & 0x7fffffff) > 0x7f800000 ? bits | 0x400000 : bits + 0x7fff + ((bits >> 16) & 1);
    return as_type<float>(bits & 0xffff0000);
}

kernel void kernel_dsv4_swiglu(
        constant ggml_metal_kargs_dsv4_swiglu & args,
        device const char * gate,
        device const char * up,
        device const char * weights,
        device      float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    const int i = 128*tgpig.x + tiitg, row = tgpig.y;
    if (i >= args.ne0) { return; }
    const int i1 = row % args.ne1, i2 = row / args.ne1 % args.ne2, i3 = row / (args.ne1*args.ne2);
    device const float * g = (device const float *) (gate + i1*args.nb_g1 + i2*args.nb_g2 + i3*args.nb_g3);
    device const float * u = (device const float *) (up + i1*args.nb_u1 + i2*args.nb_u2 + i3*args.nb_u3);
    const float w = args.weighted ? *(device const float *) (weights + i1*args.nb_w1 + i2*args.nb_w2 + i3*args.nb_w3) : 1.0f;
    const float gate_value = dsv4_round_bf16(g[i]);
    const float up_value = dsv4_round_bf16(u[i]);
    const float a = args.limit > 0 ? min(gate_value, args.limit) : gate_value;
    const float b = args.limit > 0 ? clamp(up_value, -args.limit, args.limit) : up_value;
    const float silu = precise::divide(a, 1.0f + precise::exp(-a));
    dst[row*args.ne0 + i] = dsv4_round_bf16((silu*b)*w);
}

kernel void kernel_dsv4_hc_affine_f32(
        constant ggml_metal_kargs_dsv4_hc_affine & args,
        device const char * x,
        device const float * scale,
        device const float4 * base,
        device       char * dst,
        uint it[[thread_position_in_grid]]) {
    if (it >= args.n_tokens) {
        return;
    }

    const float4 z = *(device const float4 *) (x + it*args.nb_x1) * scale[0] + base[0];
    const float4 result = 1.0f / (1.0f + exp(-z));
    *(device float4 *) (dst + it*args.nb_d1) = result*args.post_scale + args.post_bias;
}

kernel void kernel_dsv4_compress(
        constant ggml_metal_kargs_dsv4_compress & args,
        device const char * kv_state,
        device const char * score_state,
        device const char * read_idxs,
        device       char * dst,
        threadgroup int32_t * idxs [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int n_read = (args.overlap ? 2 : 1)*args.ratio;
    const int ib = tgpig.y;

    for (int j = tiitg; j < n_read; j += ntg.x) {
        const bool cur_half = args.overlap && j >= args.ratio;
        const int jr = cur_half ? j - args.ratio : j;
        const int idx_pos = (cur_half ? args.ratio*args.n_blocks : 0) + ib*args.ratio + jr;
        idxs[j] = *(device const int32_t *) (read_idxs + idx_pos*args.nb_i0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!args.overlap && args.ratio == 128) {
        const int lane = tiitg & 31;
        const int nsg = ntg.x/32;
        const int i0 = tgpig.x*(2*nsg) + (tiitg >> 5)*2 + (lane & 1);
        const int worker = lane >> 1;

        float score_max = -INFINITY;
        float sum_v = 0.0f;
        float sum_w = 0.0f;
        for (int j = worker; i0 < args.n_embd && j < 128; j += 16) {
            const int idx = idxs[j];
            if (idx < 0 || idx >= args.n_rows) {
                continue;
            }

            const float score = *(device const float *) (score_state + i0*args.nb_s0 + idx*args.nb_s1);
            const float value = *(device const float *) (kv_state + i0*args.nb_k0 + idx*args.nb_k1);
            if (score > score_max) {
                const float scale = fast::exp(score_max - score);
                sum_v = sum_v*scale + value;
                sum_w = sum_w*scale + 1.0f;
                score_max = score;
            } else {
                const float weight = fast::exp(score - score_max);
                sum_v += value*weight;
                sum_w += weight;
            }
        }

        float group_max = max(score_max, simd_shuffle_xor(score_max, 2));
        group_max = max(group_max, simd_shuffle_xor(group_max, 4));
        group_max = max(group_max, simd_shuffle_xor(group_max, 8));
        group_max = max(group_max, simd_shuffle_xor(group_max, 16));

        float scale = 0.0f;
        if (sum_w > 0.0f) {
            scale = fast::exp(score_max - group_max);
        }
        sum_v *= scale;
        sum_w *= scale;
        sum_v += simd_shuffle_xor(sum_v, 2);
        sum_w += simd_shuffle_xor(sum_w, 2);
        sum_v += simd_shuffle_xor(sum_v, 4);
        sum_w += simd_shuffle_xor(sum_w, 4);
        sum_v += simd_shuffle_xor(sum_v, 8);
        sum_w += simd_shuffle_xor(sum_w, 8);
        sum_v += simd_shuffle_xor(sum_v, 16);
        sum_w += simd_shuffle_xor(sum_w, 16);

        if (worker == 0 && i0 < args.n_embd) {
            *(device float *) (dst + i0*args.nb_d0 + ib*args.nb_d1) = sum_w > 0.0f ? sum_v/sum_w : 0.0f;
        }
        return;
    }

    const int i0 = tgpig.x*ntg.x + tiitg;
    if (i0 >= args.n_embd) {
        return;
    }

    float score_max = -INFINITY;
    float sum_v = 0.0f;
    float sum_w = 0.0f;
    for (int j = 0; j < n_read; ++j) {
        const int idx = idxs[j];
        if (idx < 0 || idx >= args.n_rows) {
            continue;
        }

        const bool cur_half = args.overlap && j >= args.ratio;
        const int i_src = (cur_half ? args.n_embd : 0) + i0;
        const float score = *(device const float *) (score_state + i_src*args.nb_s0 + idx*args.nb_s1);
        const float value = *(device const float *) (kv_state + i_src*args.nb_k0 + idx*args.nb_k1);
        if (score > score_max) {
            const float scale = fast::exp(score_max - score);
            sum_v = sum_v*scale + value;
            sum_w = sum_w*scale + 1.0f;
            score_max = score;
        } else {
            const float weight = fast::exp(score - score_max);
            sum_v += value*weight;
            sum_w += weight;
        }
    }

    *(device float *) (dst + i0*args.nb_d0 + ib*args.nb_d1) = sum_w > 0.0f ? sum_v/sum_w : 0.0f;
}

kernel void kernel_dsv4_top_k_mask(
        constant ggml_metal_kargs_dsv4_top_k_mask & args,
        device const char * raw_mask,
        device const char * comp_mask,
        device const char * comp_idx,
        device       char * dst,
        uint    tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const int iq = tgpig % args.n_query;
    const int is = tgpig / args.n_query;

    device const half * rm = (device const half *) (raw_mask + (uint64_t) iq*args.nb_rm1 + (uint64_t) is*args.nb_rm3);
    device const half * cm = (device const half *) (comp_mask + (uint64_t) iq*args.nb_cm1 + (uint64_t) is*args.nb_cm3);
    device const int  * ci = (device const int  *) (comp_idx + (uint64_t) iq*args.nb_ci1 + (uint64_t) is*args.nb_ci3);
    device       half * out = (device half *) (dst + (uint64_t) iq*args.nb_d1 + (uint64_t) is*args.nb_d3);

    for (int i = tiitg; i < args.n_raw; i += ntg) {
        out[i] = rm[i];
    }
    for (int i = tiitg; i < args.n_comp; i += ntg) {
        out[args.n_raw + i] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (int i = tiitg; i < args.n_select; i += ntg) {
        const int idx = ci[i];
        if (idx >= 0 && idx < args.n_comp) {
            out[args.n_raw + idx] = cm[idx];
        }
    }
}

template <typename k4_t, short nl, void (*dequantize_func)(device const k4_t *, short, thread half4 &)>
kernel void kernel_dsv4_sparse_pack(
        constant ggml_metal_kargs_dsv4_sparse_pack & args,
        device const char * raw_k,
        device const char * comp_k,
        device const char * raw_mask,
        device const char * comp_mask,
        device const char * comp_idx,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {
    constexpr int max_selected = 128 + 512;
    threadgroup int  selected_idx[max_selected];
    threadgroup half selected_mask[max_selected];

    if (args.n_raw == 0) {
        const int i  = tgpig.x;
        const int it = tgpig.y;
        const int iq = it % args.n_batch;
        const int is = it / args.n_batch;
        const int idx = *(device const int *) (comp_idx + (uint64_t) i*args.nb_ci0 + (uint64_t) iq*args.nb_ci1 + (uint64_t) is*args.nb_ci3);

        device const k4_t * src = (device const k4_t *) (comp_k + (uint64_t) idx*args.nb_ck2 + (uint64_t) is*args.nb_ck3);
        device half * out = (device half *) (dst + (uint64_t) it*args.nb_d1);
        device half4 * out_k = (device half4 *) out;
        for (int e4 = tiitg; e4 < args.n_embd/4; e4 += ntg.x) {
            half4 values;
            dequantize_func(src + e4/nl, e4%nl, values);
            out_k[i*args.n_embd/4 + e4] = values;
        }
        if (tiitg == 0) {
            out[args.n_embd*args.n_comp + i] = *(device const half *) (comp_mask + (uint64_t) idx*args.nb_cm0 + (uint64_t) iq*args.nb_cm1 + (uint64_t) is*args.nb_cm3);
        }
        return;
    }

    const int it = tgpig.x;
    const int iq = it % args.n_batch;
    const int is = it / args.n_batch;
    const int nk = args.n_raw + args.n_comp;

    device half * out = (device half *) (dst + (uint64_t) it*args.nb_d1);
    device half * out_k = out;
    device half * out_m = out + args.n_embd*nk;

    if (tiitg == 0) {
        int n = 0;
        for (int idx = 0; idx < args.n_raw_k && n < args.n_raw; ++idx) {
            const half m = *(device const half *) (raw_mask + (uint64_t) idx*args.nb_rm0 + (uint64_t) iq*args.nb_rm1 + (uint64_t) is*args.nb_rm3);
            if (isfinite(m)) {
                selected_idx[n] = idx;
                selected_mask[n] = m;
                ++n;
            }
        }
        for (; n < args.n_raw; ++n) {
            selected_idx[n] = -1;
            selected_mask[n] = -INFINITY;
        }
    }
    for (int i = tiitg; i < args.n_comp; i += ntg.x) {
        const int oi = args.n_raw + i;
        const int idx = *(device const int *) (comp_idx + (uint64_t) i*args.nb_ci0 + (uint64_t) iq*args.nb_ci1 + (uint64_t) is*args.nb_ci3);
        selected_idx[oi] = idx;
        selected_mask[oi] = *(device const half *) (comp_mask + (uint64_t) idx*args.nb_cm0 + (uint64_t) iq*args.nb_cm1 + (uint64_t) is*args.nb_cm3);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 0; i < args.n_raw; ++i) {
        const int idx = selected_idx[i];
        if (idx >= 0) {
            device const k4_t * src = (device const k4_t *) (raw_k + (uint64_t) idx*args.nb_rk2 + (uint64_t) is*args.nb_rk3);
            for (int e4 = tiitg; e4 < args.n_embd/4; e4 += ntg.x) {
                half4 values;
                dequantize_func(src + e4/nl, e4%nl, values);
                ((device half4 *) out_k)[i*args.n_embd/4 + e4] = values;
            }
        } else {
            for (int e4 = tiitg; e4 < args.n_embd/4; e4 += ntg.x) {
                ((device half4 *) out_k)[i*args.n_embd/4 + e4] = 0.0h;
            }
        }
        if (tiitg == 0) {
            out_m[i] = selected_mask[i];
        }
    }

    for (int i = 0; i < args.n_comp; ++i) {
        const int oi = args.n_raw + i;
        const int idx = selected_idx[oi];
        device const k4_t * src = (device const k4_t *) (comp_k + (uint64_t) idx*args.nb_ck2 + (uint64_t) is*args.nb_ck3);
        for (int e4 = tiitg; e4 < args.n_embd/4; e4 += ntg.x) {
            half4 values;
            dequantize_func(src + e4/nl, e4%nl, values);
            ((device half4 *) out_k)[oi*args.n_embd/4 + e4] = values;
        }
        if (tiitg == 0) {
            out_m[oi] = selected_mask[oi];
        }
    }
}

typedef decltype(kernel_dsv4_sparse_pack<half4, 1, dequantize_f16_t4>) dsv4_sparse_pack_t;

template [[host_name("kernel_dsv4_sparse_pack_f16")]]
kernel dsv4_sparse_pack_t kernel_dsv4_sparse_pack<half4, 1, dequantize_f16_t4>;

template [[host_name("kernel_dsv4_sparse_pack_q8_0")]]
kernel dsv4_sparse_pack_t kernel_dsv4_sparse_pack<block_q8_0, 8, dequantize_q8_0_t4>;
