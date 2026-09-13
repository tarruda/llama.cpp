#include "common.h"
#include "dequantize.h"

kernel void kernel_argmax_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    device const float * x_row = (device const float *) ((device const char *) src0 + tgpig * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = tpitg; i00 < args.ne00; i00 += ntg) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    device int32_t * dst_i32 = (device int32_t *) dst;

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        dst_i32[tgpig] = arg_val_reduced;

        return;
    }

    dst_i32[tgpig] = arg_val;
}

kernel void kernel_diag_f32(
        constant ggml_metal_kargs_diag & args,
        device   const char * src0,
        device         char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;

    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *)(src0 +                i2*args.nb02 + i3*args.nb03);
    device       float * dst_ptr  = (device       float *)(dst  + i1*args.nb01 + i2*args.nb2  + i3*args.nb3);

    for (int i0 = tiitg; i0 < args.ne0; i0 += NW) {
        dst_ptr[i0] = i0 == i1 ? src0_ptr[i0] : 0.0f;
    }
}

kernel void kernel_roll_f32(
    constant ggml_metal_kargs_roll & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *) src0;
    device       float * dst_ptr  = (device       float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        // apply shifts and wrap around
        int64_t i00 = i0 - args.s0;
        int64_t i01 = i1 - args.s1;
        int64_t i02 = i2 - args.s2;
        int64_t i03 = i3 - args.s3;

        if (i00 < 0) { i00 += args.ne00; } else if (i00 >= args.ne00) { i00 -= args.ne00; }
        if (i01 < 0) { i01 += args.ne01; } else if (i01 >= args.ne01) { i01 -= args.ne01; }
        if (i02 < 0) { i02 += args.ne02; } else if (i02 >= args.ne02) { i02 -= args.ne02; }
        if (i03 < 0) { i03 += args.ne03; } else if (i03 >= args.ne03) { i03 -= args.ne03; }

        int64_t src_idx = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00 + i00;
        int64_t dst_idx = i3 *args.ne2 *args.ne1 *args.ne0  + i2 *args.ne1 *args.ne0  + i1 *args.ne0  + i0;

        dst_ptr[dst_idx] = src0_ptr[src_idx];
    }
}

template <typename T>
kernel void kernel_pad_impl(
    constant ggml_metal_kargs_pad & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {
    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t k0 = tgpig.x/args.ne1;
    const int32_t i1 = tgpig.x - k0*args.ne1;

    const int32_t i03 = i3;
    const int32_t i02 = i2;
    const int32_t i01 = i1;

    device const T * src0_ptr = (device const T *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       T * dst_ptr  = (device       T *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);

    for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {
        const int32_t i0 = k0*1024 + tpitg.x + l0;
        if (i0 >= args.ne0) {
            break;
        }

        if (i0 < args.ne00 && i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
            dst_ptr[i0] = src0_ptr[i0];
        } else {
            dst_ptr[i0] = 0.0f;
        }
    }
}

typedef decltype(kernel_pad_impl<float>) kernel_pad_t;

template [[host_name("kernel_pad_f32")]]   kernel kernel_pad_t kernel_pad_impl<float>;
template [[host_name("kernel_pad_f32_4")]] kernel kernel_pad_t kernel_pad_impl<float4>;

// TODO: this is slow - optimize
kernel void kernel_pad_reflect_1d_f32(
    constant   ggml_metal_kargs_pad_reflect_1d & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3  tgpg[[threadgroups_per_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);

    if (i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            if (i0 < args.p0) {
                dst_ptr[i0] = src0_ptr[args.p0 - i0];
            } else if (i0 < args.ne0 - args.p1) {
                dst_ptr[i0] = src0_ptr[i0 - args.p0];
            } else {
                dst_ptr[i0] = src0_ptr[(args.ne0 - args.p1 - args.p0) - (args.p1 + 1 - (args.ne0 - i0)) - 1];
            }
        }
    }
}

kernel void kernel_arange_f32(
    constant   ggml_metal_kargs_arange & args,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    device float * dst_ptr = (device float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_ptr[i0] = args.start + args.step * i0;
    }
}

kernel void kernel_timestep_embedding_f32(
    constant  ggml_metal_kargs_timestep_embedding & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    int i = tgpig.x;
    device float * embed_data = (device float *)(dst + i*args.nb1);

    int half_ = args.dim / 2;
    for (int j = tpitg.x; j < half_; j += ntg.x) {
        float timestep = ((device float *)src0)[i];
        float freq = (float)exp(-log((float)args.max_period) * j / half_);
        float arg = timestep * freq;
        embed_data[j        ] = cos(arg);
        embed_data[j + half_] = sin(arg);
    }

    if (args.dim % 2 != 0 && tpitg.x == 0) {
        embed_data[2 * half_] = 0.f;
    }
}

kernel void kernel_opt_step_adamw_f32(
        constant    ggml_metal_kargs_opt_step_adamw & args,
        device       float * x,
        device const float * g,
        device       float * g_m,
        device       float * g_v,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];

    const float gi = g[gid];
    const float gmi = g_m[gid] * beta1 +      gi * (1.0f - beta1);
    const float gvi = g_v[gid] * beta2 + gi * gi * (1.0f - beta2);

    g_m[gid] = gmi;
    g_v[gid] = gvi;

    const float mh =      gmi * beta1h;
    const float vh = sqrt(gvi * beta2h) + eps;

    x[gid] = x[gid] * (1.0f - alpha * wd) - alpha * mh / vh;
}

kernel void kernel_opt_step_sgd_f32(
        constant    ggml_metal_kargs_opt_step_sgd & args,
        device       float * x,
        device const float * g,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    x[gid] = x[gid] * (1.0f - pars[0] * pars[1]) - pars[0] * g[gid];
}

template<typename T>
kernel void kernel_memset(
        constant ggml_metal_kargs_memset & args,
        device T * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = args.val;
}

typedef decltype(kernel_memset<int64_t>) kernel_memset_t;

template [[host_name("kernel_memset_i64")]] kernel kernel_memset_t kernel_memset<int64_t>;

constant short FC_count_equal_nsg [[function_constant(FC_COUNT_EQUAL + 0)]];

template<typename T>
kernel void kernel_count_equal(
        constant ggml_metal_kargs_count_equal & args,
        device   const char * src0,
        device   const char * src1,
        device   atomic_int * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const short NSG = FC_count_equal_nsg;

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    if (i3 >= args.ne03 || i2 >= args.ne02 || i1 >= args.ne01) {
        return;
    }

    int sum = 0;

    device const char * base0 = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;
    device const char * base1 = src1 + i1*args.nb11 + i2*args.nb12 + i3*args.nb13;

    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        const T v0 = *(device const T *)(base0 + i0*args.nb00);
        const T v1 = *(device const T *)(base1 + i0*args.nb10);
        sum += (v0 == v1);
    }

    sum = simd_sum(sum);

    if (tiisg == 0) {
        shmem_i32[sgitg] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        float v = 0.0f;
        if (tpitg.x < NSG) {
            v = shmem_i32[tpitg.x];
        }

        float total = simd_sum(v);
        if (tpitg.x == 0) {
            atomic_fetch_add_explicit(dst, (int32_t) total, memory_order_relaxed);
        }
    }
}

typedef decltype(kernel_count_equal<int32_t>) kernel_count_equal_t;

template [[host_name("kernel_count_equal_i32")]] kernel kernel_count_equal_t kernel_count_equal<int32_t>;

template <typename T>
kernel void kernel_snake(
        constant ggml_metal_kargs_snake & args,
        device const T     * x,
        device const float * a,
        device const float * inv_b,
        device       T     * dst,
        uint         tgpig [[threadgroup_position_in_grid]],
        uint         tpitg [[thread_position_in_threadgroup]],
        uint         ntg   [[threads_per_threadgroup]]) {

    const int idx = tgpig * ntg + tpitg;
    if (idx >= args.T * args.C) {
        return;
    }

    const int   c  = idx / args.T;  // x is [T, C], a / inv_b collapse to [1, C]
    const float xi = float(x[idx]);
    const float si = sin(a[c] * xi);
    dst[idx] = T(xi + si * si * inv_b[c]);
}

template [[host_name("kernel_snake_f32")]]  kernel void kernel_snake<float>(constant ggml_metal_kargs_snake &, device const float *, device const float *, device const float *, device float *, uint, uint, uint);
template [[host_name("kernel_snake_f16")]]  kernel void kernel_snake<half>(constant ggml_metal_kargs_snake &, device const half *, device const float *, device const float *, device half *, uint, uint, uint);
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_snake_bf16")]] kernel void kernel_snake<bfloat>(constant ggml_metal_kargs_snake &, device const bfloat *, device const float *, device const float *, device bfloat *, uint, uint, uint);
#endif

template<int N>
kernel void kernel_fwht_f32(
        constant ggml_metal_kargs_fwht & args,
        device const float * src,
        device float * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NW;

    const float scale = 1.0f / sqrt((float) N);

    const int sg_per_tg = ntg.x / NW;
    const int64_t r = tgpig.x * sg_per_tg + sgitg;
    if (r >= args.nrows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int lane = tiisg;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        reg[i] = src[i*NW + lane]*scale;
    }
    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = (lane & i) == 0 ? val2 + val : val2 - val;
        }
    }

    for (int i = NW; i < N; i *= 2) {
        const int step = i / NW;
        for (int j = 0; j < NE; j += (2 * step)) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k ];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    for (int i = 0; i < NE; i++) {
        dst[i*NW + lane] = reg[i];
    }
}

typedef decltype(kernel_fwht_f32<64>) kernel_fwht_t;

template [[host_name("kernel_fwht_f32_64")]]  kernel kernel_fwht_t kernel_fwht_f32<64>;
template [[host_name("kernel_fwht_f32_128")]] kernel kernel_fwht_t kernel_fwht_f32<128>;
template [[host_name("kernel_fwht_f32_256")]] kernel kernel_fwht_t kernel_fwht_f32<256>;
template [[host_name("kernel_fwht_f32_512")]] kernel kernel_fwht_t kernel_fwht_f32<512>;

kernel void kernel_moe_combine_f32(
        constant ggml_metal_kargs_moe_combine & args,
        device const float * experts,
        device const float * weights,
        device       float * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]]) {
    const int it = tgpig.y;
    const int i0 = (int) tgpig.x*32 + tiisg;

    const float weight_lane = tiisg < args.n_expert ? weights[it*args.n_expert + tiisg] : 0.0f;
    float result = 0.0f;
    for (int ie = 0; ie < args.n_expert; ++ie) {
        const float weight = simd_shuffle(weight_lane, ie);
        if (i0 < args.n_embd) {
            const int idx = (it*args.n_expert + ie)*args.n_embd + i0;
            result = fma(experts[idx], weight, result);
        }
    }

    if (i0 < args.n_embd) {
        dst[it*args.n_embd + i0] = result;
    }
}

kernel void kernel_moe_weights_f32(
        constant ggml_metal_kargs_moe_weights & args,
        device const char * probs,
        device const char * ids,
        device       char * dst,
        uint    it[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]]) {
    if (it >= args.n_tokens) {
        return;
    }

    float weight = 0.0f;
    if (tiisg < args.n_expert_used) {
        const int32_t id = *(device const int32_t *) (ids + tiisg*args.nb_i0 + it*args.nb_i1);
        weight = *(device const float *) (probs + id*args.nb_p1 + it*args.nb_p2);
    }

    const float sum = clamp(simd_sum(weight), args.clamp_min, args.clamp_max);
    if (tiisg < args.n_expert_used) {
        *(device float *) (dst + tiisg*args.nb_d1 + it*args.nb_d2) = weight/sum*args.scale + args.bias;
    }
}

kernel void kernel_dsv4_hc_comb_f32(
        constant ggml_metal_kargs_dsv4_hc_comb & args,
        device const char * mixes,
        device const char * scale,
        device const char * base,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;
    constexpr ushort comb_offset = 2*hc;

    const int it = tgpig.x*ntg.y + sgitg;
    if (it >= args.n_tokens) {
        return;
    }

    float scale_lane = 0.0f;
    if (tiisg == 0) {
        scale_lane = *(device const float *) (scale + 2*args.nb_s0);
    }
    const float scale_comb = simd_shuffle(scale_lane, 0);

    float v = 0.0f;
    if (tiisg < hc*hc) {
        v = *(device const float *) (mixes + (comb_offset + tiisg)*args.nb_m0 + it*args.nb_m1)*scale_comb
          + *(device const float *) (base   + (comb_offset + tiisg)*args.nb_b0);
    }

    // Softmax across destinations (the four contiguous lanes for each source).
    float vmax = max(v, simd_shuffle_xor(v, 1));
    vmax = max(vmax, simd_shuffle_xor(vmax, 2));
    v = exp(v - vmax);

    float sum = v + simd_shuffle_xor(v, 1);
    sum += simd_shuffle_xor(sum, 2);
    v = v/sum + args.eps;

    // Normalize columns: equal destination indices are four lanes apart.
    sum = v + simd_shuffle_xor(v, 4);
    sum += simd_shuffle_xor(sum, 8);
    v /= sum + args.eps;

    for (int i = 1; i < args.n_iter; ++i) {
        sum = v + simd_shuffle_xor(v, 1);
        sum += simd_shuffle_xor(sum, 2);
        v /= sum + args.eps;

        sum = v + simd_shuffle_xor(v, 4);
        sum += simd_shuffle_xor(sum, 8);
        v /= sum + args.eps;
    }

    if (tiisg < hc*hc) {
        const ushort idst = tiisg & 3;
        const ushort isrc = tiisg >> 2;
        *(device float *) (dst + idst*args.nb_d0 + isrc*args.nb_d1 + it*args.nb_d2) = v;
    }
}

static uchar dsv41_e4m3_rne(float x) {
    const uchar sign = (as_type<uint>(x) >> 24) & 0x80;
    if (isnan(x)) {
        return sign | 0x7f;
    }
    const float a = min(abs(x), 448.0f);
    if (a < 0x1p-6f) {
        const float scaled = a*512.0f;
        const uint lo = uint(scaled);
        const float remainder = scaled - lo;
        return sign | uchar(lo + (remainder > 0.5f || (remainder == 0.5f && (lo & 1))));
    }
    uint bits = as_type<uint>(a);
    bits += 0x7ffff + ((bits >> 20) & 1);
    return sign | uchar((bits >> 20) - 120*8);
}

static float dsv41_e4m3_value(uchar code) {
    const uint magnitude = code & 127;
    const uint exponent = magnitude >> 3;
    const float value = exponent == 0 ? float(magnitude)*0x1p-9f : as_type<float>((exponent + 120) << 23 | (magnitude & 7) << 20);
    return copysign(magnitude == 127 ? NAN : value, code & 128 ? -1.0f : 1.0f);
}

static float dsv41_e8m0_scale(float x) {
    const uint bits = as_type<uint>(x);
    const uint exponent = ((bits >> 23) & 255) + ((bits & 0x7fffff) != 0);
    return as_type<float>(exponent << 23);
}

// Metal arithmetic can flush subnormals. Apply power-of-two scales through the bits.
static float dsv41_scale_pow2(float x, int shift) {
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

static uchar dsv41_e2m1_code(float x) {
    constexpr float midpoints[] = { 0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f };
    const float a = abs(x);
    uint code = 0;
    FOR_UNROLL (uint i = 0; i < 7; ++i) {
        if (a > midpoints[i] || (a == midpoints[i] && (i & 1))) {
            code = i + 1;
        } else {
            break;
        }
    }
    return uchar(code | ((as_type<uint>(x) >> 28) & 8));
}

static float dsv41_e2m1_value(float x) {
    constexpr float values[] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    return copysign(values[dsv41_e2m1_code(x) & 7], x);
}

kernel void kernel_dsv41_act_quant(
        constant ggml_metal_kargs_dsv41_act_quant & args,
        device const char * src,
        device      float * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;
    const int row = tgpig.y;
    const int i1 = row % args.ne1;
    const int i2 = row / args.ne1 % args.ne2;
    const int i3 = row / (args.ne1*args.ne2);
    const float x = i0 < args.ne0 ? *(device const float *) (src + i1*args.nb01 + i2*args.nb02 + i3*args.nb03 + i0*sizeof(float)) : 0.0f;

    float result;
    if (args.n_bits == 16) {
        uint bits = as_type<uint>(x);
        bits = (bits & 0x7fffffff) > 0x7f800000 ? bits | 0x400000 : bits + 0x7fff + ((bits >> 16) & 1);
        result = as_type<float>(bits & 0xffff0000);
    } else {
        float amax = abs(x);
        float scale;
        if (args.block_size == 16) {
            FOR_UNROLL (ushort offset = 1; offset < 16; offset *= 2) {
                amax = max(amax, simd_shuffle_xor(amax, offset));
            }
            scale = dsv41_e4m3_value(dsv41_e4m3_rne(precise::divide(max(amax, 6.0f*0x1p-9f), 6.0f)));
        } else {
            amax = simd_max(amax);
            const float unrounded = args.n_bits == 4 ? max(amax, 6.0f*0x1p-126f)*(1.0f/6.0f) : max(amax, 1e-4f)*(1.0f/448.0f);
            scale = dsv41_e8m0_scale(unrounded);
        }
        const int exponent = int(as_type<uint>(scale) >> 23) - 127;
        const float value = args.block_size == 32 ? dsv41_scale_pow2(x, -exponent) : precise::divide(x, scale);
        const float rounded = args.n_bits == 8 ? dsv41_e4m3_value(dsv41_e4m3_rne(value)) : dsv41_e2m1_value(value);
        result = args.block_size == 32 ? dsv41_scale_pow2(rounded, exponent) : rounded*scale;
    }
    if (i0 < args.ne0) {
        dst[row*args.ne0 + i0] = result;
    }
}

kernel void kernel_dsv41_set_rows(
        constant ggml_metal_kargs_dsv41_set_rows & args,
        device const char * src,
        device const char * indices,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int block = tgpig.x*ntg.y + sgitg, row = tgpig.y;
    const int id = *(device const int *) (indices + row*args.nb_i0);
    if (id < 0 || id >= args.n_rows || block*32 >= args.dim) { return; }
    const float x = *((device const float *) (src + row*args.nb_x1) + block*32 + tiisg);
    float amax = abs(x), scale;
    if (args.block_size == 16) {
        FOR_UNROLL (ushort offset = 1; offset < 16; offset *= 2) {
            amax = max(amax, simd_shuffle_xor(amax, offset));
        }
        scale = dsv41_e4m3_value(dsv41_e4m3_rne(precise::divide(max(amax, 6.0f*0x1p-9f), 6.0f)));
    } else {
        amax = simd_max(amax);
        const float unrounded = args.n_bits == 4 ? max(amax, 6.0f*0x1p-126f)*(1.0f/6.0f) : max(amax, 1e-4f)*(1.0f/448.0f);
        scale = dsv41_e8m0_scale(unrounded);
    }
    const uint exponent = as_type<uint>(scale) >> 23;
    const float value = args.block_size == 32 ? dsv41_scale_pow2(x, 127 - int(exponent)) : precise::divide(x, scale);
    device uchar * out = (device uchar *) (dst + id*args.nb_d1);
    if (args.n_bits == 8) {
        if (tiisg == 0) { out[block*33] = uchar(exponent); }
        out[block*33 + 1 + tiisg] = dsv41_e4m3_rne(value);
    } else {
        const uint code = dsv41_e2m1_code(value);
        const ushort offset = args.block_size/2;
        const uint other = simd_shuffle_xor(code, offset);
        if (args.block_size == 16) {
            const int sub = 2*(block % 2) + tiisg/16;
            out += (block/2)*36;
            if (tiisg % 16 == 0) { out[sub] = dsv41_e4m3_rne(scale); }
            if (tiisg % 16 < 8) { out[4 + sub*8 + tiisg % 16] = uchar(code | (other << 4)); }
        } else {
            if (tiisg == 0) { out[block*17] = uchar(exponent); }
            if (tiisg < 16) { out[block*17 + 1 + tiisg] = uchar(code | (other << 4)); }
        }
    }
}

static float dsv41_round_bf16(float x) {
    uint bits = as_type<uint>(x);
    bits = (bits & 0x7fffffff) > 0x7f800000 ? bits | 0x400000 : bits + 0x7fff + ((bits >> 16) & 1);
    return as_type<float>(bits & 0xffff0000);
}

static float4 dsv41_round_bf16(float4 x) {
    return float4(dsv41_round_bf16(x.x), dsv41_round_bf16(x.y), dsv41_round_bf16(x.z), dsv41_round_bf16(x.w));
}

kernel void kernel_dsv41_moe_combine_f32(
        constant ggml_metal_kargs_dsv41_moe_combine & args,
        device const char * experts,
        device const char * shared,
        device char * dst,
        uint2 i[[thread_position_in_grid]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    if (i.x >= (uint) args.ne0 || i.y >= (uint) args.n_tokens) {
        return;
    }

    experts += i.y*args.nb_e2;
    const device float4 * e0 = (device const float4 *) (experts + 0*args.nb_e1);
    const device float4 * e1 = (device const float4 *) (experts + 1*args.nb_e1);
    const device float4 * e2 = (device const float4 *) (experts + 2*args.nb_e1);
    const device float4 * e3 = (device const float4 *) (experts + 3*args.nb_e1);
    const device float4 * e4 = (device const float4 *) (experts + 4*args.nb_e1);
    const device float4 * e5 = (device const float4 *) (experts + 5*args.nb_e1);
    float4 sum = dsv41_round_bf16(e0[i.x]) + dsv41_round_bf16(e1[i.x]);
    sum = sum + dsv41_round_bf16(e2[i.x]);
    sum = sum + dsv41_round_bf16(e3[i.x]);
    sum = sum + dsv41_round_bf16(e4[i.x]);
    sum = sum + dsv41_round_bf16(e5[i.x]);
    const float4 addend = *(device const float4 *) (shared + i.y*args.nb_s1 + i.x*sizeof(float4));
    *(device float4 *) (dst + i.y*args.nb_d1 + i.x*sizeof(float4)) = dsv41_round_bf16(sum + addend);
}

static float dsv41_index_score(
        constant ggml_metal_kargs_dsv41_index_scores & args,
        device const char * q, device const char * keys, device const char * weights,
        device const char * positions, device const char * candidates, int slot, int it, ushort tiisg) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    long ik = slot;
    const int pos = *(device const int *) (positions + it*args.nb_p0);
    if (args.n_candidates) {
        const int id = *(device const int *) (candidates + it*args.nb_c1 + (slot/args.block_size)*args.nb_c0);
        ik = long(id)*args.block_size + slot % args.block_size;
    }
    const bool valid = ik >= 0 && ik < args.n_keys && ik < (long(pos) + 1)/args.ratio;
    float sum = -INFINITY;
    if (valid) {
        device const uchar * key = (device const uchar *) (keys + ik*args.nb_k1);
        device const float * w = (device const float *) (weights + it*args.nb_w1);
        float decoded[4];
        if (args.dim == 128) {
            for (int block = 0; block < 4; ++block) {
                const uint code = (key[block*17 + 1 + tiisg % 16] >> (tiisg/16*4)) & 15;
                decoded[block] = dsv41_scale_pow2(kvalues_mxfp4_f[code], int(key[block*17]) - 127);
            }
        }
        sum = 0;
        for (int ih = 0; ih < args.heads; ++ih) {
            device const float * query = (device const float *) (q + it*args.nb_q2 + ih*args.nb_q1);
            float dot = 0;
            if (args.dim == 128) {
                dot += query[tiisg     ]*decoded[0];
                dot += query[tiisg + 32]*decoded[1];
                dot += query[tiisg + 64]*decoded[2];
                dot += query[tiisg + 96]*decoded[3];
            } else {
                for (int j = tiisg; j < args.dim; j += 32) {
                    const int block = j/32;
                    const uint code = (key[block*17 + 1 + tiisg % 16] >> (tiisg/16*4)) & 15;
                    const float value = dsv41_scale_pow2(kvalues_mxfp4_f[code], int(key[block*17]) - 127);
                    dot += query[j]*value;
                }
            }
            dot = dsv41_round_bf16(simd_sum(dot));
            sum += dsv41_round_bf16(max(dot, 0.0f)*w[ih]);
        }
    }
    return dsv41_round_bf16(sum);
}

kernel void kernel_dsv41_index_scores(
        constant ggml_metal_kargs_dsv41_index_scores & args,
        device const char * q,
        device const char * keys,
        device const char * weights,
        device const char * positions,
        device const char * candidates,
        device      float * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    const int slot = tgpig.x*ntg.y + sgitg, it = tgpig.y;
    if (slot >= args.n_scores) { return; }
    const float sum = dsv41_index_score(args, q, keys, weights, positions, candidates, slot, it, tiisg);
    if (tiisg == 0) { dst[it*args.n_scores + slot] = sum; }
}

kernel void kernel_dsv41_index_query_bounds(
        constant ggml_metal_kargs_dsv41_index_scores & args,
        device const char * q,
        device        int * dst,
        uint    it[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    threadgroup int bounds[8];
    const int tid = 32*sgitg + tiisg;
    int qmax = -1024, qmin = 1024;
    for (int i = tid; i < 32*128; i += 128) {
        const float value = *((device const float *) (q + it*args.nb_q2 + (i/128)*args.nb_q1) + i % 128);
        const uint bits = as_type<uint>(value) & 0x7fffffffu;
        if (bits) {
            const int exp = int(bits >> 23);
            const uint mantissa = (bits & 0x7fffffu) | (exp ? 0x800000u : 0u);
            qmax = max(qmax, exp == 255 ? 1024 : max(exp, 1) - 126);
            qmin = min(qmin, exp == 255 ? -1024 : max(exp, 1) - 150 + int(ctz(mantissa)));
        }
    }
    qmax = simd_max(qmax); qmin = simd_min(qmin);
    if (tiisg == 0) { bounds[sgitg*2] = qmax; bounds[sgitg*2 + 1] = qmin; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (int i = 0; i < 4; ++i) {
            qmax = max(qmax, bounds[i*2]); qmin = min(qmin, bounds[i*2 + 1]);
        }
        dst[it*2] = qmax; dst[it*2 + 1] = qmin;
    }
}

kernel void kernel_dsv41_index_scores_matrix(
        constant ggml_metal_kargs_dsv41_index_scores & args,
        device const char * q,
        device const char * keys,
        device const char * weights,
        device const char * positions,
        device const char * candidates,
        device      float * dst,
        device const int * query_bounds,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    constexpr int NK = 32;
    threadgroup float scores[32*NK];
    threadgroup half * sq = (threadgroup half *) scores;
    threadgroup half * sk = sq + 32*32;
    const int tid = 32*sgitg + tiisg, it = tgpig.y;
    const int pos = *(device const int *) (positions + it*args.nb_p0);
    threadgroup int bounds[8];
    const int qmax = query_bounds[it*2], qmin = query_bounds[it*2 + 1];
    int kmax = -1024, kmin = 1024;
    for (int i = tid; i < NK*4; i += 128) {
        const int slot = tgpig.x*NK + i/4;
        long ik = slot;
        if (slot < args.n_scores && args.n_candidates) {
            const int id = *(device const int *) (candidates + it*args.nb_c1 + (slot/args.block_size)*args.nb_c0);
            ik = long(id)*args.block_size + slot % args.block_size;
        }
        if (slot < args.n_scores && ik >= 0 && ik < args.n_keys && ik < (long(pos) + 1)/args.ratio) {
            device const uchar * key = (device const uchar *) (keys + ik*args.nb_k1);
            const int exp = int(key[(i % 4)*17]) - 127;
            kmax = max(kmax, exp + 3);
            kmin = min(kmin, exp - 1);
        }
    }
    kmax = simd_max(kmax); kmin = simd_min(kmin);
    if (tiisg == 0) {
        bounds[sgitg*2] = kmax; bounds[sgitg*2 + 1] = kmin;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < 4; ++i) {
        kmax = max(kmax, bounds[i*2]); kmin = min(kmin, bounds[i*2 + 1]);
    }
    // Half inputs must be exact; every partial dot sum must fit in 24 F32 bits.
    const bool exact = qmax <= 16 && kmax <= 16 && qmin >= -14 && kmin >= -14 && qmax - qmin <= 11 && kmax - kmin <= 11 && qmax + kmax + 7 - qmin - kmin <= 24;
    if (!exact) {
        for (int i = sgitg; i < NK; i += 4) {
            const int slot = tgpig.x*NK + i;
            if (slot < args.n_scores) {
                const float sum = dsv41_index_score(args, q, keys, weights, positions, candidates, slot, it, tiisg);
                if (tiisg == 0) { dst[it*args.n_scores + slot] = sum; }
            }
        }
        return;
    }
    simdgroup_float8x8 accum[NK/8];
    for (int i = 0; i < NK/8; ++i) { accum[i] = make_filled_simdgroup_matrix<float, 8>(0.0f); }
    for (int block = 0; block < 4; ++block) {
        for (int i = tid; i < 32*32; i += 128) {
            sq[i] = *((device const float *) (q + it*args.nb_q2 + (i/32)*args.nb_q1) + block*32 + i % 32);
        }
        for (int i = tid; i < NK*32; i += 128) {
            const int slot = tgpig.x*NK + i/32, lane = i % 32;
            long ik = slot;
            if (slot < args.n_scores && args.n_candidates) {
                const int id = *(device const int *) (candidates + it*args.nb_c1 + (slot/args.block_size)*args.nb_c0);
                ik = long(id)*args.block_size + slot % args.block_size;
            }
            float value = 0;
            if (slot < args.n_scores && ik >= 0 && ik < args.n_keys && ik < (long(pos) + 1)/args.ratio) {
                device const uchar * key = (device const uchar *) (keys + ik*args.nb_k1);
                const uint code = (key[block*17 + 1 + lane % 16] >> (lane/16*4)) & 15;
                value = dsv41_scale_pow2(kvalues_mxfp4_f[code], int(key[block*17]) - 127);
            }
            sk[i] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = 0; k < 32; k += 8) {
            simdgroup_half8x8 a, b;
            simdgroup_load(a, sq + sgitg*8*32 + k, 32, 0, false);
            for (int i = 0; i < NK/8; ++i) {
                simdgroup_load(b, sk + i*8*32 + k, 32, 0, true);
                simdgroup_multiply_accumulate(accum[i], a, b, accum[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The last tile barrier allows scores to reuse the query/key staging memory.
    for (int i = 0; i < NK/8; ++i) { simdgroup_store(accum[i], scores + sgitg*8*NK + i*8, NK, 0, false); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NK) {
        const int slot = tgpig.x*NK + tid;
        if (slot >= args.n_scores) { return; }
        long ik = slot;
        if (args.n_candidates) {
            const int id = *(device const int *) (candidates + it*args.nb_c1 + (slot/args.block_size)*args.nb_c0);
            ik = long(id)*args.block_size + slot % args.block_size;
        }
        float sum = -INFINITY;
        if (ik >= 0 && ik < args.n_keys && ik < (long(pos) + 1)/args.ratio) {
            device const float * w = (device const float *) (weights + it*args.nb_w1);
            sum = 0;
            for (int ih = 0; ih < 32; ++ih) {
                const float dot = dsv41_round_bf16(scores[ih*NK + tid]);
                sum += dsv41_round_bf16(max(dot, 0.0f)*w[ih]);
            }
        }
        dst[it*args.n_scores + slot] = dsv41_round_bf16(sum);
    }
}

static uint dsv41_select_key(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7fffffffu) == 0) { bits = 0; }
    return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}

static float dsv41_select_value(device const float * scores, int i, int visible, int block_size) {
    if (block_size == 0) { return scores[i]; }
    if (i == (visible - 1)/block_size) { return INFINITY; }
    float value = -INFINITY;
    for (int j = i*block_size; j < min((i + 1)*block_size, visible); ++j) {
        if (dsv41_select_key(scores[j]) > dsv41_select_key(value)) { value = scores[j]; }
    }
    return value;
}

static void dsv41_select_ids(
        device const float * scores, device int * out, int visible, int block_size, uint top_k, uint tid,
        int min_shift,
        threadgroup atomic_uint * hist, threadgroup uint * counts, threadgroup uint * ties, threadgroup uint * shared) {
    const uint n = block_size ? (visible + block_size - 1)/block_size : visible;
    if (n == 0) { return; }
    uint prefix = 0, desired = min(top_k, n);
    for (int shift = 28; shift >= min_shift; shift -= 4) {
        if (tid < 16) { atomic_store_explicit(hist + tid, 0u, memory_order_relaxed); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint local[16] = {};
        const uint mask = shift == 28 ? 0u : 0xffffffffu << (shift + 4);
        for (uint i = tid; i < n; i += 256) {
            const uint key = dsv41_select_key(dsv41_select_value(scores, i, visible, block_size));
            if ((key & mask) == prefix) { ++local[(key >> shift) & 15]; }
        }
        for (uint b = 0; b < 16; ++b) {
            if (local[b]) { atomic_fetch_add_explicit(hist + b, local[b], memory_order_relaxed); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint above = 0;
            for (int b = 15; b >= 0; --b) {
                const uint count = atomic_load_explicit(hist + b, memory_order_relaxed);
                if (above + count >= desired) { shared[0] = b; shared[1] = above; break; }
                above += count;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix |= shared[0] << shift;
        desired -= shared[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint key_mask = min_shift ? 0xffffffffu << min_shift : 0xffffffffu;
    const uint begin = tid*((n + 255)/256), end = min(begin + (n + 255)/256, n);
    uint above = 0, equal = 0;
    for (uint i = begin; i < end; ++i) {
        const float value = dsv41_select_value(scores, i, visible, block_size);
        const uint key = dsv41_select_key(value) & key_mask;
        above += value > -INFINITY && key > prefix;
        equal += value > -INFINITY && key == prefix;
    }
    counts[tid] = above;
    ties[tid] = equal;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint n_above = 0, n_equal = 0;
        for (uint i = 0; i < 256; ++i) {
            n_above += counts[i];
            const uint count = ties[i];
            ties[i] = n_equal;
            n_equal += count;
        }
        shared[0] = top_k - n_above;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint tie_limit = shared[0], tie_start = ties[tid];
    counts[tid] = above + min(equal, tie_limit > tie_start ? tie_limit - tie_start : 0u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint offset = 0;
        for (uint i = 0; i < 256; ++i) {
            const uint count = counts[i];
            counts[i] = offset;
            offset += count;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint offset = counts[tid], tie = tie_start;
    for (uint i = begin; i < end; ++i) {
        const float value = dsv41_select_value(scores, i, visible, block_size);
        const uint key = dsv41_select_key(value) & key_mask;
        if (value > -INFINITY && (key > prefix || (key == prefix && tie++ < tie_limit))) { out[offset++] = i; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

kernel void kernel_dsv41_select(
        constant ggml_metal_kargs_dsv41_select & args,
        device const char * scores,
        device const char * positions,
        device const char * candidates,
        device        int * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tid[[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint hist[16];
    threadgroup uint counts[256], ties[256], shared[2];
    const int it = tgpig.x;
    const int pos = *(device const int *) (positions + it*args.nb_p0);
    const long n_visible = max((long(pos) + 1)/args.ratio, 0l);
    int visible = int(min(n_visible, long(args.n_keys)));
    if (args.n_candidates) {
        int lo = 0, hi = args.n_candidates;
        while (lo < hi) {
            const int mid = (lo + hi)/2;
            const int id = *(device const int *) (candidates + it*args.nb_c1 + mid*args.nb_c0);
            if (id >= 0 && id < n_visible/args.block_size) { lo = mid + 1; } else { hi = mid; }
        }
        const int id = lo < args.n_candidates ? *(device const int *) (candidates + it*args.nb_c1 + lo*args.nb_c0) : -1;
        const long tail = id == n_visible/args.block_size ? n_visible % args.block_size : 0;
        visible = int(min(long(args.n_keys), long(lo)*args.block_size + tail));
    }
    device const float * row = (device const float *) (scores + it*args.nb_s1);
    device int * out = dst + it*(args.top_k + args.top_k_blocks);
    for (int i = tid; i < args.top_k + args.top_k_blocks; i += 256) { out[i] = -1; }
    threadgroup_barrier(mem_flags::mem_device);
    const int min_shift = args.score_bf16 ? 16 : 0;
    dsv41_select_ids(row, out, visible, 0, args.top_k, tid, min_shift, hist, counts, ties, shared);
    if (args.n_candidates) {
        threadgroup_barrier(mem_flags::mem_device);
        for (int i = tid; i < args.top_k; i += 256) {
            const int slot = out[i];
            if (slot < 0) { continue; }
            const int id = *(device const int *) (candidates + it*args.nb_c1 + (slot/args.block_size)*args.nb_c0);
            out[i] = id*args.block_size + slot % args.block_size;
        }
    }
    if (args.candidate_source) {
        dsv41_select_ids(row, out + args.top_k, visible, args.block_size, args.top_k_blocks, tid, min_shift, hist, counts, ties, shared);
    }
}

static float2 dsv41_pair_add(float2 a, float2 b) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    const float sum = a.x + b.x, v = sum - a.x;
    const float error = (a.x - (sum - v)) + (b.x - v) + (a.y + b.y);
    const float hi = sum + error;
    return float2(hi, error - (hi - sum));
}

static float2 dsv41_pair_mul(float2 a, float2 b) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    const float product = a.x*b.x;
    const float error = fma(a.x, b.x, -product) + (a.x*b.y + a.y*b.x);
    const float hi = product + error;
    return float2(hi, error - (hi - product));
}

// Extra precision keeps exp error from crossing BF16 probability and output boundaries.
static float dsv41_exp(float x) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    if (isnan(x)) { return x; }
    if (x > 89.0f) { return INFINITY; }
    if (x < -104.0f) { return 0.0f; }
    const float n = rint(x*0x1.715476p+0f);
    const float2 r = dsv41_pair_add(float2(x, 0), dsv41_pair_mul(float2(-n, 0), float2(0x1.62e4300000000p-1f, -0x1.05c6100000000p-29f)));
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
    for (int i = 11; i >= 0; --i) { value = dsv41_pair_add(dsv41_pair_mul(value, r), coefficients[i]); }
    if (n <= -126) {
        const float2 scaled = float2(dsv41_scale_pow2(value.x, int(n) + 149), dsv41_scale_pow2(value.y, int(n) + 149));
        const float base = floor(scaled.x), fraction = scaled.x - base;
        uint bits = uint(base);
        bits += fraction > 0.5f || (fraction == 0.5f && (scaled.y > 0 || (scaled.y == 0 && (bits & 1))));
        return as_type<float>(bits);
    }
    return dsv41_scale_pow2(value.x, int(n));
}

#if defined(GGML_METAL_HAS_BF16)
kernel void kernel_dsv41_attn_matrix_register(
        constant ggml_metal_kargs_dsv41_attn & args,
        device const char * q,
        device const char * raw,
        device const char * sinks,
        device const char * positions,
        device const char * indices,
        device const char * kv,
        device float * dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    threadgroup bfloat queries[8*64];
    threadgroup float storage[2048], stats[8*3];
    threadgroup bfloat * keys = (threadgroup bfloat *) storage;
    threadgroup bfloat * weights = queries;
    threadgroup float * scores = storage;
    float output[32] = {};
    threadgroup int ids[64];
    const int head = tgpig.x*8, it = tgpig.y;
    const int pos = *(device const int *) (positions + it*args.nb_p0);
    const bool prefill = *(device const int *) positions == 0;
    const int window = prefill ? min(args.tokens, args.window) : args.window;
    const int topk = args.ratio ? int(clamp((long(pos) + 1)/args.ratio, 0l, long(args.top_k))) : 0;
    if (tid < 8) { stats[tid*3] = -1e30f; stats[tid*3 + 1] = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int start = 0; start < window + topk; start += 64) {
        const int count = min(64, window + topk - start);
        if (tid < 64) {
            int id = -1;
            if (tid < count) {
                if (start + tid < window) {
                    const int first = pos - args.window + 1;
                    const int p = (prefill ? max(0, first) : first) + start + tid;
                    if (p >= args.window_start && p <= pos) { id = p % args.n_ring; }
                } else {
                    id = ((device const int *) (indices + it*args.nb_i1))[start + tid - window];
                    if (id < 0 || id >= args.n_kv || id >= (long(pos) + 1)/args.ratio) { id = -1; }
                }
            }
            ids[tid] = id;
        }
        simdgroup_float8x8 dots[2];
        dots[0] = make_filled_simdgroup_matrix<float, 8>(0.f);
        dots[1] = make_filled_simdgroup_matrix<float, 8>(0.f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int base = 0; base < args.dim; base += 64) {
            for (int i = tid; i < 8*64; i += 128) {
                const int ih = head + i/64;
                device const float * input = (device const float *) (q + ih*args.nb_q1 + it*args.nb_q2);
                queries[i] = ih < args.heads ? bfloat(input[base + i%64]) : bfloat(0);
            }
            for (int i = tid; i < 64*64; i += 128) {
                const int row_id = ids[i/64];
                const bool window_row = start + i/64 < window;
                device const float * row = (device const float *) ((window_row ? raw : kv) + max(row_id, 0)*(window_row ? args.nb_r1 : args.nb_k1));
                keys[i] = row_id >= 0 ? bfloat(row[base + i%64]) : bfloat(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int j = 0; j < 64; j += 8) {
                simdgroup_bfloat8x8 a, b;
                simdgroup_load(a, queries + j, 64);
                simdgroup_load(b, keys + sgitg*16*64 + j, 64, 0, true);
                simdgroup_multiply_accumulate(dots[0], a, b, dots[0]);
                simdgroup_load(b, keys + (sgitg*16 + 8)*64 + j, 64, 0, true);
                simdgroup_multiply_accumulate(dots[1], a, b, dots[1]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_store(dots[0], scores + sgitg*16, 64);
        simdgroup_store(dots[1], scores + sgitg*16 + 8, 64);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < 8*64; i += 128) { scores[i] = ids[i%64] >= 0 ? scores[i]*args.scale : -INFINITY; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 8) {
            float maximum = stats[tid*3];
            for (int slot = 0; slot < count; ++slot) { maximum = max(maximum, scores[tid*64 + slot]); }
            stats[tid*3 + 2] = fast::exp(stats[tid*3] - maximum);
            stats[tid*3] = maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tid; i < 8*64; i += 128) {
            const float probability = fast::exp(scores[i] - stats[(i/64)*3]);
            scores[i] = probability;
            weights[i] = bfloat(dsv41_round_bf16(probability));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 8) {
            float sum = 0;
            for (int slot = 0; slot < count; ++slot) { sum += scores[tid*64 + slot]; }
            stats[tid*3 + 1] = stats[tid*3 + 1]*stats[tid*3 + 2] + sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int base = 0; base < args.dim; base += 64) {
            for (int i = tid; i < 64*64; i += 128) {
                const int row_id = ids[i/64];
                const bool window_row = start + i/64 < window;
                device const float * row = (device const float *) ((window_row ? raw : kv) + max(row_id, 0)*(window_row ? args.nb_r1 : args.nb_k1));
                keys[i] = row_id >= 0 ? bfloat(row[base + i%64]) : bfloat(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 values[2];
            values[0] = make_filled_simdgroup_matrix<float, 8>(0.f);
            values[1] = make_filled_simdgroup_matrix<float, 8>(0.f);
            for (int j = 0; j < 64; j += 8) {
                simdgroup_bfloat8x8 a, b;
                simdgroup_load(a, weights + j, 64);
                simdgroup_load(b, keys + j*64 + sgitg*16, 64);
                simdgroup_multiply_accumulate(values[0], a, b, values[0]);
                simdgroup_load(b, keys + j*64 + sgitg*16 + 8, 64);
                simdgroup_multiply_accumulate(values[1], a, b, values[1]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_store(values[0], scores + sgitg*16, 64);
            simdgroup_store(values[1], scores + sgitg*16 + 8, 64);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = tid; i < 8*64; i += 128) {
                const int index = (base/64)*4 + i/128;
                output[index] = output[index]*stats[(i/64)*3 + 2] + scores[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid < 8 && head + tid < args.heads) {
        const float sink = *(device const float *) (sinks + (head + tid)*args.nb_s0);
        stats[tid*3 + 1] += fast::exp(sink - stats[tid*3]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int base = 0; base < 8; ++base) {
        for (int k = 0; k < 4; ++k) {
            const int i = tid + 128*k, ih = i/64, j = base*64 + i%64;
            if (head + ih < args.heads) {
                dst[(it*args.heads + head + ih)*args.dim + j] = dsv41_round_bf16(precise::divide(output[base*4 + k], stats[ih*3 + 1]));
            }
        }
    }
}
#endif

kernel void kernel_dsv41_pool(
        constant ggml_metal_kargs_dsv41_pool & args,
        device const char * kv,
        device const char * scores,
        device const char * positions,
        device      float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tid[[thread_index_in_threadgroup]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    const int j = tgpig.x*128 + tid, it = tgpig.y;
    if (j >= args.dim) { return; }
    const int pos = *(device const int *) (positions + it*args.nb_p0);
    float result = 0;
    if (pos >= 0 && pos % 2) {
        const int prev = (pos - 1) % args.n_rows, curr = pos % args.n_rows;
        const float a = ((device const float *) (scores + prev*args.nb_s1))[j];
        const float b = ((device const float *) (scores + curr*args.nb_s1))[j];
        const float maximum = max(a, b);
        const float ea = dsv41_exp(a - maximum), eb = dsv41_exp(b - maximum);
        const float va = ((device const float *) (kv + prev*args.nb_k1))[j];
        const float vb = ((device const float *) (kv + curr*args.nb_k1))[j];
        result = va*precise::divide(ea, ea + eb) + vb*precise::divide(eb, ea + eb);
    }
    dst[it*args.dim + j] = dsv41_round_bf16(result);
}

static float dsv41_attn_value(device const uchar * row, int j, bool raw) {
    if (raw) {
        const int block = j/32;
        return dsv41_scale_pow2(dsv41_e4m3_value(row[block*33 + 1 + j % 32]), int(row[block*33]) - 127);
    }
    row += (j/64)*36;
    const int sub = (j % 64)/16;
    const uint code = (row[4 + sub*8 + j % 8] >> ((j % 16)/8*4)) & 15;
    return kvalues_mxfp4_f[code]*dsv41_e4m3_value(row[sub]);
}

kernel void kernel_dsv41_attn_pack(
        constant ggml_metal_kargs_dsv41_attn & args,
        device const char * raw,
        device const char * positions,
        device const char * indices,
        device const char * kv,
        device half * dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const int row = tgpig.x*(ntg.x/32) + sgitg, it = tgpig.y;
    const int rows = args.window + args.top_k;
    if (row >= rows) { return; }
    const int pos = *(device const int *) (positions + it*args.nb_p0);
    const bool is_raw = row < args.window;
    int id;
    if (is_raw) {
        const int p = pos - args.window + 1 + row;
        id = p >= args.window_start && p <= pos ? p % args.n_ring : -1;
    } else {
        id = ((device const int *) (indices + it*args.nb_i1))[row - args.window];
        if (id < 0 || id >= args.n_kv || id >= (long(pos) + 1)/args.ratio) { id = -1; }
    }
    device const uchar * input = (device const uchar *) ((is_raw ? raw : kv) + max(id, 0)*(is_raw ? args.nb_r1 : args.nb_k1));
    device half * out = dst + (it*rows + row)*args.dim;
    for (int j = tiisg; j < args.dim; j += 32) {
        const float value = id >= 0 ? dsv41_attn_value(input, j, is_raw) : 0;
        out[j] = half(value);
    }
}

kernel void kernel_dsv41_attn_mask(
        constant ggml_metal_kargs_dsv41_attn & args,
        device const char * positions,
        device const char * indices,
        device half * dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const int row = tgpig.x*ntg.x + tid, it = tgpig.y;
    const int rows = args.window + args.top_k;
    if (row >= rows) { return; }
    const int pos = *(device const int *) (positions + it*args.nb_p0);
    bool valid;
    if (row < args.window) {
        const int p = pos - args.window + 1 + row;
        valid = p >= args.window_start && p <= pos;
    } else {
        const int id = ((device const int *) (indices + it*args.nb_i1))[row - args.window];
        valid = id >= 0 && id < args.n_kv && id < (long(pos) + 1)/args.ratio;
    }
    dst[it*rows + row] = valid ? half(0.0f) : half(-INFINITY);
}

static float dsv41_attn_unpack_value(
        constant ggml_metal_kargs_dsv41_attn & args,
        device const uchar * raw, device const uchar * kv, device const int * indices,
        int last, int row, int j) {
    const int raw_rows = args.selected ? args.window : args.n_ring;
    const bool is_raw = row < raw_rows;
    int id = is_raw ? row : row - raw_rows;
    if (args.selected) {
        if (is_raw) {
            const int first = last - args.window + 1;
            const int pos = (last == 0 ? max(0, first) : first) + id;
            id = pos >= args.window_start && pos <= last ? pos % args.n_ring : -1;
        } else {
            id = indices[id];
            if (id < 0 || id >= args.n_kv || id >= (long(last) + 1)/args.ratio) { id = -1; }
        }
    }
    if (id < 0) { return 0; }
    device const uchar * input = is_raw ? raw + id*args.nb_r1 : kv + id*args.nb_k1;
    return dsv41_attn_value(input, j, is_raw);
}

kernel void kernel_dsv41_attn_unpack(
        constant ggml_metal_kargs_dsv41_attn & args,
        device const uchar * raw,
        device const uchar * kv,
        device float * dst,
        device const char * positions,
        device const int * indices,
        uint tid [[thread_position_in_grid]],
        uint groups [[threadgroups_per_grid]]) {
    const int last = *(device const int *) (positions + (args.tokens - 1)*args.nb_p0);
    const int visible = args.ratio ? clamp((last + 1)/args.ratio, 0, args.selected ? args.top_k : args.n_kv) : 0;
    const int raw_rows = args.selected ? args.window : args.n_ring;
    const uint values = uint(raw_rows + visible)*args.dim;
    if (args.transposed && visible >= 256) {
        threadgroup float tile[16*17];
        const uint lane = tid % 256, x = lane % 16, y = lane/16;
        const int columns = args.dim/16, rows = args.window + args.top_k;
        const uint tiles = ((raw_rows + visible + 15)/16)*columns;
        for (uint i = tid/256; i < tiles; i += groups) {
            const int first_row = i/columns*16, first_column = i%columns*16;
            const int row = first_row + y, j = first_column + x;
            const float value = row < raw_rows + visible ? dsv41_attn_unpack_value(args, raw, kv, indices, last, row, j) : 0;
            tile[y*17 + x] = value;
            if (row < raw_rows + visible) { dst[row*args.dim + j] = value; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (first_row + x < raw_rows + visible) {
                dst[rows*(args.dim + args.heads) + (first_column + y)*rows + first_row + x] = tile[x*17 + y];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        return;
    }
    for (uint i = tid; i < values; i += groups*256) {
        const uint row = i/args.dim, j = i%args.dim;
        dst[i] = dsv41_attn_unpack_value(args, raw, kv, indices, last, row, j);
    }
}

kernel void kernel_dsv41_attn_unpack_rows(
        constant ggml_metal_kargs_dsv41_attn & args,
        device const uchar * raw,
        device const uchar * kv,
        device float * dst,
        device const char * positions,
        device const int * indices,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const int row = tgpig.x*(ntg.x/32) + sgitg;
    const int last = *(device const int *) (positions + (args.tokens - 1)*args.nb_p0);
    const int visible = args.ratio ? clamp((last + 1)/args.ratio, 0, args.top_k) : 0;
    if (row >= args.window + visible) { return; }
    const bool is_raw = row < args.window;
    int id = row - (is_raw ? 0 : args.window);
    if (is_raw) {
        const int first = last - args.window + 1;
        const int pos = (last == 0 ? max(0, first) : first) + id;
        id = pos >= args.window_start && pos <= last ? pos % args.n_ring : -1;
    } else {
        id = indices[id];
        if (id < 0 || id >= args.n_kv || id >= (long(last) + 1)/args.ratio) { id = -1; }
    }
    device const uchar * input = is_raw ? raw + max(id, 0)*args.nb_r1 : kv + max(id, 0)*args.nb_k1;
    for (int j = tiisg; j < args.dim; j += 32) { dst[row*args.dim + j] = id >= 0 ? dsv41_attn_value(input, j, is_raw) : 0; }
}

kernel void kernel_dsv41_attn_scores(
        constant ggml_metal_kargs_dsv41_attn & args,
        device const char * q,
        device const float * cache,
        device float * dst,
        device const char * positions,
        device const int * indices,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    threadgroup float query[512];
    const int head = tgpig.x, row = tgpig.y*64 + tid;
    device const float * input = (device const float *) (q + head*args.nb_q1);
    for (int j = tid; j < args.dim; j += 64) { query[j] = input[j]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int last = *(device const int *) positions;
    const int visible = args.ratio ? clamp((last + 1)/args.ratio, 0, args.top_k) : 0;
    if (row >= args.window + visible) { return; }
    if (row < args.window) {
        const int first = last - args.window + 1;
        const int pos = (last == 0 ? max(0, first) : first) + row;
        if (pos < args.window_start || pos > last) { return; }
    } else {
        const int id = indices[row - args.window];
        if (id < 0 || id >= args.n_kv || id >= (long(last) + 1)/args.ratio) { return; }
    }
    float dot = 0;
    if (args.transposed && visible >= 256) {
        const int rows = args.window + args.top_k;
        device const float * keys = cache + rows*(args.dim + args.heads);
        for (int j = 0; j < args.dim; ++j) { dot += query[j]*keys[j*rows + row]; }
    } else {
        for (int j = 0; j < args.dim; ++j) { dot += query[j]*cache[row*args.dim + j]; }
    }
    dst[head*(args.window + args.top_k) + row] = dot*args.scale;
}

kernel void kernel_dsv41_attn_scores_simd(
        constant ggml_metal_kargs_dsv41_attn & args,
        device const char * q,
        device const float * cache,
        device float * dst,
        device const char * positions,
        device const int * indices,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    threadgroup float query[512];
    const int head = tgpig.x, row = tgpig.y*(ntg.x/32) + sgitg;
    device const float * input = (device const float *) (q + head*args.nb_q1);
    for (int j = tid; j < args.dim; j += ntg.x) { query[j] = input[j]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int last = *(device const int *) positions;
    const int visible = args.ratio ? clamp((last + 1)/args.ratio, 0, args.top_k) : 0;
    if (row >= args.window + visible) { return; }
    if (row < args.window) {
        const int first = last - args.window + 1;
        const int pos = (last == 0 ? max(0, first) : first) + row;
        if (pos < args.window_start || pos > last) { return; }
    } else {
        const int id = indices[row - args.window];
        if (id < 0 || id >= args.n_kv || id >= (long(last) + 1)/args.ratio) { return; }
    }
    float2 dot = {0, 0};
    for (int j = tiisg; j < args.dim; j += 32) { dot = dsv41_pair_add(dot, float2(query[j]*cache[row*args.dim + j], 0)); }
    for (int offset = 16; offset > 0; offset /= 2) {
        dot = dsv41_pair_add(dot, float2(simd_shuffle_down(dot.x, offset), simd_shuffle_down(dot.y, offset)));
    }
    if (tiisg == 0) { dst[head*(args.window + args.top_k) + row] = (dot.x + dot.y)*args.scale; }
}

template <bool unpacked>
static float dsv41_attn_read(device const uchar * row, int j, bool raw) {
    return unpacked ? ((device const float *) row)[j] : dsv41_attn_value(row, j, raw);
}

template <bool unpacked>
kernel void kernel_dsv41_attn_impl(
        constant ggml_metal_kargs_dsv41_attn & args,
        device const char * q,
        device const char * raw,
        device const char * sinks,
        device const char * positions,
        device const char * indices,
        device const char * kv,
        device      float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tid[[thread_index_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    threadgroup float query[512], scores[64], weights[64], stats[3];
    threadgroup int ids[64];
    const int ih = tgpig.x, it = tgpig.y;
    const int width = unpacked && args.precomputed ? args.dim/args.slices : args.dim;
    const int first_channel = tgpig.z*width, end_channel = first_channel + width;
    const int pos = *(device const int *) (positions + it*args.nb_p0);
    const int last = *(device const int *) (positions + (args.tokens - 1)*args.nb_p0);
    const bool prefill = *(device const int *) positions == 0;
    const int window = prefill ? min(args.tokens, args.window) : args.window;
    const int topk = args.ratio ? int(clamp((long(last) + 1)/args.ratio, 0l, long(args.top_k))) : 0;
    device const float * input = (device const float *) (q + ih*args.nb_q1 + it*args.nb_q2);
    for (int j = tid; j < args.dim; j += ntg.x) { query[j] = input[j]; }
    float accumulator[8] = {};
    if (tid == 0) { stats[0] = -1e30f; stats[1] = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int start = 0; start < window + topk; start += 64) {
        const int count = min(64, window + topk - start);
        const bool is_raw = start + tid < window;
        int id = -1;
        if (tid < count) {
            if (is_raw) {
                const int first = pos - args.window + 1;
                const int p = (prefill ? max(0, first) : first) + start + tid;
                if (p >= args.window_start && p <= pos) { id = p % args.n_ring; }
            } else {
                id = ((device const int *) (indices + it*args.nb_i1))[start + tid - window];
                if (id < 0 || id >= args.n_kv || id >= (long(pos) + 1)/args.ratio) { id = -1; }
            }
        }
        if (unpacked && args.selected && id >= 0) { id = is_raw ? start + tid : start + tid - window; }
        if (tid < 64) { ids[tid] = id; }
        float dot = 0;
        if (id >= 0 && !(unpacked && args.precomputed)) {
            device const uchar * row = (device const uchar *) ((is_raw ? raw : kv) + id*(is_raw ? args.nb_r1 : args.nb_k1));
#pragma clang loop unroll_count(4)
            for (int j = 0; j < args.dim; ++j) { dot += query[j]*dsv41_attn_read<unpacked>(row, j, is_raw); }
        }
        if (tid < 64) { scores[tid] = id >= 0 ? dot*args.scale : -INFINITY; }
        if (unpacked && args.precomputed && id >= 0) {
            const int rows = args.window + args.top_k;
            device const float * cached_scores = (device const float *) raw + rows*args.dim;
            scores[tid] = cached_scores[ih*rows + (is_raw ? id : args.window + id)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float maximum = stats[0];
            for (int slot = 0; slot < count; ++slot) { maximum = max(maximum, scores[slot]); }
            stats[2] = dsv41_exp(stats[0] - maximum);
            stats[0] = maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 64) {
            const float probability = dsv41_exp(scores[tid] - stats[0]);
            scores[tid] = probability;
            weights[tid] = dsv41_round_bf16(probability);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float sum = 0;
            for (int slot = 0; slot < count; ++slot) { sum += scores[slot]; }
            stats[1] = stats[1]*stats[2] + sum;
        }
        for (int j = first_channel + tid; j < end_channel; j += ntg.x) {
            float value = 0;
#pragma clang loop unroll_count(8)
            for (int slot = 0; slot < count; ++slot) {
                const int row_id = ids[slot];
                if (row_id >= 0) {
                    const bool window_row = start + slot < window;
                    device const uchar * row = (device const uchar *) ((window_row ? raw : kv) + row_id*(window_row ? args.nb_r1 : args.nb_k1));
                    value += weights[slot]*dsv41_attn_read<unpacked>(row, j, window_row);
                }
            }
            const int channel = (j - first_channel)/ntg.x;
            accumulator[channel] = accumulator[channel]*stats[2] + value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        const float sink = *(device const float *) (sinks + ih*args.nb_s0);
        stats[1] += dsv41_exp(sink - stats[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = first_channel + tid; j < end_channel; j += ntg.x) {
        dst[(it*args.heads + ih)*args.dim + j] = dsv41_round_bf16(precise::divide(accumulator[(j - first_channel)/ntg.x], stats[1]));
    }
}

typedef decltype(kernel_dsv41_attn_impl<false>) kernel_dsv41_attn_t;
template [[host_name("kernel_dsv41_attn")]] kernel kernel_dsv41_attn_t kernel_dsv41_attn_impl<false>;
template [[host_name("kernel_dsv41_attn_unpacked")]] kernel kernel_dsv41_attn_t kernel_dsv41_attn_impl<true>;

kernel void kernel_dsv41_engram(
        constant ggml_metal_kargs_dsv41_engram & args,
        device const char * x,
        device const char * key,
        device const char * value,
        device const char * weight,
        device      float * dst,
        uint3    tgpig[[threadgroup_position_in_grid]],
        ushort   tiisg[[thread_index_in_simdgroup]],
        ushort   sgitg[[simdgroup_index_in_threadgroup]],
        ushort3    ntg[[threads_per_threadgroup]]) {
    const int row = tgpig.x*ntg.y + sgitg;
    if (row >= args.n_rows) { return; }
    const int ih = row % args.hc, it = row / args.hc;
    device const float * h = (device const float *) (x + ih*args.nb_x1 + it*args.nb_x2);
    device const float * k = (device const float *) (key + ih*args.nb_k1 + it*args.nb_k2);
    device const float * v = (device const float *) (value + it*args.nb_v1);
    device const float * w = (device const float *) (weight + ih*args.nb_w1);
    float h2 = 0, k2 = 0, dot = 0;
    for (int i = tiisg; i < args.dim; i += 32) {
        h2 += h[i]*h[i];
        k2 += k[i]*k[i];
        dot += (h[i]*w[i])*k[i];
    }
    h2 = simd_sum(h2);
    k2 = simd_sum(k2);
    dot = simd_sum(dot);
    const float rstd = rsqrt(h2/args.dim + args.eps)*rsqrt(k2/args.dim + args.eps);
    dot = (dot*rstd)*rsqrt(float(args.dim));
    const float gate = 1.0f/(1.0f + exp(-copysign(sqrt(max(abs(dot), args.clamp)), dot)));
    for (int i = tiisg; i < args.dim; i += 32) { dst[row*args.dim + i] = h[i] + gate*v[i]; }
}

template <bool mxfp8>
kernel void kernel_dsv41_swiglu_impl(
        constant ggml_metal_kargs_dsv41_swiglu & args,
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
    if (args.active_mask != 0 && (args.active_mask & (1u << i1)) == 0) { return; }
    device const float * g = (device const float *) (gate + i1*args.nb_g1 + i2*args.nb_g2 + i3*args.nb_g3);
    device const float * u = (device const float *) (up + i1*args.nb_u1 + i2*args.nb_u2 + i3*args.nb_u3);
    const float w = args.weighted ? *(device const float *) (weights + i1*args.nb_w1 + i2*args.nb_w2 + i3*args.nb_w3) : 1.0f;
    const float gate_value = args.input_bf16 ? dsv41_round_bf16(g[i]) : g[i];
    const float up_value = args.input_bf16 ? dsv41_round_bf16(u[i]) : u[i];
    const float a = args.limit > 0 ? min(gate_value, args.limit) : gate_value;
    const float b = args.limit > 0 ? clamp(up_value, -args.limit, args.limit) : up_value;
    const float silu = precise::divide(a, 1.0f + precise::exp(-a));
    uint bits = as_type<uint>((silu*b)*w);
    bits = (bits & 0x7fffffff) > 0x7f800000 ? bits | 0x400000 : bits + 0x7fff + ((bits >> 16) & 1);
    float result = as_type<float>(bits & 0xffff0000);
    if (mxfp8) {
        const float amax = simd_max(abs(result));
        const float scale = dsv41_e8m0_scale(max(amax, 1e-4f)*(1.0f/448.0f));
        const int exponent = int(as_type<uint>(scale) >> 23) - 127;
        const float value = dsv41_scale_pow2(result, -exponent);
        result = dsv41_scale_pow2(dsv41_e4m3_value(dsv41_e4m3_rne(value)), exponent);
    }
    dst[row*args.ne0 + i] = result;
}

typedef decltype(kernel_dsv41_swiglu_impl<false>) kernel_dsv41_swiglu_t;
template [[host_name("kernel_dsv41_swiglu")]] kernel kernel_dsv41_swiglu_t kernel_dsv41_swiglu_impl<false>;
template [[host_name("kernel_dsv41_swiglu_mxfp8")]] kernel kernel_dsv41_swiglu_t kernel_dsv41_swiglu_impl<true>;

kernel void kernel_dsv41_rope(
        constant ggml_metal_kargs_dsv41_rope & args,
        device const char * x,
        device const char * rotations,
        device      float * dst,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    const int i = 2*(128*tgpig.x + tiitg), row = tgpig.y;
    if (i >= args.dim) { return; }
    const int ih = row % args.heads, it = row / args.heads;
    device const float * src = (device const float *) (x + ih*args.nb_x1 + it*args.nb_x2);
    float a = src[i], b = src[i + 1];
    if (i >= args.offset) {
        device const float * cs = (device const float *) (rotations + it*args.nb_r1);
        const float c = cs[i - args.offset], s = args.sign*cs[i - args.offset + 1];
        a = fma(src[i], c, -src[i + 1]*s);
        b = fma(src[i + 1], c, src[i]*s);
    }
    uint2 bits = as_type<uint2>(float2(a, b));
    bits = select(bits + 0x7fff + ((bits >> 16) & 1), bits | 0x400000, (bits & 0x7fffffff) > 0x7f800000);
    *(device float2 *) (dst + row*args.dim + i) = as_type<float2>(bits & 0xffff0000);
}

kernel void kernel_dsv41_hc_split(
        constant ggml_metal_kargs_dsv41_hc_split & args,
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
        const float p = precise::divide(1.0f, 1.0f + dsv41_exp(-v));
        dst[24*it + tiisg] = tiisg < 4 ? p + args.eps : 2*p;
    }
    const ushort first = tiisg & ~3;
    const ushort column = 8 + tiisg % 4;
    const float maximum = max(max(simd_shuffle(v, first), simd_shuffle(v, first + 1)), max(simd_shuffle(v, first + 2), simd_shuffle(v, first + 3)));
    v = dsv41_exp(v - maximum);
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

kernel void kernel_dsv4_hc_pre_f32(
        constant ggml_metal_kargs_dsv4_hc_pre & args,
        device const char * x,
        device const char * weights,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float weight_lane = 0.0f;
    if (tiisg < hc) {
        weight_lane = *(device const float *) (weights + tiisg*args.nb_w0 + it*args.nb_w1);
    }

    float w[hc];
    FOR_UNROLL (ushort ih = 0; ih < hc; ++ih) {
        w[ih] = simd_shuffle(weight_lane, ih);
    }

    if (i0 >= args.n_embd) {
        return;
    }

    device const char * xb = x + i0*args.nb_x0 + it*args.nb_x2;
    float result = 0.0f;
    FOR_UNROLL (ushort ih = 0; ih < hc; ++ih) {
        result = fma(*(device const float *) (xb + ih*args.nb_x1), w[ih], result);
    }

    *(device float *) (dst + i0*args.nb_d0 + it*args.nb_d1) = result;
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

kernel void kernel_dsv41_hc_pre_norm_f32(
        constant ggml_metal_kargs_dsv4_hc_pre_norm & args,
        device const char * x,
        device const char * weights,
        device const char * norm,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint tgpig [[threadgroup_position_in_grid]],
        ushort tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    float weight_lane = 0;
    if (tiisg < 4) { weight_lane = *(device const float *) (weights + tiisg*args.nb_w0 + tgpig*args.nb_w1); }
    float4 values[2] = {};
    float sum = 0;
    for (int k = 0; k < 2; ++k) {
        const int i = tpitg + k*1024;
        if (i >= args.n_embd/4) { continue; }
        device const char * xb = x + 4*i*args.nb_x0 + tgpig*args.nb_x2;
        float4 result = 0;
        FOR_UNROLL (ushort ih = 0; ih < 4; ++ih) {
            result = fma(*(device const float4 *) (xb + ih*args.nb_x1), simd_shuffle(weight_lane, ih), result);
        }
        result = float4(dsv41_round_bf16(result.x), dsv41_round_bf16(result.y), dsv41_round_bf16(result.z), dsv41_round_bf16(result.w));
        values[k] = result;
        sum += dot(result, result);
    }
    sum = simd_sum(sum);
    if (tiisg == 0) { shmem_f32[sgitg] = sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = simd_sum(shmem_f32[tiisg]);
    const float scale = 1.0f/sqrt(sum/args.n_embd + args.eps);
    for (int k = 0; k < 2; ++k) {
        const int i = tpitg + k*1024;
        if (i >= args.n_embd/4) { continue; }
        const float4 result = (values[k]*scale)*(*(device const float4 *) (norm + 4*i*args.nb_n0));
        *(device float4 *) (dst + 4*i*args.nb_d0 + tgpig*args.nb_d1) = float4(dsv41_round_bf16(result.x), dsv41_round_bf16(result.y), dsv41_round_bf16(result.z), dsv41_round_bf16(result.w));
    }
}

template <bool add_x, bool residual_first = false>
kernel void kernel_dsv4_hc_post_f32_impl(
        constant ggml_metal_kargs_dsv4_hc_post & args,
        device const char * x,
        device const char * y,
        device const char * residual,
        device const char * post,
        device const char * comb,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float coeff_lane = 0.0f;
    if (tiisg < hc) {
        coeff_lane = *(device const float *) (post + tiisg*args.nb_p0 + it*args.nb_p1);
    } else if (tiisg < hc + hc*hc) {
        const ushort idx  = tiisg - hc;
        const ushort idst = idx & 3;
        const ushort isrc = idx >> 2;
        coeff_lane = *(device const float *) (comb + idst*args.nb_c0 + isrc*args.nb_c1 + it*args.nb_c2);
    }

    float post_reg[hc];
    float comb_reg[hc][hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        post_reg[idst] = simd_shuffle(coeff_lane, idst);
    }
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            comb_reg[isrc][idst] = simd_shuffle(coeff_lane, hc + idst + hc*isrc);
        }
    }

    if (i0 >= args.n_embd) {
        return;
    }

    float xv = *(device const float *) (x + i0*args.nb_x0 + it*args.nb_x1);
    if (add_x) {
        xv += *(device const float *) (y + i0*args.nb_x0 + it*args.nb_x1);
    }
    float result[hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        result[idst] = residual_first ? 0.0f : xv*post_reg[idst];
    }

    device const char * rb = residual + i0*args.nb_r0 + it*args.nb_r2;
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        const float rv = *(device const float *) (rb + isrc*args.nb_r1);
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            if (residual_first) {
                const float product = rv*comb_reg[isrc][idst];
                result[idst] = isrc == 0 ? product : result[idst] + product;
            } else {
                result[idst] = fma(rv, comb_reg[isrc][idst], result[idst]);
            }
        }
    }

    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        if (residual_first) {
            result[idst] = xv*post_reg[idst] + result[idst];
        }
        if (args.output_bf16) {
            result[idst] = dsv41_round_bf16(result[idst]);
        }
        *(device float *) (dst + i0*args.nb_d0 + idst*args.nb_d1 + it*args.nb_d2) = result[idst];
    }
}

typedef decltype(kernel_dsv4_hc_post_f32_impl<false>) kernel_dsv4_hc_post_f32_t;

template [[host_name("kernel_dsv4_hc_post_f32")]]     kernel kernel_dsv4_hc_post_f32_t kernel_dsv4_hc_post_f32_impl<false>;
template [[host_name("kernel_dsv4_hc_post_add_f32")]] kernel kernel_dsv4_hc_post_f32_t kernel_dsv4_hc_post_f32_impl<true>;
template [[host_name("kernel_dsv4_hc_post_ordered_f32")]] kernel kernel_dsv4_hc_post_f32_t kernel_dsv4_hc_post_f32_impl<false, true>;
template [[host_name("kernel_dsv4_hc_post_add_ordered_f32")]] kernel kernel_dsv4_hc_post_f32_t kernel_dsv4_hc_post_f32_impl<true, true>;

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

kernel void kernel_qwen4exp_hc_reduce_f32(
        constant ggml_metal_kargs_qwen4exp_hc_reduce & args,
        device const float * x,
        device const float * gate,
        device       float * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;
    if (i0 >= args.n_embd) {
        return;
    }

    const int offset = it*hc*args.n_embd + i0;
    float result = 0.0f;
    FOR_UNROLL (ushort ih = 0; ih < hc; ++ih) {
        const int idx = offset + ih*args.n_embd;
        float weight = gate[idx];
        if (args.gate_sigmoid) {
            weight = 1.0f/(1.0f + exp(-weight));
        }
        result = fma(x[idx], weight, result);
    }

    dst[it*args.n_embd + i0] = result*0.25f;
}

kernel void kernel_qwen4exp_hc_combine_f32(
        constant ggml_metal_kargs_qwen4exp_hc_combine & args,
        device const float * residual,
        device const float * x,
        device const float * injection,
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
        const float iv = injection[it*hc + tiisg]*0.25f;
        weight_lane = 2.0f/(1.0f + exp(-iv));
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

kernel void kernel_qsa_block_score_f32(
        constant ggml_metal_kargs_qsa_block_score & args,
        device const char * q,
        device const char * k,
        device const char * cells,
        device const char * mask,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {
    const int iq = tgpig.y;
    const int is = tgpig.z;
    device const char * q_row = q + iq*args.nb_q2 + is*args.nb_q3 + tiisg*sizeof(float4);
    float4 q_reg[OP_QSA_BLOCK_SCORE_NH];
    FOR_UNROLL (ushort ih = 0; ih < OP_QSA_BLOCK_SCORE_NH; ++ih) {
        q_reg[ih] = *(device const float4 *) (q_row + ih*args.nb_q1);
    }

    const int ib0 = (tgpig.x*ntg.y + sgitg)*OP_QSA_BLOCK_SCORE_NKPSG;
    FOR_UNROLL (ushort ik = 0; ik < OP_QSA_BLOCK_SCORE_NKPSG; ++ik) {
        const int ib = ib0 + ik;
        if (ib >= args.n_blocks) {
            return;
        }

        const int cell = *(device const int *) (cells + ib*sizeof(int) + iq*args.nb_c1 + is*args.nb_c3);
        device const float4 * k_row = (device const float4 *) (k + cell*args.nb_k1);
        const float4 kv = k_row[tiisg];
        float score = 0.0f;
        FOR_UNROLL (ushort ih = 0; ih < OP_QSA_BLOCK_SCORE_NH; ++ih) {
            score += max(simd_sum(dot(q_reg[ih], kv)), 0.0f);
        }

        if (tiisg == 0) {
            const float mv = *(device const float *) (mask + ib*sizeof(float) + iq*args.nb_m1 + is*args.nb_m3);
            *(device float *) (dst + ib*sizeof(float) + iq*args.nb_d1 + is*args.nb_d3) = score*args.scale + mv;
        }
    }
}
